import cv2
import pyglet
import zmq
import threading
import time
import heapq
import numpy as np
import os
import json
from queue import Queue, Empty
from typing import Optional
from trace_logger import TraceLogger
import re
from pyglet.gl import *

# Constants
CAPTURE_WIDTH = 1920
CAPTURE_HEIGHT = 1080
TARGET_SIZE = 1024
QUEUE_SIZE = 2

# Add OpenGL configuration for antialiasing and alpha blending
config = pyglet.gl.Config(
    double_buffer=True,
    sample_buffers=1,
    samples=4,
    alpha_size=8,
    depth_size=24
)

class WebcamApp:
    def __init__(self):
        # Initialize logger
        self.logger = TraceLogger("local", "webcam_display")
        
        # Add debug counters and timing
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.processed_count = 0  # Track processed frames
        
        # Initialize settings
        self.settings_file = 'settings.json'
        self.last_settings_mtime = 0
        self.load_settings()  # This will override the default if settings exist
        
        # Initialize threading events
        self.shutdown = threading.Event()
        
        # Initialize ZMQ context and sockets
        self.context = zmq.Context()
        
        # Socket for sending frames to workers
        self.distribute_socket = self.context.socket(zmq.PUSH)
        self.distribute_socket.set_hwm(2)  # Increase slightly to prevent blocking
        ipc_path = os.path.join(os.getcwd(), ".distribute_socket")
        self.distribute_socket.bind(f"ipc://{ipc_path}")
        self.distribute_socket.setsockopt(zmq.LINGER, 0)
        
        # Socket for receiving processed frames
        self.collect_socket = self.context.socket(zmq.PULL)
        ipc_path = os.path.join(os.getcwd(), ".collect_socket")
        self.collect_socket.bind(f"ipc://{ipc_path}")
        self.collect_socket.setsockopt(zmq.RCVTIMEO, 0)  # Non-blocking
        self.collect_socket.setsockopt(zmq.LINGER, 0)
        
        # Frame queues - increase size slightly
        self.frame_queue = Queue(maxsize=2)
        self.processed_queue = Queue(maxsize=2)
        
        # Initialize threads
        self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
        self.process_thread = threading.Thread(target=self.process_loop, daemon=True)
        
        # Create window with OpenGL config
        try:
            self.window = pyglet.window.Window(fullscreen=True, config=config, vsync=True)
        except pyglet.window.NoSuchConfigException:
            self.window = pyglet.window.Window(fullscreen=True, vsync=True)
        
        # Register event handlers
        self.window.event(self.on_draw)
        self.window.event(self.on_key_press)  # Register key press handler
        
        # Create circle shape
        self.batch = pyglet.graphics.Batch()
        self.circle = pyglet.shapes.Circle(
            x=self.mask_settings["x"],
            y=self.mask_settings["y"],
            radius=self.mask_settings["radius"],
            color=(0, 0, 0),
            batch=self.batch
        )
        
        # Frame storage
        self.current_texture = None
        self.processed_texture = None
        
        # Load prompts
        self.prompts = self.load_prompts()
        self.current_prompt_idx = 0
        self.last_prompt_change = time.time()
        
        # Schedule updates
        pyglet.clock.schedule_interval(self.update_frame, 1/60.0)
        pyglet.clock.schedule_interval(self.check_settings, 1.0)

    def load_prompts(self):
        try:
            with open('prompts.txt', 'r') as f:
                return [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            return ["A beautiful portrait"]

    def get_current_prompt(self):
        current_time = time.time()
        if current_time - self.last_prompt_change >= self.prompt_cycle_time:
            self.current_prompt_idx = (self.current_prompt_idx + 1) % len(self.prompts)
            self.last_prompt_change = current_time
        return self.prompts[self.current_prompt_idx]

    def load_settings(self):
        with open(self.settings_file, 'r') as f:
            settings = json.load(f)
            self.mask_settings = settings.get("mask", {
                "x": 941.0,
                "y": 215.0,
                "radius": 50
            })
            self.camera_fps = settings.get("camera_fps", 20)
            self.prompt_cycle_time = settings.get("prompt_cycle_time", 10)
            self.settings_show_processed = settings.get("show_processed", False)

    def check_settings(self, dt):
        try:
            # Only reload settings if file has changed
            try:
                mtime = os.path.getmtime(self.settings_file)
                if mtime <= self.last_settings_mtime:
                    return
                self.last_settings_mtime = mtime
            except OSError:
                return

            self.load_settings()
            # Update circle in main thread
            self.circle.x = self.mask_settings["x"]
            self.circle.y = self.mask_settings["y"]
            self.circle.radius = self.mask_settings["radius"]
        except Exception:
            pass

    @property
    def show_processed(self):
        if hasattr(self, 'user_show_processed'):
            return self.user_show_processed
        return self.settings_show_processed

    @show_processed.setter
    def show_processed(self, value):
        self.user_show_processed = value

    def capture_loop(self):
        cap = cv2.VideoCapture(1)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, self.camera_fps)
        
        last_send_time = 0
        send_interval = 1.0 / 30  # Limit sending to workers to 30fps
        
        try:
            while not self.shutdown.is_set():
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue

                # Crop and convert frame
                h, w = frame.shape[:2]
                start_x = (w - TARGET_SIZE) // 2
                start_y = (h - TARGET_SIZE) // 2
                cropped = frame[start_y:start_y+TARGET_SIZE, start_x:start_x+TARGET_SIZE]
                cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                
                # Put frame in display queue
                try:
                    if self.frame_queue.empty():  # Only update if empty
                        self.frame_queue.put_nowait(cropped)
                except:
                    pass
                
                # Send frame to workers at reduced rate
                current_time = time.time()
                if current_time - last_send_time >= send_interval:
                    try:
                        timestamp = str(current_time)
                        current_prompt = self.get_current_prompt()
                        frame_float = np.float32(cropped) / 255.0
                        
                        self.distribute_socket.send_multipart([
                            timestamp.encode(),
                            frame_float.tobytes(),
                            current_prompt.encode()
                        ], flags=zmq.DONTWAIT)
                        last_send_time = current_time
                    except zmq.Again:
                        pass  # Skip if can't send
                
        finally:
            cap.release()

    def process_loop(self):
        """Receive processed frames from workers"""
        last_debug = time.time()
        frames_received = 0
        
        while not self.shutdown.is_set():
            try:
                multipart_msg = self.collect_socket.recv_multipart(flags=zmq.NOBLOCK)
                frames_received += 1
                
                if len(multipart_msg) != 2:
                    continue
                    
                timestamp_bytes, frame_data = multipart_msg
                
                try:
                    # Convert back to numpy array
                    processed_frame = np.frombuffer(frame_data, dtype=np.uint8).reshape(TARGET_SIZE, TARGET_SIZE, 3)
                    
                    # Update processed queue
                    try:
                        # Clear old frames
                        while not self.processed_queue.empty():
                            try:
                                self.processed_queue.get_nowait()
                            except Empty:
                                break
                        
                        self.processed_queue.put_nowait(processed_frame)
                        self.processed_count += 1
                    except Full:
                        pass
                except Exception:
                    pass
                    
            except zmq.Again:
                time.sleep(0.001)  # Small sleep to prevent busy waiting
            except Exception:
                time.sleep(0.001)  # Sleep on error to prevent tight loop

    def update_frame(self, dt):
        try:
            # Always try to update raw frame texture
            frame = self.frame_queue.get_nowait()
            image = pyglet.image.ImageData(
                TARGET_SIZE, TARGET_SIZE,
                'RGB', frame.tobytes(),
                pitch=TARGET_SIZE * 3
            )
            if self.current_texture is not None:
                self.current_texture.delete()
            self.current_texture = image.get_texture().get_transform(flip_y=True, flip_x=True)
        except Empty:
            pass
        except Exception:
            pass
            
        # Update processed texture only if we're showing it
        if self.show_processed:
            try:
                processed = self.processed_queue.get_nowait()
                image = pyglet.image.ImageData(
                    TARGET_SIZE, TARGET_SIZE,
                    'RGB', processed.tobytes(),
                    pitch=TARGET_SIZE * 3
                )
                if self.processed_texture is not None:
                    self.processed_texture.delete()
                self.processed_texture = image.get_texture().get_transform(flip_y=True, flip_x=True)
            except Empty:
                pass
            except Exception:
                pass

    def on_draw(self):
        self.frame_count += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:
            fps = self.frame_count / (current_time - self.last_fps_time)
            print(f"FPS: {fps:.2f}")
            self.frame_count = 0
            self.processed_count = 0
            self.last_fps_time = current_time
        
        try:
            self.window.clear()
            
            # Choose which texture to display
            texture = None
            if self.show_processed:
                if self.processed_texture is not None:
                    texture = self.processed_texture
                else:
                    texture = self.current_texture
            else:
                texture = self.current_texture
            
            if texture is not None:
                texture.anchor_x = 0
                texture.anchor_y = 0
                window_width = self.window.width
                side = 1200
                x = (window_width - side) / 2
                texture.blit(x, 0, width=side, height=side)
            
            self.batch.draw()
        except Exception:
            pass

    def on_key_press(self, symbol, modifiers):
        if symbol == pyglet.window.key.ESCAPE:
            self.shutdown.set()
            pyglet.app.exit()
        elif symbol == pyglet.window.key.SPACE:
            self.show_processed = not self.show_processed

    def run(self):
        self.capture_thread.start()
        self.process_thread.start()
        try:
            pyglet.app.run()
        finally:
            self.shutdown.set()
            self.capture_thread.join()
            self.process_thread.join()
            self.context.destroy()
            self.collect_socket.close()
            self.distribute_socket.close()

if __name__ == '__main__':
    app = WebcamApp()
    app.run() 
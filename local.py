import cv2
import pyglet
from pyglet.gl import *
import zmq
import threading
import time
import numpy as np
import os
import json
from queue import Queue, Empty
from trace_logger import TraceLogger
import re

# Constants
CAPTURE_WIDTH = 1920
CAPTURE_HEIGHT = 1080
TARGET_SIZE = 1024
QUEUE_SIZE = 2

# OpenGL configuration for antialiasing and alpha blending
config = pyglet.gl.Config(
    double_buffer=True,
    sample_buffers=1,
    samples=4,
    alpha_size=8,
    depth_size=24
)

class WebcamApp:
    def __init__(self):
        # Initialize logger and debug counters
        self.logger = TraceLogger("local", "webcam_display")
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.processed_count = 0
        
        # Initialize settings
        self.settings_file = 'settings.json'
        self.last_settings_mtime = 0
        self.load_settings()
        
        # Initialize ZMQ context and sockets
        self.context = zmq.Context()
        self.setup_sockets()
        
        # Initialize queues with fixed size
        self.frame_queue = Queue(maxsize=QUEUE_SIZE)
        self.processed_queue = Queue(maxsize=QUEUE_SIZE)
        
        # Initialize threading events
        self.shutdown = threading.Event()
        self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
        self.process_thread = threading.Thread(target=self.process_loop, daemon=True)
        
        # Initialize window and graphics
        self.setup_window()
        
        # Initialize frame storage
        self.current_texture = None
        self.processed_texture = None
        
        # Load prompts and initialize prompt state
        self.prompts = self.load_prompts()
        self.current_prompt_idx = 0
        self.last_prompt_change = time.time()
        
        # Schedule updates
        pyglet.clock.schedule_interval(self.update_frame, 1/60.0)
        pyglet.clock.schedule_interval(self.check_settings, 1.0)

    def setup_sockets(self):
        # Socket for sending frames to workers
        self.distribute_socket = self.context.socket(zmq.PUSH)
        self.distribute_socket.set_hwm(QUEUE_SIZE)
        ipc_path = os.path.join(os.getcwd(), ".distribute_socket")
        self.distribute_socket.bind(f"ipc://{ipc_path}")
        self.distribute_socket.setsockopt(zmq.LINGER, 0)
        
        # Socket for receiving processed frames
        self.collect_socket = self.context.socket(zmq.PULL)
        ipc_path = os.path.join(os.getcwd(), ".collect_socket")
        self.collect_socket.bind(f"ipc://{ipc_path}")
        self.collect_socket.setsockopt(zmq.RCVTIMEO, 0)
        self.collect_socket.setsockopt(zmq.LINGER, 0)

    def setup_window(self):
        try:
            self.window = pyglet.window.Window(fullscreen=True, config=config, vsync=True)
        except pyglet.window.NoSuchConfigException:
            self.window = pyglet.window.Window(fullscreen=True, vsync=True)
        
        # Register event handlers
        self.window.event(self.on_draw)
        self.window.event(self.on_key_press)
        
        # Create shapes batch
        self.batch = pyglet.graphics.Batch()
        
        # Create circle shape
        self.circle = pyglet.shapes.Circle(
            x=self.mask_settings["x"],
            y=self.mask_settings["y"],
            radius=self.mask_settings["radius"],
            color=(0, 0, 0),
            batch=self.batch
        )
        
        # Calculate image position and size
        self.side = 1200  # Fixed image size
        window_width = self.window.width
        x = (window_width - self.side) / 2  # Left edge of image
        
        # Create keystone mask triangles
        self.left_mask = pyglet.shapes.Triangle(
            x, self.side,  # Top left (at image edge)
            x, 0,  # Bottom left
            x + self.keystone_mask, 0,  # Bottom right
            color=(0, 0, 0),  # Red color
            batch=self.batch
        )
        
        self.right_mask = pyglet.shapes.Triangle(
            x + self.side, self.side,  # Top right (at image edge)
            x + self.side, 0,  # Bottom right
            x + self.side - self.keystone_mask, 0,  # Bottom left
            color=(0, 0, 0),  # Red color
            batch=self.batch
        )

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
            self.keystone_mask = settings.get("keystone_mask", 100)

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
            
            # Update keystone masks
            window_width = self.window.width
            x = (window_width - self.side) / 2  # Left edge of image
            
            # Update left mask
            self.left_mask.x1 = x  # Top left x
            self.left_mask.x2 = x  # Bottom left x
            self.left_mask.x3 = x + self.keystone_mask  # Bottom right x
            
            # Update right mask
            self.right_mask.x1 = x + self.side  # Top right x
            self.right_mask.x2 = x + self.side  # Bottom right x
            self.right_mask.x3 = x + self.side - self.keystone_mask  # Bottom left x
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
        cap = cv2.VideoCapture(0)
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
                
                # Update display queue if empty
                if self.frame_queue.empty():
                    try:
                        self.frame_queue.put_nowait(cropped)
                    except Full:
                        pass
                
                # Send frame to workers at reduced rate
                current_time = time.time()
                if current_time - last_send_time >= send_interval:
                    try:
                        frame_float = np.float32(cropped) / 255.0
                        self.distribute_socket.send_multipart([
                            str(current_time).encode(),
                            frame_float.tobytes(),
                            self.get_current_prompt().encode()
                        ], flags=zmq.DONTWAIT)
                        last_send_time = current_time
                    except zmq.Again:
                        pass
        finally:
            cap.release()

    def process_loop(self):
        """Receive processed frames from workers"""
        while not self.shutdown.is_set():
            try:
                multipart_msg = self.collect_socket.recv_multipart()
                if len(multipart_msg) != 2:
                    continue
                    
                _, frame_data = multipart_msg
                processed_frame = np.frombuffer(frame_data, dtype=np.uint8).reshape(TARGET_SIZE, TARGET_SIZE, 3)
                
                # Update processed queue, clearing old frames first
                while not self.processed_queue.empty():
                    try:
                        self.processed_queue.get_nowait()
                    except Empty:
                        break
                
                try:
                    self.processed_queue.put_nowait(processed_frame)
                    self.processed_count += 1
                except Full:
                    pass
                    
            except zmq.Again:
                continue
            except Exception:
                continue

    def update_frame(self, dt):
        texture_to_update = None
        data_queue = None
        
        if self.show_processed:
            texture_to_update = self.processed_texture
            data_queue = self.processed_queue
        else:
            texture_to_update = self.current_texture
            data_queue = self.frame_queue
            
        try:
            frame = data_queue.get_nowait()
            image = pyglet.image.ImageData(
                TARGET_SIZE, TARGET_SIZE,
                'RGB', frame.tobytes(),
                pitch=TARGET_SIZE * 3
            )
            if texture_to_update is not None:
                texture_to_update.delete()
            new_texture = image.get_texture().get_transform(flip_y=True, flip_x=True)
            
            if self.show_processed:
                self.processed_texture = new_texture
            else:
                self.current_texture = new_texture
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
            
            # Choose texture to display
            texture = self.processed_texture if self.show_processed else self.current_texture
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
import cv2
import pyglet
from pyglet.gl import *
import zmq
import threading
import time
import numpy as np
import os
import json
from queue import Queue, Empty, Full
from trace_logger import TraceLogger
import re
import pygame  # Add pygame import

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
        pygame.mixer.init()
        
        # Initialize logger and debug counters
        self.logger = TraceLogger("local", "webcam_display")
        self.frame_count = 0
        self.last_fps_time = time.time()
        
        # Initialize settings
        self.settings_file = 'settings.json'
        self.last_settings_mtime = 0
        self.load_settings()
        
        # Initialize ZMQ context and sockets
        self.context = zmq.Context()
        self.setup_sockets()
        
        # Initialize queues with fixed size
        self.frame_queue = Queue(maxsize=QUEUE_SIZE)
        
        # Initialize threading events
        self.shutdown = threading.Event()
        self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
        
        # Initialize window and graphics
        self.setup_window()
        
        # Initialize frame storage
        self.current_texture = None
        self.processed_texture = None
        
        # Load prompts and initialize prompt state
        self.prompts = self.load_prompts()
        self.current_prompt_idx = 0
        self.last_prompt_change = None
        
        # Add white square toggle
        self.show_white_square = False
        
        # Load mask texture and track its modification time
        self.mask_file = 'mask.png'
        self.last_mask_mtime = os.path.getmtime(self.mask_file)
        mask_image = pyglet.image.load(self.mask_file)
        self.mask_texture = mask_image.get_texture()
        
        # Schedule updates
        pyglet.clock.schedule_interval(self.update_frame, 1/60.0)
        pyglet.clock.schedule_interval(self.check_settings, 1.0)

    def setup_sockets(self):
        # Socket for sending frames to workers
        self.distribute_socket = self.context.socket(zmq.PUSH)
        self.distribute_socket.set_hwm(QUEUE_SIZE)
        # Bind to all interfaces for network access
        self.distribute_socket.bind("tcp://*:5555")
        self.distribute_socket.setsockopt(zmq.LINGER, 0)
        
        # Socket for receiving processed frames
        self.collect_socket = self.context.socket(zmq.PULL)
        # Bind to all interfaces for network access
        self.collect_socket.bind("tcp://*:5556")
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

    def load_prompts(self):
        try:
            with open('prompts.txt', 'r') as f:
                return [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            return ["A beautiful portrait"]

    def get_current_prompt(self):
        current_time = time.time()
        if self.last_prompt_change is None or current_time - self.last_prompt_change >= self.prompt_cycle_time:
            self.current_prompt_idx = (self.current_prompt_idx + 1) % len(self.prompts)
            self.last_prompt_change = current_time
            
            # Play corresponding audio file when prompt changes
            try:
                audio_file = f"audio/{self.current_prompt_idx:02d}.wav"
                if os.path.exists(audio_file):
                    pygame.mixer.music.stop()
                    pygame.mixer.music.load(audio_file)
                    pygame.mixer.music.play()
            except Exception as e:
                self.logger.error(f"Error playing audio: {str(e)}")
            
        return self.prompts[self.current_prompt_idx]

    def load_settings(self):
        with open(self.settings_file, 'r') as f:
            settings = json.load(f)
            self.camera_fps = settings.get("camera_fps", 20)
            self.prompt_cycle_time = settings.get("prompt_cycle_time", 10)
            self.settings_show_processed = settings.get("show_processed", False)

    def check_settings(self, dt):
        try:
            # Check settings file
            try:
                settings_mtime = os.path.getmtime(self.settings_file)
                if settings_mtime > self.last_settings_mtime:
                    self.last_settings_mtime = settings_mtime
                    self.load_settings()
            except OSError:
                pass

            # Check mask file
            try:
                mask_mtime = os.path.getmtime(self.mask_file)
                if mask_mtime > self.last_mask_mtime:
                    self.last_mask_mtime = mask_mtime
                    # Reload mask texture
                    if self.mask_texture:
                        self.mask_texture.delete()
                    mask_image = pyglet.image.load(self.mask_file)
                    self.mask_texture = mask_image.get_texture()
            except OSError:
                pass

        except Exception as e:
            self.logger.error(f"Error in check_settings: {str(e)}")

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

    def update_frame(self, dt):
        # Try to receive processed frame from workers (non-blocking)
        try:
            multipart_msg = self.collect_socket.recv_multipart(flags=zmq.NOBLOCK)
            if len(multipart_msg) == 2:
                _, frame_data = multipart_msg
                processed_frame = np.frombuffer(frame_data, dtype=np.uint8).reshape(TARGET_SIZE, TARGET_SIZE, 3)
                
                # Update processed texture directly
                image = pyglet.image.ImageData(
                    TARGET_SIZE, TARGET_SIZE,
                    'RGB', processed_frame.tobytes(),
                    pitch=TARGET_SIZE * 3
                )
                
                if self.processed_texture is None:
                    self.processed_texture = image.get_texture().get_transform(flip_y=True, flip_x=True)
                else:
                    self.processed_texture.blit_into(image, 0, 0, 0)
        except zmq.Again:
            # No message available, continue with normal frame update
            pass
        except Exception:
            # Handle any other exceptions silently
            pass
        
        # Update current frame texture
        texture_to_update = self.current_texture
        data_queue = self.frame_queue
            
        try:
            frame = data_queue.get_nowait()
            image = pyglet.image.ImageData(
                TARGET_SIZE, TARGET_SIZE,
                'RGB', frame.tobytes(),
                pitch=TARGET_SIZE * 3
            )
            
            # Only recreate texture if it doesn't exist or if we need to update it
            if texture_to_update is None:
                self.current_texture = image.get_texture().get_transform(flip_y=True, flip_x=True)
            else:
                # Update existing texture data instead of recreating
                texture_to_update.blit_into(image, 0, 0, 0)
                
        except Empty:
            pass
        except Exception:
            pass

    def on_draw(self):
        self.frame_count += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:
            fps = self.frame_count / (current_time - self.last_fps_time)
            # if fps < 30:
            print(f"FPS: {fps:.2f}")
            self.frame_count = 0
            self.last_fps_time = current_time
            
        # return
        
        try:
            self.window.clear()
            
            # Cache window dimensions to avoid repeated property access
            window_width = self.window.width
            window_height = self.window.height
            side = 1200
            x = (window_width - side) / 2
            
            
            if self.show_white_square:
                # Draw white square
                white_square = pyglet.shapes.Rectangle(x=x, y=0, width=side, height=side, 
                                                     color=(255, 255, 255))
                white_square.draw()
            else:
                # Draw webcam preview
                texture = self.processed_texture if self.show_processed else self.current_texture
                if texture is not None:
                    # Draw the main texture (cache anchor settings)
                    if not hasattr(texture, '_anchors_set'):
                        texture.anchor_x = 0
                        texture.anchor_y = 0
                        texture._anchors_set = True
                    texture.blit(x, 0, width=side, height=side)
                    
        except Exception:
            pass
        
        # Draw the mask texture with multiply blend mode
        if self.mask_texture is not None:
            glEnable(GL_BLEND)
            glBlendFunc(GL_DST_COLOR, GL_ZERO)  # Multiply blend mode
            self.mask_texture.blit(0, 0, width=window_width, height=window_height)
            glDisable(GL_BLEND)

    def on_key_press(self, symbol, modifiers):
        if symbol == pyglet.window.key.ESCAPE:
            self.shutdown.set()
            pyglet.app.exit()
        elif symbol == pyglet.window.key.SPACE:
            self.show_processed = not self.show_processed
        elif symbol == pyglet.window.key.W:
            self.show_white_square = not self.show_white_square

    def run(self):
        self.capture_thread.start()
        try:
            pyglet.app.run()
        finally:
            self.shutdown.set()
            self.capture_thread.join()
            pygame.mixer.quit()  # Clean up pygame mixer
            self.context.destroy()
            self.collect_socket.close()
            self.distribute_socket.close()

if __name__ == '__main__':
    app = WebcamApp()
    app.run() 
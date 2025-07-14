import pyglet
from pyglet.gl import *
import zmq
import threading
import time
import numpy as np
import os
import json
import signal
import sys

from trace_logger import TraceLogger
import re
import pygame  # Add pygame import
from collections import OrderedDict
import subprocess

# Constants
CAPTURE_WIDTH = 1920
CAPTURE_HEIGHT = 1080
TARGET_SIZE = 1024
QUEUE_SIZE = 4
FRAME_LATENCY_MS = 1000  # latency for frame ordering - adjust this value as needed
DEBUG_FRAME_PRINTS = False  # Set to True to enable frame-related debug prints

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
        
        # Initialize input frame rate tracking
        self.input_frame_count = 0
        self.last_input_frame_time = time.time()
        self.last_input_frame_count = 0
        
        # Initialize frame ordering
        self.output_frame_count = 0
        self.frame_buffer = {} # frame_index -> (timestamp, texture)
        self.last_display_time = time.time()
        
        # Initialize display frame rate tracking
        self.display_frame_count = 0
        self.last_display_fps_time = time.time()
        
        # Initialize settings
        self.settings_file = 'settings.json'
        self.last_settings_mtime = 0
        self.load_settings()
        
        # Initialize ZMQ context and sockets
        self.context = zmq.Context()
        self.setup_sockets()
        
        # Initialize threading events
        self.shutdown = threading.Event()
        self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
        
        # Initialize window and graphics
        self.setup_window()
        
        # Initialize frame storage
        self.current_processed_timestamp = None
        self.processed_texture = None
        
        # Load prompts and initialize prompt state
        self.prompts = self.load_prompts()
        self.current_prompt_idx = 0
        self.last_prompt_change = None
        
        # Add white square toggle
        self.show_white_square = False
        
        # Load mask texture and track its modification time
        # self.mask_file = 'mask.png'
        # self.last_mask_mtime = os.path.getmtime(self.mask_file)
        # mask_image = pyglet.image.load(self.mask_file)
        # self.mask_texture = mask_image.get_texture()
        self.mask_texture = None
        
        # Initialize FFmpeg pipe
        self.ffmpeg_pipe = None
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # Schedule updates
        pyglet.clock.schedule_interval(self.update_frame, 1/60.0)
        pyglet.clock.schedule_interval(self.check_settings, 1.0)

    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print(f"\nReceived signal {signum}, shutting down gracefully...")
        self.shutdown.set()
        pyglet.app.exit()

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
            # try:
            #     mask_mtime = os.path.getmtime(self.mask_file)
            #     if mask_mtime > self.last_mask_mtime:
            #         self.last_mask_mtime = mask_mtime
            #         # Reload mask texture
            #         if self.mask_texture:
            #             self.mask_texture.delete()
            #         mask_image = pyglet.image.load(self.mask_file)
            #         self.mask_texture = mask_image.get_texture()
            # except OSError:
            #     pass

        except Exception as e:
            self.logger.error(f"Error in check_settings: {str(e)}")

    def setup_ffmpeg_pipe(self):
        """Setup FFmpeg pipe for webcam capture with cropping"""
        # Calculate crop parameters
        crop_x = (CAPTURE_WIDTH - TARGET_SIZE) // 2
        crop_y = (CAPTURE_HEIGHT - TARGET_SIZE) // 2
        
        # FFmpeg command with cropping and RGB output
        ffmpeg_cmd = (
            f"ffmpeg -hide_banner -loglevel error "
            f"-f v4l2 -input_format mjpeg -framerate {self.camera_fps} "
            f"-video_size {CAPTURE_WIDTH}x{CAPTURE_HEIGHT} -i /dev/video0 "
            f"-vf crop={TARGET_SIZE}:{TARGET_SIZE}:{crop_x}:{crop_y} "
            "-f rawvideo -pix_fmt rgb24 -"
        )
        
        try:
            self.ffmpeg_pipe = subprocess.Popen(
                ffmpeg_cmd.split(), 
                stdout=subprocess.PIPE, 
                stderr=subprocess.DEVNULL
            )
            print(f"FFmpeg pipe started: {ffmpeg_cmd}")
        except FileNotFoundError:
            print("Error: ffmpeg not found. Please install ffmpeg.")
            self.shutdown.set()
            return False
        except Exception as e:
            print(f"Error starting FFmpeg pipe: {e}")
            self.shutdown.set()
            return False
        
        return True

    def cleanup_ffmpeg_pipe(self):
        """Clean shutdown of ffmpeg process"""
        if self.ffmpeg_pipe:
            print("Cleaning up ffmpeg process...")
            try:
                # Send SIGTERM first
                self.ffmpeg_pipe.terminate()
                try:
                    self.ffmpeg_pipe.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    print("FFmpeg process didn't terminate gracefully, forcing kill...")
                    # If SIGTERM doesn't work, force kill
                    self.ffmpeg_pipe.kill()
                    self.ffmpeg_pipe.wait()
                finally:
                    self.ffmpeg_pipe.stdout.close()
                    self.ffmpeg_pipe = None
                    print("FFmpeg process cleaned up successfully")
            except Exception as e:
                print(f"Error during FFmpeg cleanup: {e}")

    def capture_loop(self):
        if not self.setup_ffmpeg_pipe():
            return
        
        try:
            while not self.shutdown.is_set():
                current_time = time.time()
            
                # Read frame data from FFmpeg pipe
                frame_data = self.ffmpeg_pipe.stdout.read(TARGET_SIZE * TARGET_SIZE * 3)
                if not frame_data:
                    time.sleep(0.01)
                    continue

                # Convert frame data to numpy array (already RGB from FFmpeg)
                frame = np.frombuffer(frame_data, dtype=np.uint8).reshape(TARGET_SIZE, TARGET_SIZE, 3)
                
                try:
                    frame_float = np.float32(frame) / 255.0
                    self.distribute_socket.send_multipart([
                        str(current_time).encode(),
                        frame_float.tobytes(),
                        self.get_current_prompt().encode(),
                        str(self.input_frame_count).encode()
                    ], flags=zmq.DONTWAIT)
                    if DEBUG_FRAME_PRINTS:
                        print(f"frame {self.input_frame_count}: sent to workers")
                except zmq.Again:
                    self.frame_buffer[self.input_frame_count] = "dropped"
                    if DEBUG_FRAME_PRINTS:
                        print(f"frame {self.input_frame_count}: dropped, distribute_socket ZMQ buffer full")
                    pass
                
                # Track input frame rate
                if current_time - self.last_input_frame_time >= 10.0 and self.input_frame_count > 0:
                    input_fps = (self.input_frame_count - self.last_input_frame_count) / \
                        (current_time - self.last_input_frame_time)
                    print(f"Camera FPS: {input_fps:.2f}")
                    self.last_input_frame_count = self.input_frame_count
                    self.last_input_frame_time = current_time
                self.input_frame_count += 1

        finally:
            self.cleanup_ffmpeg_pipe()

    def update_frame(self, dt):
        current_time = time.time()
        
        # Try to receive processed frame from workers (non-blocking)
        try:
            multipart_msg = self.collect_socket.recv_multipart(flags=zmq.NOBLOCK)
            if len(multipart_msg) == 4:  # Now expecting 4 parts including frame index
                timestamp_str, frame_data, worker_id_bytes, frame_index_bytes = multipart_msg
                timestamp = float(timestamp_str.decode())
                worker_id = worker_id_bytes.decode()
                frame_index = int(frame_index_bytes.decode())
                processed_frame = np.frombuffer(frame_data, dtype=np.uint8).reshape(TARGET_SIZE, TARGET_SIZE, 3)
                self.frame_buffer[frame_index] = (timestamp, processed_frame)
                if DEBUG_FRAME_PRINTS:
                    print(f"frame {frame_index}: received from worker {worker_id}")
                
                frame_age = current_time - timestamp
                worker_id_number = 1 if 'transformirror1' in worker_id else 2
                # with open('frame_age.csv', 'a') as f:
                #     f.write(f"{worker_id_number},{frame_age*1000:.0f}\n")
                
        except zmq.Again:
            # No message available
            pass
    
        # Check if current frame is available and not too old
        while self.output_frame_count in self.frame_buffer:
            if self.frame_buffer[self.output_frame_count] == "dropped":
                del self.frame_buffer[self.output_frame_count]
                self.output_frame_count += 1
                continue
            
            timestamp, frame_data = self.frame_buffer[self.output_frame_count]
            frame_age = current_time - timestamp
            
            # Check if frame is within latency window
            if frame_age <= FRAME_LATENCY_MS / 1000.0:
                del self.frame_buffer[self.output_frame_count]
                if DEBUG_FRAME_PRINTS:
                    print(f"frame {self.output_frame_count}: creating texture (age: {frame_age*1000:.1f}ms)", flush=True)
                self.output_frame_count += 1
                break
            else:
                # Frame is too old, skip it and try the next one
                if DEBUG_FRAME_PRINTS:
                    print(f"frame {self.output_frame_count}: too old ({frame_age*1000:.1f}ms), skipping", flush=True)
                del self.frame_buffer[self.output_frame_count]
                self.output_frame_count += 1
        else:
            # No frames available in buffer
            if DEBUG_FRAME_PRINTS:
                print(f"frame {self.output_frame_count}: not ready in frame buffer")
            return
        
        # Create texture from frame data
        image = pyglet.image.ImageData(
            TARGET_SIZE, TARGET_SIZE,
            'RGB', frame_data.tobytes(),
            pitch=TARGET_SIZE * 3
        )
        texture = image.get_texture().get_transform(flip_y=True, flip_x=True)
        texture.anchor_x = 0
        texture.anchor_y = 0
        
        self.processed_texture = texture
        self.current_processed_timestamp = timestamp
        self.last_display_time = current_time
        self.display_frame_count += 1
        
        # Track display frame rate
        if current_time - self.last_display_fps_time >= 10.0:
            display_fps = self.display_frame_count / (current_time - self.last_display_fps_time)
            print(f"Display Frame Rate: {display_fps:.2f} fps")
            self.display_frame_count = 0
            self.last_display_fps_time = current_time
        
    def on_draw(self):
        self.frame_count += 1
        current_time = time.time()
        
        if self.current_processed_timestamp is not None:
            processed_age = current_time - self.current_processed_timestamp
            # with open('processed_age.csv', 'a') as f:
            #     f.write(f"{processed_age*1000:.0f}\n")
                
            # with open('output_timing.csv', 'a') as f:
            #     f.write(f"{current_time:0.3f},{self.current_processed_timestamp:0.3f}\n")
        
        if current_time - self.last_fps_time >= 10.0:
            fps = self.frame_count / (current_time - self.last_fps_time)
            # if fps < 30:
            print(f"Display FPS: {fps:.2f}")
            self.frame_count = 0
            self.last_fps_time = current_time
            
        try:
            self.window.clear()
            
            # Cache window dimensions to avoid repeated property access
            window_width = self.window.width
            window_height = self.window.height
            # side = window_height
            # x = (window_width - side) / 2
            
            if self.show_white_square:
                white_square = pyglet.shapes.Rectangle(x=0, y=0, width=window_width, height=window_height, 
                                                     color=(255, 255, 255))
                white_square.draw()
            else:
                texture = self.processed_texture
                if texture is not None:
                    texture.blit(0, 0, width=window_width, height=window_height)
                    
        except Exception:
            pass
        
        # draw X for projection boundaries
        # Line from top-left to bottom-right
        # line1 = pyglet.shapes.Line(0, window_height, window_width, 0, color=(255, 0, 0))
        # line1.draw()
        # line2 = pyglet.shapes.Line(window_width, window_height, 0, 0, color=(0, 255, 0))
        # line2.draw()
        
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
        elif symbol == pyglet.window.key.W:
            self.show_white_square = not self.show_white_square

    def run(self):
        print("Starting webcam application...")
        self.capture_thread.start()
        try:
            pyglet.app.run()
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt received, shutting down...")
        except Exception as e:
            print(f"Unexpected error: {e}")
        finally:
            print("Cleaning up resources...")
            self.shutdown.set()
            
            # Wait for capture thread to finish
            if self.capture_thread.is_alive():
                print("Waiting for capture thread to finish...")
                self.capture_thread.join(timeout=3)
                if self.capture_thread.is_alive():
                    print("Warning: Capture thread did not finish gracefully")
            
            # Clean up FFmpeg pipe
            self.cleanup_ffmpeg_pipe()
            
            # Clean up pygame
            try:
                pygame.mixer.quit()
            except Exception as e:
                print(f"Error cleaning up pygame: {e}")
            
            # Clean up ZMQ
            try:
                self.collect_socket.close()
                self.distribute_socket.close()
                self.context.destroy()
            except Exception as e:
                print(f"Error cleaning up ZMQ: {e}")
            
            print("Cleanup completed successfully")

if __name__ == '__main__':
    try:
        app = WebcamApp()
        app.run()
    except KeyboardInterrupt:
        print("\nExiting due to KeyboardInterrupt")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)
    finally:
        print("Application terminated") 
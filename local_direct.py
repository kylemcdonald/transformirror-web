from pyglet.gl import *
import threading
import time
import numpy as np
import os
import json
import signal
import sys
import subprocess
import pyglet
import pygame  # Add pygame import
from diffusion_processor import DiffusionProcessor
from PIL import Image

CAPTURE_WIDTH = 1920
CAPTURE_HEIGHT = 1080
TARGET_SIZE = 1024

config = pyglet.gl.Config(
    double_buffer=True,
    sample_buffers=1,
    samples=4,
    alpha_size=8,
    depth_size=24
)

class WebcamApp:
    def __init__(self):
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.display_frame_count = 0
        self.last_display_fps_time = time.time()
        
        self.settings_file = 'settings.json'
        self.last_settings_mtime = 0
        self.load_settings()
        
        self.processor = DiffusionProcessor(local_files_only=True, gpu_id=0, use_compel=True)
        
        self.shutdown = threading.Event()
        self.frame_buffer = None
        self.frame_lock = threading.Lock()
        self.texture_needs_update = False
        self.ffmpeg_pipe = None
        
        # Initialize pygame mixer for audio
        for attempt in range(3):
            try:
                pygame.mixer.init()
                break
            except pygame.error:
                print("Failed to initialize pygame mixer. Retrying...", flush=True)
                time.sleep(1)
        print("Successfully initialized pygame mixer", flush=True)
        
        # Load prompts and initialize prompt state
        self.prompts = self.load_prompts()
        self.current_prompt_idx = 0
        self.last_prompt_change = None
        
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        self.setup_window()
        self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
        pyglet.clock.schedule_interval(self.check_settings, 1.0)

    def signal_handler(self, signum, frame):
        print(f"\nReceived signal {signum}, shutting down gracefully...")
        self.shutdown.set()
        pyglet.app.exit()

    def setup_window(self):
        try:
            self.window = pyglet.window.Window(fullscreen=True, config=config, vsync=True)
        except pyglet.window.NoSuchConfigException:
            try:
                self.window = pyglet.window.Window(fullscreen=True, vsync=True)
            except Exception as e:
                print(f"Failed to create fullscreen window: {e}")
                self.window = pyglet.window.Window(width=1024, height=768, vsync=True)
        
        self.window.event(self.on_draw)
        self.window.event(self.on_key_press)

    def load_settings(self):
        try:
            with open(self.settings_file, 'r') as f:
                settings = json.load(f)
                self.camera_fps = settings.get("camera_fps", 20)
                self.prompt_cycle_time = settings.get("prompt_cycle_time", 10)
        except FileNotFoundError:
            print(f"Error: Settings file '{self.settings_file}' not found.")
            print("This script requires a settings file to control its behavior.")
            sys.exit(1)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error: Invalid settings file '{self.settings_file}': {e}")
            print("Please check the JSON format and required fields.")
            sys.exit(1)

    def load_prompts(self):
        try:
            with open('prompts.txt', 'r') as f:
                return [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            return ["A beautiful portrait"]

    def get_current_prompt(self):
        current_time = time.time()
        if self.last_prompt_change is None or current_time - self.last_prompt_change >= self.prompt_cycle_time:
            n = len(self.prompts)
            self.current_prompt_idx = (self.current_prompt_idx + 1) % n
            self.last_prompt_change = current_time
            
            # Play corresponding audio file when prompt changes
            try:
                audio_idx = self.current_prompt_idx % (n // 2)
                audio_file = f"audio/{audio_idx:02d}.wav"
                if os.path.exists(audio_file):
                    pygame.mixer.music.stop()
                    pygame.mixer.music.load(audio_file)
                    pygame.mixer.music.play()
                print(f"Playing audio: {audio_file} ({self.current_prompt_idx} of {n})", flush=True)
            except Exception as e:
                print(f"Error playing audio: {str(e)}", flush=True)
            
        return self.prompts[self.current_prompt_idx]

    def check_settings(self, dt):
        try:
            try:
                settings_mtime = os.path.getmtime(self.settings_file)
                if settings_mtime > self.last_settings_mtime:
                    self.last_settings_mtime = settings_mtime
                    self.load_settings()
            except OSError:
                pass
        except Exception as e:
            print(f"Error in check_settings: {str(e)}")

    def setup_ffmpeg_pipe(self):
        crop_x = (CAPTURE_WIDTH - TARGET_SIZE) // 2
        crop_y = (CAPTURE_HEIGHT - TARGET_SIZE) // 2
        
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
        if self.ffmpeg_pipe:
            print("Cleaning up ffmpeg process...")
            try:
                self.ffmpeg_pipe.terminate()
                try:
                    self.ffmpeg_pipe.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    print("FFmpeg process didn't terminate gracefully, forcing kill...")
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
            
                frame_data = self.ffmpeg_pipe.stdout.read(TARGET_SIZE * TARGET_SIZE * 3)
                if not frame_data:
                    time.sleep(0.01)
                    continue

                try:
                    frame = np.frombuffer(frame_data, dtype=np.uint8).reshape(TARGET_SIZE, TARGET_SIZE, 3)
                except ValueError as e:
                    print(f"Error reshaping frame data: {e}")
                    continue
                
                try:                    
                    frame = np.float32(frame) / 255.0
                    processed_frame = self.processor([frame], self.get_current_prompt())
                    processed_frame = np.uint8(processed_frame[0] * 255)
                except Exception as e:
                    print(f"Error processing frame: {e}")
                    continue
                
                with self.frame_lock:
                    self.frame_buffer = processed_frame.copy()
                    self.texture_needs_update = True
                
                self.display_frame_count += 1
                if current_time - self.last_display_fps_time >= 10.0:
                    display_fps = self.display_frame_count / (current_time - self.last_display_fps_time)
                    print(f"Capture FPS: {display_fps:.2f}")
                    self.display_frame_count = 0
                    self.last_display_fps_time = current_time

        except Exception as e:
            print(f"Error in capture loop: {e}")
        finally:
            self.cleanup_ffmpeg_pipe()
        
    def on_draw(self):
        self.frame_count += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 10.0:
            fps = self.frame_count / (current_time - self.last_fps_time)
            print(f"Display FPS: {fps:.2f}")
            self.frame_count = 0
            self.last_fps_time = current_time
            
        try:
            self.window.clear()
            
            window_width = self.window.width
            window_height = self.window.height
            
            with self.frame_lock:
                if self.texture_needs_update and self.frame_buffer is not None:
                    try:
                        image = pyglet.image.ImageData(
                            TARGET_SIZE, TARGET_SIZE,
                            'RGB', self.frame_buffer.tobytes(),
                            pitch=TARGET_SIZE * 3
                        )
                        texture = image.get_texture().get_transform(flip_y=True, flip_x=True)
                        texture.anchor_x = 0
                        texture.anchor_y = 0
                        
                        self.current_texture = texture
                        self.texture_needs_update = False
                        
                    except Exception as e:
                        print(f"Error creating texture: {e}")
                        self.texture_needs_update = False
            
            if hasattr(self, 'current_texture') and self.current_texture is not None:
                try:
                    self.current_texture.blit(0, 0, width=window_width, height=window_height)
                except Exception as e:
                    print(f"Error blitting texture: {e}")
                    try:
                        self.current_texture.delete()
                    except:
                        pass
                    self.current_texture = None
                    
        except Exception as e:
            print(f"Error in on_draw: {e}")

    def on_key_press(self, symbol, modifiers):
        if symbol == pyglet.window.key.ESCAPE:
            self.shutdown.set()
            pyglet.app.exit()

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
            
            if self.capture_thread.is_alive():
                print("Waiting for capture thread to finish...")
                self.capture_thread.join(timeout=3)
                if self.capture_thread.is_alive():
                    print("Warning: Capture thread did not finish gracefully")
            
            self.cleanup_ffmpeg_pipe()
            
            # Clean up pygame
            try:
                pygame.mixer.quit()
            except Exception as e:
                print(f"Error cleaning up pygame: {e}")
            
            if hasattr(self, 'current_texture') and self.current_texture:
                try:
                    self.current_texture.delete()
                except Exception as e:
                    print(f"Error deleting texture during cleanup: {e}")
            
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
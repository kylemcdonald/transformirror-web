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
CROP_SIZE = 1080
INPUT_SIZE = 768
DISPLAY_SIZE = 768
OVERLAY_SIZE = 768
RECONNECT_DELAY = 1.0  # seconds between reconnection attempts

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
        self.overlay_texture = None
        self.overlay_path = os.path.join(os.path.dirname(__file__), 'overlay.png')
        
        self.shutdown = threading.Event()
        self.frame_buffer = None
        self.frame_lock = threading.Lock()
        self.texture_needs_update = False
        self.ffmpeg_pipe = None
        self.reconnect_event = threading.Event()
        self.is_connected = False
        
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
        self.load_overlay_texture()
        self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
        pyglet.clock.schedule_interval(self.check_settings, 1.0)

    def signal_handler(self, signum, frame):
        print(f"\nReceived signal {signum}, shutting down gracefully...")
        self.shutdown.set()
        pyglet.app.exit()

    def setup_window(self):
        screens = pyglet.display.get_display().get_screens()
        screen = screens[1]
        try:
            self.window = pyglet.window.Window(fullscreen=True, config=config, vsync=True, display=0, screen=screen)
        except pyglet.window.NoSuchConfigException:
            try:
                self.window = pyglet.window.Window(fullscreen=True, vsync=True)
            except Exception as e:
                print(f"Failed to create fullscreen window: {e}")
                self.window = pyglet.window.Window(width=1920, height=1080, vsync=True)
        
        # Hide the system cursor so it doesn't distract during display
        try:
            self.window.set_mouse_visible(False)
        except Exception as e:
            print(f"Warning: Unable to hide cursor: {e}")
        
        # Verify window resolution
        actual_width = self.window.width
        actual_height = self.window.height
        expected_width = 1920
        expected_height = 1080
        
        print(f"Window created with resolution: {actual_width}x{actual_height}", flush=True)
        
        if actual_width == expected_width and actual_height == expected_height:
            print(f"Resolution verified: {actual_width}x{actual_height} (correct)", flush=True)
        else:
            print(f"WARNING: Expected {expected_width}x{expected_height}, but got {actual_width}x{actual_height}", flush=True)
        
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        self.window.event(self.on_draw)
        self.window.event(self.on_key_press)

    def load_overlay_texture(self):
        try:
            if not os.path.exists(self.overlay_path):
                print(f"Overlay file not found at {self.overlay_path}")
                return
            overlay_image = pyglet.image.load(self.overlay_path)
            self.overlay_texture = overlay_image.get_texture()
            self.overlay_texture.anchor_x = 0
            self.overlay_texture.anchor_y = 0
            print("Overlay texture loaded successfully")
        except Exception as e:
            print(f"Error loading overlay texture: {e}")
            self.overlay_texture = None

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

    def check_webcam_available(self):
        """Check if /dev/video0 exists and is accessible"""
        return os.path.exists('/dev/video0')

    def setup_ffmpeg_pipe(self):
        if not self.check_webcam_available():
            print("Webcam device /dev/video0 not found")
            return False
            
        crop_x = (CAPTURE_WIDTH - CROP_SIZE) // 2
        crop_y = (CAPTURE_HEIGHT - CROP_SIZE) // 2
        
        ffmpeg_cmd = (
            f"ffmpeg -hide_banner -loglevel error "
            f"-f v4l2 -input_format mjpeg -framerate {self.camera_fps} "
            f"-video_size {CAPTURE_WIDTH}x{CAPTURE_HEIGHT} -i /dev/video0 "
            f"-vf crop={CROP_SIZE}:{CROP_SIZE}:{crop_x}:{crop_y},"
            f"scale={INPUT_SIZE}:{INPUT_SIZE} "
            "-f rawvideo -pix_fmt rgb24 -"
        )
        
        try:
            self.ffmpeg_pipe = subprocess.Popen(
                ffmpeg_cmd.split(), 
                stdout=subprocess.PIPE, 
                stderr=subprocess.DEVNULL
            )
            print(f"FFmpeg pipe started: {ffmpeg_cmd}")
            self.is_connected = True
            return True
        except FileNotFoundError:
            print("Error: ffmpeg not found. Please install ffmpeg.")
            return False
        except Exception as e:
            print(f"Error starting FFmpeg pipe: {e}")
            return False

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
                    self.is_connected = False
                    print("FFmpeg process cleaned up successfully")
            except Exception as e:
                print(f"Error during FFmpeg cleanup: {e}")

    def attempt_reconnection(self):
        """Attempt to reconnect to the webcam"""
        print("Attempting to reconnect to webcam...")
        
        # Clean up existing pipe
        self.cleanup_ffmpeg_pipe()
        
        # Wait a bit before attempting reconnection
        time.sleep(RECONNECT_DELAY)
        
        # Try to reconnect
        if self.setup_ffmpeg_pipe():
            print("Successfully reconnected to webcam")
            return True
        else:
            print("Failed to reconnect to webcam")
            return False

    def capture_loop(self):
        if not self.setup_ffmpeg_pipe():
            print("Initial webcam setup failed")
            return
        
        consecutive_failures = 0
        
        try:
            while not self.shutdown.is_set():
                if not self.is_connected or not self.ffmpeg_pipe:
                    # Try to reconnect indefinitely
                    if self.attempt_reconnection():
                        consecutive_failures = 0
                    else:
                        consecutive_failures += 1
                        time.sleep(RECONNECT_DELAY)
                        continue
                
                try:
                    frame_data = self.ffmpeg_pipe.stdout.read(INPUT_SIZE * INPUT_SIZE * 3)
                    if not frame_data:
                        # Check if process is still alive
                        if self.ffmpeg_pipe.poll() is not None:
                            print("FFmpeg process terminated unexpectedly")
                            self.is_connected = False
                            consecutive_failures += 1
                            continue
                        time.sleep(0.01)
                        continue

                    # Reset failure counter on successful read
                    consecutive_failures = 0

                    try:
                        frame = np.frombuffer(frame_data, dtype=np.uint8).reshape(INPUT_SIZE, INPUT_SIZE, 3)
                    except ValueError as e:
                        print(f"Error reshaping frame data: {e}")
                        continue
                    
                    try:
                        frame = np.float32(frame) / 255.0
                        processed_frame = self.processor(frame, self.get_current_prompt())
                        processed_frame = np.uint8(processed_frame * 255)

                    except Exception as e:
                        print(f"Error processing frame: {e}")
                        continue
                    
                    with self.frame_lock:
                        self.frame_buffer = processed_frame.copy()
                        self.texture_needs_update = True
                    
                    self.display_frame_count += 1
                    current_time = time.time()
                    if current_time - self.last_display_fps_time >= 10.0:
                        display_fps = self.display_frame_count / (current_time - self.last_display_fps_time)
                        print(f"Capture FPS: {display_fps:.2f}")
                        self.display_frame_count = 0
                        self.last_display_fps_time = current_time

                except (OSError, IOError) as e:
                    print(f"IO Error in capture loop: {e}")
                    self.is_connected = False
                    consecutive_failures += 1
                    continue
                except Exception as e:
                    print(f"Error in capture loop: {e}")
                    consecutive_failures += 1
                    continue

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
            glClearColor(1.0, 1.0, 1.0, 1.0)  # Set clear color to white (R, G, B, A)
            self.window.clear()
            
            window_width = self.window.width
            window_height = self.window.height
            
            with self.frame_lock:
                if self.texture_needs_update and self.frame_buffer is not None:
                    try:
                        image = pyglet.image.ImageData(
                            INPUT_SIZE, INPUT_SIZE,
                            'RGB', self.frame_buffer.tobytes(),
                            pitch=INPUT_SIZE * 3
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
                    # Calculate center position for 768x768 image on 1920x1080 screen
                    image_x = (window_width - DISPLAY_SIZE) // 2
                    image_y = (window_height - DISPLAY_SIZE) // 2
                    # Blit at native 768x768 size (pixel-perfect) at center position
                    self.current_texture.blit(image_x, image_y, width=DISPLAY_SIZE, height=DISPLAY_SIZE)
                except Exception as e:
                    print(f"Error blitting texture: {e}")
                    try:
                        self.current_texture.delete()
                    except:
                        pass
                    self.current_texture = None
            else:
                # Display connection status when no texture is available
                if not self.is_connected:
                    label = pyglet.text.Label(
                        'Webcam disconnected - attempting to reconnect...',
                        font_name='Arial',
                        font_size=16,
                        x=window_width//2, y=window_height//2,
                        anchor_x='center', anchor_y='center',
                        color=(255, 255, 255, 255)
                    )
                    label.draw()
            
            if self.overlay_texture is not None:
                try:
                    overlay_x = (window_width - OVERLAY_SIZE) // 2
                    overlay_y = (window_height - OVERLAY_SIZE) // 2
                    self.overlay_texture.blit(
                        overlay_x,
                        overlay_y,
                        width=OVERLAY_SIZE,
                        height=OVERLAY_SIZE
                    )
                except Exception as e:
                    print(f"Error drawing overlay: {e}")
                    self.overlay_texture = None
                    
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
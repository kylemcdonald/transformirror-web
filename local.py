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
from collections import OrderedDict
import subprocess

# Constants
CAPTURE_WIDTH = 1920
CAPTURE_HEIGHT = 1080
TARGET_SIZE = 1024
QUEUE_SIZE = 2
FRAME_LATENCY_MS = 600  # latency for frame ordering - adjust this value as needed

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
        self.last_input_fps_time = time.time()
        
        # Initialize frame ordering
        self.frame_index = 0
        self.frame_buffer = OrderedDict()  # frame_index -> (timestamp, texture)
        self.last_display_time = time.time()
        self.display_interval = None  # Will be set after loading settings
        
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
        # self.mask_file = 'mask.png'
        # self.last_mask_mtime = os.path.getmtime(self.mask_file)
        # mask_image = pyglet.image.load(self.mask_file)
        # self.mask_texture = mask_image.get_texture()
        self.mask_texture = None
        
        # Initialize FFmpeg pipe
        self.ffmpeg_pipe = None
        
        # Schedule updates
        pyglet.clock.schedule_interval(self.update_frame, 1/60.0)
        pyglet.clock.schedule_interval(self.check_settings, 1.0)
        pyglet.clock.schedule_interval(self.cleanup_old_frames, 5.0)  # Clean up every 5 seconds

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
            self.camera_fps = settings.get("camera_fps", 15)
            self.prompt_cycle_time = settings.get("prompt_cycle_time", 10)
            self.settings_show_processed = settings.get("show_processed", False)
            # Set display interval to match camera FPS
            self.display_interval = 1.0 / self.camera_fps

    def check_settings(self, dt):
        try:
            # Check settings file
            try:
                settings_mtime = os.path.getmtime(self.settings_file)
                if settings_mtime > self.last_settings_mtime:
                    self.last_settings_mtime = settings_mtime
                    self.load_settings()
                    # Update display interval when settings change
                    self.display_interval = 1.0 / self.camera_fps
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

    @property
    def show_processed(self):
        if hasattr(self, 'user_show_processed'):
            return self.user_show_processed
        return self.settings_show_processed

    @show_processed.setter
    def show_processed(self, value):
        self.user_show_processed = value

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
            # Send SIGTERM first
            self.ffmpeg_pipe.terminate()
            try:
                self.ffmpeg_pipe.wait(timeout=1)
            except subprocess.TimeoutExpired:
                # If SIGTERM doesn't work, force kill
                self.ffmpeg_pipe.kill()
                self.ffmpeg_pipe.wait()
            finally:
                self.ffmpeg_pipe.stdout.close()
                self.ffmpeg_pipe = None

    def capture_loop(self):
        if not self.setup_ffmpeg_pipe():
            return
        
        last_send_time = 0
        send_interval = 1.0 / 30  # Limit sending to workers to 30fps
        
        try:
            while not self.shutdown.is_set():
                # Read frame data from FFmpeg pipe
                frame_data = self.ffmpeg_pipe.stdout.read(TARGET_SIZE * TARGET_SIZE * 3)
                if not frame_data:
                    time.sleep(0.01)
                    continue

                # Track input frame rate
                self.input_frame_count += 1
                current_time = time.time()
                if current_time - self.last_input_fps_time >= 10.0:
                    input_fps = self.input_frame_count / (current_time - self.last_input_fps_time)
                    print(f"Camera FPS: {input_fps:.2f}")
                    self.input_frame_count = 0
                    self.last_input_fps_time = current_time

                # Convert frame data to numpy array (already RGB from FFmpeg)
                frame = np.frombuffer(frame_data, dtype=np.uint8).reshape(TARGET_SIZE, TARGET_SIZE, 3)
                
                # Update display queue if empty
                if self.frame_queue.empty():
                    try:
                        self.frame_queue.put_nowait(frame)
                    except Full:
                        pass
                
                # Send frame to workers at reduced rate
                current_time = time.time()
                if current_time - last_send_time >= send_interval:
                    try:
                        frame_float = np.float32(frame) / 255.0
                        self.distribute_socket.send_multipart([
                            str(current_time).encode(),
                            frame_float.tobytes(),
                            self.get_current_prompt().encode(),
                            str(self.frame_index).encode()  # Add frame index
                        ], flags=zmq.DONTWAIT)
                        last_send_time = current_time
                        self.frame_index += 1
                    except zmq.Again:
                        pass
        finally:
            self.cleanup_ffmpeg_pipe()

    def update_frame(self, dt):
        current_time = time.time()
        
        # Check if it's time to display a frame
        if current_time - self.last_display_time < self.display_interval:
            return
        
        # Try to receive processed frame from workers (non-blocking)
        try:
            multipart_msg = self.collect_socket.recv_multipart(flags=zmq.NOBLOCK)
            if len(multipart_msg) == 4:  # Now expecting 4 parts including frame index
                timestamp_str, frame_data, worker_id_bytes, frame_index_bytes = multipart_msg
                timestamp = float(timestamp_str.decode())
                worker_id = worker_id_bytes.decode()
                frame_index = int(frame_index_bytes.decode())
                
                # Check if frame arrived within latency window
                frame_age = current_time - timestamp
                
                # write the frame age to a file
                with open('frame_age.csv', 'a') as f:
                    f.write(f"{frame_age*1000:.1f}\n")
                
                if frame_age <= FRAME_LATENCY_MS / 1000.0:
                    processed_frame = np.frombuffer(frame_data, dtype=np.uint8).reshape(TARGET_SIZE, TARGET_SIZE, 3)
                    
                    # Create texture for this frame
                    image = pyglet.image.ImageData(
                        TARGET_SIZE, TARGET_SIZE,
                        'RGB', processed_frame.tobytes(),
                        pitch=TARGET_SIZE * 3
                    )
                    texture = image.get_texture().get_transform(flip_y=True, flip_x=True)
                    
                    # Store frame in buffer
                    self.frame_buffer[frame_index] = (timestamp, texture)
                else:
                    print(f"Dropped frame {frame_index} - too old ({frame_age*1000:.1f}ms)")
                    
        except zmq.Again:
            # No message available
            pass
        except Exception as e:
            # Handle any other exceptions silently
            pass
        
        # Find the next frame to display (in order)
        next_frame_index = None
        if self.frame_buffer:
            # Get the next expected frame index
            expected_frame = min(self.frame_buffer.keys())
            if expected_frame in self.frame_buffer:
                next_frame_index = expected_frame
        
        if next_frame_index is not None:
            # Display the frame
            timestamp, texture = self.frame_buffer.pop(next_frame_index)
            
            # Clean up the previous processed texture before replacing it
            if self.processed_texture is not None and self.processed_texture != texture:
                try:
                    self.processed_texture.delete()
                except:
                    pass  # Texture might already be deleted
            
            self.processed_texture = texture
            self.last_display_time = current_time
        else:
            # No frame ready, check if we should drop frames
            if self.frame_buffer:
                oldest_frame = min(self.frame_buffer.keys())
                oldest_timestamp = self.frame_buffer[oldest_frame][0]
                if current_time - oldest_timestamp > FRAME_LATENCY_MS / 1000.0:
                    dropped_frame = self.frame_buffer.pop(oldest_frame)
                    # Clean up the dropped frame's texture
                    try:
                        dropped_frame[1].delete()
                    except:
                        pass  # Texture might already be deleted
                    print(f"Dropped frame {oldest_frame} - exceeded latency window")
        
        # Update current frame texture (for unprocessed view)
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

    def cleanup_old_frames(self, dt):
        """Cleans up frames older than a certain threshold to prevent memory buildup."""
        current_time = time.time()
        threshold_time = current_time - (FRAME_LATENCY_MS / 1000.0 + 1) # Clean up frames older than 501ms
        
        frames_to_delete = []
        for frame_index, (timestamp, texture) in list(self.frame_buffer.items()):
            if timestamp < threshold_time:
                frames_to_delete.append(frame_index)
        
        for frame_index in frames_to_delete:
            timestamp, texture = self.frame_buffer.pop(frame_index)
            try:
                texture.delete()
            except:
                pass
            print(f"Cleaned up frame {frame_index} (old)")

    def on_draw(self):
        self.frame_count += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 10.0:
            fps = self.frame_count / (current_time - self.last_fps_time)
            # if fps < 30:
            print(f"Display FPS: {fps:.2f}")
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
            
            # Clean up textures
            self.cleanup_textures()
            
            # Clean up FFmpeg pipe
            self.cleanup_ffmpeg_pipe()
            
            pygame.mixer.quit()  # Clean up pygame mixer
            self.context.destroy()
            self.collect_socket.close()
            self.distribute_socket.close()

    def cleanup_textures(self):
        """Clean up all textures to free GPU memory"""
        try:
            # Clean up current textures
            if self.current_texture:
                self.current_texture.delete()
                self.current_texture = None
            
            if self.processed_texture:
                self.processed_texture.delete()
                self.processed_texture = None
            
            # Clean up frame buffer textures
            for frame_index, (timestamp, texture) in list(self.frame_buffer.items()):
                try:
                    texture.delete()
                except:
                    pass
            self.frame_buffer.clear()
            
            # Clean up mask texture
            if self.mask_texture:
                self.mask_texture.delete()
                self.mask_texture = None
                
        except Exception as e:
            self.logger.error(f"Error during texture cleanup: {str(e)}")

if __name__ == '__main__':
    app = WebcamApp()
    app.run() 
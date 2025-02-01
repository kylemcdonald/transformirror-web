import cv2
import pyglet
import zmq
import threading
import time
import heapq
import numpy as np
import os
from queue import Queue, Empty
from typing import Optional
from trace_logger import TraceLogger

# Constants
CAPTURE_WIDTH = 1920
CAPTURE_HEIGHT = 1080
TARGET_SIZE = 1024
APP_FPS = 60
CAMERA_FPS = 30
OUTPUT_QUEUE_SIZE = 2 # should match the number of workers

class WebcamApp:
    def __init__(self):
        # Initialize logger
        self.logger = TraceLogger("local", "webcam_display")
        
        # Initialize ZMQ context
        self.context = zmq.Context()
        
        # Initialize threading events
        self.shutdown = threading.Event()
        
        # Initialize frame buffer for batching
        self.frame_buffer = []
        
        # Initialize frame queue for display
        self.frame_queue = Queue(maxsize=10)
        
        # Add frame counter for pacing
        self.render_frame_counter = 0
        
        # Initialize capture thread
        self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
        
        # Create fullscreen window directly
        self.window = pyglet.window.Window(fullscreen=True)
        self.window.event(self.on_draw)

        # Initialize ZMQ socket for collection
        self.collect_socket = self.context.socket(zmq.PULL)
        ipc_path = os.path.join(os.getcwd(), ".collect_socket")
        self.collect_socket.bind(f"ipc://{ipc_path}")
        self.collect_socket.setsockopt(zmq.RCVTIMEO, 1000)
        self.collect_socket.setsockopt(zmq.LINGER, 0)
        
        self.recent_timestamp = 0

        # Warm up Pyglet's image handling
        self.logger.startEvent("pyglet_warmup")
        dummy_data = np.zeros((TARGET_SIZE, TARGET_SIZE, 3), dtype=np.uint8)
        pyglet.image.ImageData(
            TARGET_SIZE, TARGET_SIZE, 'RGB',
            dummy_data.tobytes(),
            pitch=TARGET_SIZE * 3
        )
        self.logger.stopEvent("pyglet_warmup")

    def capture_loop(self):
        # Initialize webcam
        cap = cv2.VideoCapture(10)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        
        # Initialize ZMQ socket for distribution
        socket = self.context.socket(zmq.PUSH)
        socket.set_hwm(1)
        ipc_path = os.path.join(os.getcwd(), ".distribute_socket")
        socket.bind(f"ipc://{ipc_path}")
        socket.setsockopt(zmq.LINGER, 0)
        
        try:
            while not self.shutdown.is_set():
                
                # Start a new flow for this frame
                self.logger.startEvent("capture_frame")
                ret, frame = cap.read()
                timestamp = str(time.time())
                self.logger.stopEvent("capture_frame")
                if not ret:
                    continue
                
                h, w = frame.shape[:2]
                start_x = (w - TARGET_SIZE) // 2
                start_y = (h - TARGET_SIZE) // 2
                cropped = frame[start_y:start_y+TARGET_SIZE, start_x:start_x+TARGET_SIZE]
                cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                
                # Add frame to buffer
                self.frame_buffer.append((timestamp, cropped))
                
                # If we have 2 frames, send them as a batch
                if len(self.frame_buffer) == 2:
                    self.logger.startEvent("send_frame_batch")
                    try:
                        # Prepare batch data
                        timestamps = [f[0] for f in self.frame_buffer]
                        frames = [f[1].tobytes() for f in self.frame_buffer]
                        
                        socket.send_multipart([
                            *[t.encode() for t in timestamps],
                            *frames,
                            "a psychedelic landscape".encode()
                        ], flags=zmq.DONTWAIT)
                        
                    except zmq.Again:
                        self.logger.instantEvent("frame_batch_dropped_hwm_reached")
                        pass  # Drop batch if HWM reached
                    
                    self.logger.stopEvent("send_frame_batch")
                    self.frame_buffer.clear()
                
        finally:
            cap.release()
            socket.close()

    def on_draw(self):
        # Try to get the latest frame batch
        try:
            # Receive batch of 2 frames
            multipart_msg = self.collect_socket.recv_multipart(flags=zmq.NOBLOCK)
            
            # First two elements are timestamps, next two are frame data
            timestamp_bytes1, timestamp_bytes2, frame_data1, frame_data2 = multipart_msg
            
            self.logger.startEvent("process_frames")
            timestamp1 = float(timestamp_bytes1.decode())
            timestamp2 = float(timestamp_bytes2.decode())
            
            # Process both frames and add them to queue in order
            frames = [
                (timestamp1, frame_data1),
                (timestamp2, frame_data2)
            ]
            # Sort by timestamp
            frames.sort(key=lambda x: x[0])
            
            for timestamp, frame_data in frames:
                # Drop frames older than our most recent processed frame
                if timestamp <= self.recent_timestamp:
                    self.logger.instantEvent("frame_dropped_out_of_order")
                    print(f"dropping out-of-order frame: {1000*(self.recent_timestamp - timestamp):.1f}ms late")
                    continue
                
                self.recent_timestamp = timestamp
                
                # Try to add to queue, skip if full
                try:
                    self.frame_queue.put_nowait((timestamp, frame_data))
                except Queue.Full:
                    self.logger.instantEvent("frame_dropped_queue_full")
                    pass
            
            self.logger.stopEvent("process_frames")
            
        except zmq.Again:
            pass
        
        # Check if we should process a frame based on queue size and counter
        should_process = False
        queue_size = self.frame_queue.qsize()
        
        if queue_size > 4:  # More than 2 batches (4 frames) in queue
            should_process = True
        else:
            # Increment and wrap counter
            self.render_frame_counter = (self.render_frame_counter + 1) % 2
            should_process = self.render_frame_counter == 0
        
        # Try to draw the oldest frame from the queue if we should process
        if should_process:
            try:
                timestamp, frame_data = self.frame_queue.get_nowait()
                
                self.logger.startEvent("update_frame")
                img = np.frombuffer(frame_data, dtype=np.uint8).reshape(TARGET_SIZE, TARGET_SIZE, 3)
                
                # Calculate scaling and position to center the image
                window_width = self.window.width
                window_height = self.window.height
                
                # Calculate scale factor to fit the window while maintaining aspect ratio
                scale_x = window_width / TARGET_SIZE
                scale_y = window_height / TARGET_SIZE
                scale = min(scale_x, scale_y)
                
                # Calculate centered position
                x = (window_width - TARGET_SIZE * scale) / 2
                scaled_height = TARGET_SIZE * scale
                y = TARGET_SIZE
                
                # Get a flipped version of the image
                pyglet_image = pyglet.image.ImageData(
                    TARGET_SIZE, TARGET_SIZE, 'RGB', img.tobytes(), pitch=TARGET_SIZE * 3
                )
                flipped_frame = pyglet_image.get_texture().get_transform(flip_y=True)
                self.logger.stopEvent("update_frame")
            
                # Draw the scaled, centered, and flipped image
                self.logger.startEvent("blit_frame")
                flipped_frame.blit(x, y, width=TARGET_SIZE * scale, height=scaled_height)
                self.logger.stopEvent("blit_frame")
            
            except Empty:
                pass  # No frames to draw

    def on_key_press(self, symbol, modifiers):
        if symbol == pyglet.window.key.ESCAPE:
            self.shutdown.set()
            pyglet.app.exit()

    def run(self):
        # Start capture thread
        self.capture_thread.start()
        
        # Run Pyglet event loop
        pyglet.app.run()
        
        # Cleanup on exit
        self.shutdown.set()
        self.capture_thread.join()
        self.context.destroy()
        self.collect_socket.close()

if __name__ == '__main__':
    app = WebcamApp()
    app.run() 
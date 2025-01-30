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

# Constants
CAPTURE_WIDTH = 1920
CAPTURE_HEIGHT = 1080
TARGET_SIZE = 1024
FPS = 24
OUTPUT_QUEUE_SIZE = 2

class WebcamApp:
    def __init__(self):
        # Initialize ZMQ context
        self.context = zmq.Context()
        
        # Initialize queues and threading events
        self.processed_frames = Queue()
        self.shutdown = threading.Event()
        
        # Initialize capture thread
        self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
        self.collect_thread = threading.Thread(target=self.collect_loop, daemon=True)
        
        # Create fullscreen window directly
        self.window = pyglet.window.Window(fullscreen=True)
        self.window.event(self.on_draw)
        
        # Current frame to display
        self.current_frame = None
        
        # Schedule frame updates
        pyglet.clock.schedule_interval(self.update, 1.0/FPS)

    def capture_loop(self):
        # Initialize webcam
        cap = cv2.VideoCapture(10)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, FPS)
        
        # Initialize ZMQ socket for distribution
        socket = self.context.socket(zmq.PUSH)
        socket.set_hwm(2)
        ipc_path = os.path.join(os.getcwd(), ".distribute_socket")
        socket.bind(f"ipc://{ipc_path}")
        socket.setsockopt(zmq.LINGER, 0)
        
        try:
            while not self.shutdown.is_set():
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Get center crop
                h, w = frame.shape[:2]
                start_x = (w - TARGET_SIZE) // 2
                start_y = (h - TARGET_SIZE) // 2
                cropped = frame[start_y:start_y+TARGET_SIZE, start_x:start_x+TARGET_SIZE]
                
                # Encode image
                _, img_encoded = cv2.imencode('.jpg', cropped)
                img_bytes = img_encoded.tobytes()
                
                # Send frame with timestamp
                timestamp = time.time()
                try:
                    socket.send_multipart([
                        str(timestamp).encode(),
                        img_bytes,
                        "a psychedelic landscape".encode()  # Default prompt
                    ], flags=zmq.DONTWAIT)
                except zmq.Again:
                    pass  # Drop frame if HWM reached
                
        finally:
            cap.release()
            socket.close()

    def collect_loop(self):
        # Initialize ZMQ socket for collection
        socket = self.context.socket(zmq.PULL)
        ipc_path = os.path.join(os.getcwd(), ".collect_socket")
        socket.bind(f"ipc://{ipc_path}")
        socket.setsockopt(zmq.RCVTIMEO, 1000)
        socket.setsockopt(zmq.LINGER, 0)
        
        recent_timestamp = 0
        frame_queue = []  # Priority queue using heapq
        
        while not self.shutdown.is_set():
            try:
                timestamp_bytes, frame = socket.recv_multipart()
                timestamp = float(timestamp_bytes)
                
                # Drop frames older than our most recent processed frame
                if timestamp <= recent_timestamp:
                    print(f"dropping out-of-order frame: {1000*(recent_timestamp - timestamp):.1f}ms late")
                    continue
                
                # Add frame to priority queue
                heapq.heappush(frame_queue, (timestamp, frame))
                
                # If we have more than OUTPUT_QUEUE_SIZE frames, process the oldest one
                if len(frame_queue) > OUTPUT_QUEUE_SIZE:
                    oldest_timestamp, oldest_frame = heapq.heappop(frame_queue)
                    # Convert frame bytes to image
                    img_array = np.frombuffer(oldest_frame, np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    # Convert BGR to RGB for Pyglet
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    # Create Pyglet image
                    height, width = img.shape[:2]
                    pyglet_image = pyglet.image.ImageData(
                        width, height, 'RGB', img.tobytes(), pitch=width * 3
                    )
                    self.processed_frames.put(pyglet_image)
                    recent_timestamp = oldest_timestamp
                    
            except zmq.Again:
                continue
        
        socket.close()

    def update(self, dt):
        try:
            # Get latest processed frame
            frame = self.processed_frames.get_nowait()
            self.current_frame = frame
        except Empty:
            pass

    def on_draw(self):
        self.window.clear()
        if self.current_frame:
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
            flipped_frame = self.current_frame.get_texture().get_transform(flip_y=True)
            
            # Draw the scaled, centered, and flipped image
            flipped_frame.blit(x, y, width=TARGET_SIZE * scale, height=scaled_height)

    def on_key_press(self, symbol, modifiers):
        if symbol == pyglet.window.key.ESCAPE:
            self.shutdown.set()
            pyglet.app.exit()

    def run(self):
        # Start threads
        self.capture_thread.start()
        self.collect_thread.start()
        
        # Run Pyglet event loop
        pyglet.app.run()
        
        # Cleanup on exit
        self.shutdown.set()
        self.capture_thread.join()
        self.collect_thread.join()
        self.context.destroy()

if __name__ == '__main__':
    app = WebcamApp()
    app.run() 
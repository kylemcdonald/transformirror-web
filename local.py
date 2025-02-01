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
            frame_count = 0
            while not self.shutdown.is_set():
                
                # Start a new flow for this frame
                self.logger.startEvent("capture_frame")
                ret, frame = cap.read()
                timestamp = str(time.time())
                self.logger.stopEvent("capture_frame")
                if not ret:
                    continue
                
                frame_count += 1
                if frame_count % 3 == 0:
                    continue
                
                h, w = frame.shape[:2]
                self.logger.startEvent("send_frame")
                h, w = frame.shape[:2]
                start_x = (w - TARGET_SIZE) // 2
                start_y = (h - TARGET_SIZE) // 2
                cropped = frame[start_y:start_y+TARGET_SIZE, start_x:start_x+TARGET_SIZE]
                cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                try:
                    socket.send_multipart([
                        timestamp.encode(),
                        cropped.tobytes(),
                        "a psychedelic landscape".encode()
                    ], flags=zmq.DONTWAIT)
                except zmq.Again:
                    self.logger.instantEvent("frame_dropped_hwm_reached")
                    pass  # Drop frame if HWM reached
                self.logger.stopEvent("send_frame")
                
        finally:
            cap.release()
            socket.close()

    def on_draw(self):
        # self.window.clear()
        
        # Try to get the latest frame
        try:
            timestamp_bytes, frame_data = self.collect_socket.recv_multipart(flags=zmq.NOBLOCK)
            
            self.logger.startEvent("update_frame")
            timestamp = timestamp_bytes.decode()
            
            # Drop frames older than our most recent processed frame
            if float(timestamp) <= self.recent_timestamp:
                self.logger.instantEvent("frame_dropped_out_of_order")
                print(f"dropping out-of-order frame: {1000*(self.recent_timestamp - float(timestamp)):.1f}ms late")
                self.logger.stopEvent("update_frame")
                return
            
            self.recent_timestamp = float(timestamp)
            
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
            
        except zmq.Again:
            pass

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
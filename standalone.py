import cv2
import pyglet
import zmq
import threading
import time
import heapq
import numpy as np
import os
import torch
from queue import Queue, Empty
from typing import Optional
from trace_logger import TraceLogger
from diffusion_processor import DiffusionProcessor

# Constants
CAPTURE_WIDTH = 1920
CAPTURE_HEIGHT = 1080
TARGET_SIZE = 1024
APP_FPS = 60
CAMERA_FPS = 15
OUTPUT_QUEUE_SIZE = 2  # should match the number of workers

class Worker(threading.Thread):
    def __init__(self, gpu_id: int = 0):
        super().__init__(daemon=True)
        self.process_name = f"worker_gpu{gpu_id}"
        self.logger = TraceLogger("worker", self.process_name)
        self.running = True
        self.processor = DiffusionProcessor(gpu_id=gpu_id)
        
        # Setup ZMQ sockets
        self.context = zmq.Context()
        self.pull_socket = self.setup_socket(zmq.PULL, ".distribute_socket")
        self.push_socket = self.setup_socket(zmq.PUSH, ".collect_socket")
    
    def setup_socket(self, socket_type, socket_name):
        socket = self.context.socket(socket_type)
        if socket_type == zmq.PULL:
            socket.set_hwm(1)
        ipc_path = os.path.join(os.getcwd(), socket_name)
        socket.connect(f"ipc://{ipc_path}")
        socket.setsockopt(zmq.RCVTIMEO if socket_type == zmq.PULL else zmq.SNDTIMEO, 1000)
        socket.setsockopt(zmq.LINGER, 0)
        return socket
    
    def run(self):
        try:
            while self.running:
                try:
                    # Receive and validate frame
                    timestamp_bytes, frame_data, prompt = self.pull_socket.recv_multipart()
                    timestamp = timestamp_bytes.decode()
                    
                    if not self.check_frame_delay(timestamp):
                        continue
                    
                    # Process frame through pipeline
                    with self.logger.event_scope("preprocess_frame"):
                        img = self.preprocess_image(frame_data)
                    
                    with self.logger.event_scope("inference"):
                        processed_img = self.processor(img, prompt)
                    
                    with self.logger.event_scope("postprocess_image"):
                        processed_img = np.uint8(processed_img * 255)
                    
                    with self.logger.event_scope("send_processed_frame"):
                        self.push_socket.send_multipart([timestamp.encode(), processed_img.tobytes()])
                    
                except zmq.Again:
                    continue
                except Exception as e:
                    print(f"Error processing frame: {e}")
                    continue
        finally:
            self.shutdown()
    
    def check_frame_delay(self, timestamp):
        delay = time.time() - float(timestamp)
        maximum_delay = 1
        if delay > maximum_delay:
            self.logger.instantEvent(f"frame_dropped_processing_delay", timestamp)
            print(f"dropping frame: {1000*delay:.1f}ms late")
            return False
        return True
    
    def preprocess_image(self, frame_data):
        img = np.frombuffer(frame_data, dtype=np.uint8).reshape(1024, 1024, 3)
        return np.float32(img) / 255
    
    def shutdown(self):
        self.running = False
        self.pull_socket.close()
        self.push_socket.close()
        self.context.destroy()

class WebcamApp:
    def __init__(self):
        # Initialize logger
        self.logger = TraceLogger("local", "webcam_display")
        
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
        pyglet.clock.schedule_interval(self.update, 1.0/APP_FPS)
        
        # Initialize workers
        self.workers = []
        self.init_workers()

    def init_workers(self):
        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            raise RuntimeError("No CUDA devices available")
        
        print(f"Initializing {num_gpus} worker(s)...")
        for gpu_id in range(num_gpus):
            worker = Worker(gpu_id)
            self.workers.append(worker)
            worker.start()
        print("All workers initialized")

    def capture_loop(self):
        # Initialize webcam
        cap = cv2.VideoCapture(10)  # Changed from 10 to 0 for default camera
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
                if frame_count % 2 == 0:
                    continue
                
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
                timestamp_bytes, frame_data = socket.recv_multipart()
                timestamp = timestamp_bytes.decode()
                
                self.logger.startEvent("receive_frame")
                # Drop frames older than our most recent processed frame
                if float(timestamp) <= recent_timestamp:
                    self.logger.instantEvent("frame_dropped_out_of_order")
                    print(f"dropping out-of-order frame: {1000*(recent_timestamp - float(timestamp)):.1f}ms late")
                    self.logger.stopEvent("receive_frame")
                    continue
                
                heapq.heappush(frame_queue, (frame_data, timestamp))
                self.logger.stopEvent("receive_frame")
                
                # If we have more than OUTPUT_QUEUE_SIZE frames, process the oldest one
                if len(frame_queue) > OUTPUT_QUEUE_SIZE:
                    oldest_frame, oldest_timestamp = heapq.heappop(frame_queue)
                    
                    # Create Pyglet image
                    self.logger.startEvent("create_pyglet_image")
                    img = np.frombuffer(oldest_frame, dtype=np.uint8).reshape(TARGET_SIZE, TARGET_SIZE, 3)
                    pyglet_image = pyglet.image.ImageData(
                        TARGET_SIZE, TARGET_SIZE, 'RGB', img.tobytes(), pitch=TARGET_SIZE * 3
                    )
                    self.logger.stopEvent("create_pyglet_image")
                    
                    # Put frame in queue for display
                    self.processed_frames.put((oldest_timestamp, pyglet_image))
                    
                    recent_timestamp = float(oldest_timestamp)
            except zmq.Again:
                continue
        
        socket.close()

    def update(self, dt):
        try:
            # Get latest processed frame
            timestamp, frame = self.processed_frames.get_nowait()
            self.logger.startEvent("update_frame")
            self.current_frame = frame
            self.current_timestamp = timestamp
            self.logger.stopEvent("update_frame")
        except Empty:
            pass

    def on_draw(self):
        self.window.clear()
        if self.current_frame:  # Only proceed if we have both frame and timestamp
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
            self.logger.startEvent("blit_frame")
            flipped_frame.blit(x, y, width=TARGET_SIZE * scale, height=scaled_height)
            self.logger.stopEvent("blit_frame")

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
        for worker in self.workers:
            worker.join()
        self.capture_thread.join()
        self.collect_thread.join()
        self.context.destroy()

if __name__ == '__main__':
    app = WebcamApp()
    app.run() 
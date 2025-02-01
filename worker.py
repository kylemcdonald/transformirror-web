import zmq
import cv2
import numpy as np
import os
import time
from trace_logger import TraceLogger
from diffusion_processor import DiffusionProcessor as Processor
import sys

maximum_delay = 1

class Worker:
    def __init__(self, gpu_id: int = 0):
        self.process_name = f"worker_gpu{gpu_id}"
        self.logger = TraceLogger("worker", self.process_name)
        self.running = True
        self.processor = Processor()
        
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
    
    def process_frame(self):
        try:
            # Receive and validate frame
            timestamp_bytes, frame_data, prompt = self.pull_socket.recv_multipart()
            timestamp = timestamp_bytes.decode()
            
            if not self.check_frame_delay(timestamp):
                return
            
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
            return
        except Exception as e:
            print(f"Error processing frame: {e}")
            return
    
    def check_frame_delay(self, timestamp):
        delay = time.time() - float(timestamp)
        if delay > maximum_delay:
            self.logger.instantEvent(f"frame_dropped_processing_delay", timestamp)
            print(f"dropping frame: {1000*delay:.1f}ms late")
            return False
        return True
    
    def preprocess_image(self, frame_data):
        img = np.frombuffer(frame_data, dtype=np.uint8).reshape(1024, 1024, 3)
        return np.float32(img) / 255
    
    def run(self):
        try:
            while self.running:
                self.process_frame()
        except KeyboardInterrupt:
            self.shutdown()
    
    def shutdown(self):
        self.running = False
        self.pull_socket.close()
        self.push_socket.close()
        self.context.destroy()

if __name__ == "__main__":
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    worker = Worker(gpu_id)
    worker.run()

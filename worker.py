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
            # Receive and validate frame batch
            multipart_msg = self.pull_socket.recv_multipart()
            
            # First two elements are timestamps, next two are frames, last is prompt
            timestamp_bytes1, timestamp_bytes2, frame_data1, frame_data2, prompt = multipart_msg
            timestamp1 = timestamp_bytes1.decode()
            timestamp2 = timestamp_bytes2.decode()
            
            # Check both frames for delay
            if not (self.check_frame_delay(timestamp1) and self.check_frame_delay(timestamp2)):
                return
            
            # Process frames through pipeline
            with self.logger.event_scope("preprocess_frame"):
                img1 = self.preprocess_image(frame_data1)
                img2 = self.preprocess_image(frame_data2)
                batch = [img1, img2]
            
            with self.logger.event_scope("inference"):
                processed_batch = self.processor(batch, prompt)
            
            with self.logger.event_scope("postprocess_image"):
                processed_batch = [np.uint8(img * 255) for img in processed_batch]
            
            with self.logger.event_scope("send_processed_frame"):
                # Send both processed frames back with their original timestamps
                self.push_socket.send_multipart([
                    timestamp_bytes1,
                    timestamp_bytes2,
                    processed_batch[0].tobytes(),
                    processed_batch[1].tobytes()
                ])
            
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
        img = np.frombuffer(frame_data, dtype=np.float32).reshape(1024, 1024, 3)
        return img
    
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

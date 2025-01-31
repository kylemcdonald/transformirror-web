import zmq
import cv2
import numpy as np
import os
import time
from trace_logger import TraceLogger
# from inversion_processor import InversionProcessor as Processor
from diffusion_processor import DiffusionProcessor as Processor
import sys

maximum_delay = 1

# Get GPU ID from command line argument or default to 0
gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
process_name = f"worker_gpu{gpu_id}"

logger = TraceLogger("worker", process_name)

context = zmq.Context()

pull_socket = context.socket(zmq.PULL)
pull_socket.set_hwm(1)
ipc_path = os.path.join(os.getcwd(), ".distribute_socket")
pull_socket.connect(f"ipc://{ipc_path}")
pull_socket.setsockopt(zmq.RCVTIMEO, 1000)
pull_socket.setsockopt(zmq.LINGER, 0)

push_socket = context.socket(zmq.PUSH)
ipc_path = os.path.join(os.getcwd(), ".collect_socket")
push_socket.connect(f"ipc://{ipc_path}")
push_socket.setsockopt(zmq.SNDTIMEO, 1000)
push_socket.setsockopt(zmq.LINGER, 0)

processor = Processor()

try:
    while True:
        try:
            logger.startEvent("process_frame")
            timestamp, frame_data, prompt = pull_socket.recv_multipart()
            delay = time.time() - float(timestamp)
            if delay > maximum_delay:
                print(f"dropping frame: {1000*delay:.1f}ms late")
                logger.stopEvent("process_frame")
                continue

            # decode the image
            logger.startEvent("decode_image")
            nparr = np.frombuffer(frame_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            logger.stopEvent("decode_image")

            processed_img = processor(img, prompt)

            # encode the image
            logger.startEvent("encode_image")
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
            processed_img = cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode(".jpg", processed_img, encode_param)
            logger.stopEvent("encode_image")

            push_socket.send_multipart([timestamp, buffer])
            logger.stopEvent("process_frame")
        except zmq.Again:
            continue
except KeyboardInterrupt:
    pass

push_socket.close()
pull_socket.close()
context.destroy()

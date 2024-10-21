import zmq
import cv2
import numpy as np
import os
import time
# from inversion_processor import InversionProcessor as Processor
from diffusion_processor import DiffusionProcessor as Processor

maximum_delay = 1

context = zmq.Context()

pull_socket = context.socket(zmq.PULL)
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
            timestamp, frame_data, prompt = pull_socket.recv_multipart()
            delay = time.time() - float(timestamp)
            if delay > maximum_delay:
                print(f"dropping frame: {1000*delay:.1f}ms late")
                continue

            # decode the image
            nparr = np.frombuffer(frame_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            processed_img = processor(img, prompt)

            # encode the image
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
            processed_img = cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode(".jpg", processed_img, encode_param)

            push_socket.send_multipart([timestamp, buffer])
        except zmq.Again:
            continue
except KeyboardInterrupt:
    pass

push_socket.close()
pull_socket.close()
context.destroy()

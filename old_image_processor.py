import queue
import threading
import torch
import numpy as np
import cv2
import time
import logging

from diffusion_processor import DiffusionProcessor

# Set up logging
logger = logging.getLogger(__name__)

class ImageProcessor:
    def __init__(self):
        self.input_queue = queue.Queue(maxsize=1)
        self.output_queue = queue.Queue()#maxsize=1)
        self.threads = []
        
        gpu_count = torch.cuda.device_count()
        logger.info(f"Total number of available GPUs: {gpu_count}")
        
        for i in range(gpu_count):
            thread = threading.Thread(target=self.process_images, args=(i,))
            logger.info(f"ImageProcessor initialized with GPU ID: {i}")
            thread.start()
            self.threads.append(thread)

    def process_images(self, gpu_id):
        processor = DiffusionProcessor(local_files_only=False, warmup="1x1024x1024x3", gpu_id=gpu_id)
        logger.info(f"DiffusionProcessor warmed up on GPU ID: {gpu_id}")
        
        inference_frame_count = 0
        while True:
            try:
                data = self.input_queue.get(timeout=1)
                if data is None:
                    self.input_queue.put(None)
                    break

                start_time = time.time()

                nparr = np.frombuffer(data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_LINEAR)

                img = np.float32(np.fliplr(img)) / 255
                filtered_img = processor.run(
                    images=[img],
                    prompt=settings.prompt,
                    num_inference_steps=2,
                    strength=0.7
                )[0]
                filtered_img = np.uint8(filtered_img * 255)

                encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), settings.jpeg_quality]
                _, buffer = cv2.imencode(".jpg", cv2.cvtColor(filtered_img, cv2.COLOR_RGB2BGR), encode_params)
                buffer = buffer.tobytes()

                try:
                    self.output_queue.put(buffer, block=False)
                except queue.Full:
                    logger.info(f"#{gpu_id}: dropping output frame, output queue full")
                    pass

                if inference_frame_count % 30 == 0:
                    inference_time_ms = (time.time() - start_time) * 1000
                    print(f"#{gpu_id} inference time: {inference_time_ms:.2f} ms")
                inference_frame_count += 1

            except queue.Empty:
                print(f"#{gpu_id}: no input images to process")
                pass

    def stop(self):
        self.input_queue.put(None)
        for thread in self.threads:
            thread.join()

import asyncio
import logging
from aiohttp import web, WSMsgType
import cv2
import numpy as np
import threading
import queue
from diffusion_processor import DiffusionProcessor
import time
from dataclasses import dataclass
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server")

@dataclass
class Settings:
    prompt: str = "a psychedelic landscape"
    jpeg_quality: int = 50

settings = Settings()

class ImageProcessor:
    def __init__(self):
        self.input_queue = queue.Queue(maxsize=1)
        self.output_queue = queue.Queue(maxsize=1)
        self.thread = threading.Thread(target=self.process_images)
        self.thread.start()
        self.processor = DiffusionProcessor(local_files_only=False, warmup="1x1024x1024x3")
        logger.info("ImageProcessor initialized and warmed up")

    def process_images(self):
        inference_frame_count = 0
        while True:
            try:
                data = self.input_queue.get(timeout=1)
                if data is None:
                    break

                start_time = time.time()

                nparr = np.frombuffer(data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_LINEAR)

                img = np.float32(np.fliplr(img)) / 255
                filtered_img = self.processor.run(
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
                    pass

                if inference_frame_count % 30 == 0:
                    inference_time_ms = (time.time() - start_time) * 1000
                    print(f"Inference time: {inference_time_ms:.2f} ms")
                inference_frame_count += 1

            except queue.Empty:
                pass

    def stop(self):
        self.input_queue.put(None)
        self.thread.join()

async def index(request):
    with open("index.html", "r") as f:
        content = f.read()
    return web.Response(content_type="text/html", text=content)

async def websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    processor = request.app['image_processor']

    try:
        async for msg in ws:
            if msg.type == WSMsgType.BINARY:
                if processor.input_queue.full():
                    try:
                        processor.input_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                processor.input_queue.put_nowait(msg.data)

                try:
                    buffer = processor.output_queue.get(block=False)
                    await ws.send_bytes(buffer)
                except queue.Empty:
                    pass
            elif msg.type == WSMsgType.ERROR:
                logger.error(
                    "WebSocket connection closed with exception %s", ws.exception()
                )
    finally:
        pass

    return ws

async def set_parameters(request):
    response = []
    
    if 'prompt' in request.query:
        new_prompt = request.query['prompt']
        settings.prompt = new_prompt
        response.append(f"Prompt updated to: {settings.prompt}")
    
    if 'quality' in request.query:
        try:
            new_quality = int(request.query['quality'])
            if 1 <= new_quality <= 100:
                settings.jpeg_quality = new_quality
                response.append(f"JPEG quality updated to: {settings.jpeg_quality}")
            else:
                response.append("Invalid quality value. Please use a number between 1 and 100.")
        except ValueError:
            response.append("Invalid quality value. Please provide an integer.")
    
    if not response:
        return web.Response(status=400, text="No valid parameters provided.")
    
    return web.Response(text="\n".join(response))

async def on_shutdown(app):
    for ws in set(app['websockets']):
        await ws.close(code=WSMsgType.CLOSE, message="Server shutdown")
    app['websockets'].clear()
    
    app['image_processor'].stop()

if __name__ == '__main__':
    app = web.Application()
    app['websockets'] = set()
    
    app['image_processor'] = ImageProcessor()
    
    app.router.add_get('/', index)
    app.router.add_get('/ws', websocket_handler)
    app.router.add_get('/set', set_parameters)
    app.on_shutdown.append(on_shutdown)

    web.run_app(app, access_log=None, port=8080)

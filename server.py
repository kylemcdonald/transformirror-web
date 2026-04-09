import logging
from aiohttp import web, WSMsgType
import numpy as np
import queue
import time
from dataclasses import dataclass
from typing import Optional
from queue import Queue, Empty
import threading
import zmq
import os
import ssl
import heapq
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server")
output_queue_size = 2


def load_ssl_context():
    cert_path = os.path.join(os.getcwd(), "cert.pem")
    key_path = os.path.join(os.getcwd(), "key.pem")
    if not (os.path.exists(cert_path) and os.path.exists(key_path)):
        logger.info("SSL certificates not found, starting without TLS")
        return None

    ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ssl_context.load_cert_chain(cert_path, key_path)
    logger.info("Loaded TLS certificate from %s", cert_path)
    return ssl_context

async def index(request):
    with open("index.html", "r") as f:
        content = f.read()
    return web.Response(
        content_type="text/html",
        text=content,
        headers={
            "Cache-Control": "no-store, max-age=0",
            "Pragma": "no-cache",
        },
    )

async def websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    incoming_client_frames = request.app['incoming_client_frames']
    processed_frames = request.app['processed_frames']

    try:
        async for msg in ws:
            if msg.type == WSMsgType.BINARY:
                incoming_client_frames.put(msg.data)
                while not processed_frames.empty():
                    processed_frame = processed_frames.get()
                    await ws.send_bytes(processed_frame)
            elif msg.type == WSMsgType.ERROR:
                logger.error("WebSocket connection closed with exception %s", ws.exception())
    finally:
        pass

    return ws

async def set_parameters(request):
    response = []
    
    if 'prompt' in request.query:
        new_prompt = request.query['prompt']
        request.app['settings']['prompt'] = new_prompt
        response.append(f"Prompt updated to: {new_prompt}")
    
    if not response:
        return web.Response(status=400, text="No valid parameters provided.")
    
    return web.Response(text="\n".join(response))

async def get_prompts(request):
    try:
        # Read the prompts.txt file from the data folder
        prompts_file_path = os.path.join(os.getcwd(), "data", "prompts.txt")
        
        with open(prompts_file_path, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f.readlines() if line.strip()]
        
        # Return as JSON list
        return web.json_response(prompts)
    
    except FileNotFoundError:
        logger.error(f"Prompts file not found at {prompts_file_path}")
        return web.Response(status=404, text="Prompts file not found")
    except Exception as e:
        logger.error(f"Error reading prompts file: {e}")
        return web.Response(status=500, text="Internal server error")

async def health(request):
    return web.json_response({"status": "ok"})


def distribute_loop(app):
    incoming_client_frames = app['incoming_client_frames']
    
    context = zmq.Context()
    socket = context.socket(zmq.PUSH)
    socket.set_hwm(2)
    ipc_path = os.path.join(os.getcwd(), ".distribute_socket")
    socket.bind(f"ipc://{ipc_path}")
    socket.setsockopt(zmq.LINGER, 0)

    while not app['shutdown'].is_set():
        try:
            frame = incoming_client_frames.get(timeout=1)
            timestamp = time.time()
            try:
                socket.send_multipart([str(timestamp).encode(), frame, app['settings']['prompt'].encode()], flags=zmq.DONTWAIT)
            except zmq.Again:
                # logger.info("dropping frame (HWM)")
                pass
        except Empty:
            continue
        
    socket.close()
    context.destroy()
        
def collect_loop(app):
    processed_frames = app['processed_frames']
    
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    ipc_path = os.path.join(os.getcwd(), ".collect_socket")
    socket.bind(f"ipc://{ipc_path}")
    socket.setsockopt(zmq.RCVTIMEO, 1000)
    socket.setsockopt(zmq.LINGER, 0)
    
    recent_timestamp = 0
    # Priority queue using heapq (min heap)
    frame_queue = []
    
    while not app['shutdown'].is_set():
        try:
            timestamp_bytes, frame = socket.recv_multipart()
            timestamp = float(timestamp_bytes)
            
            # Drop frames older than our most recent processed frame
            if timestamp <= recent_timestamp:
                print(f"dropping out-of-order frame: {1000*(recent_timestamp - timestamp):.1f}ms late")
                continue
                
            # Add frame to priority queue
            heapq.heappush(frame_queue, (timestamp, frame))
            
            # If we have more than 4 frames, process the oldest one
            if len(frame_queue) > output_queue_size:
                oldest_timestamp, oldest_frame = heapq.heappop(frame_queue)
                processed_frames.put(oldest_frame)
                recent_timestamp = oldest_timestamp
                
        except zmq.Again:
            continue

    socket.close()
    context.destroy()

async def on_shutdown(app):
    print("Starting shutdown")
    app['shutdown'].set()
    for ws in set(app['websockets']):
        await ws.close(code=WSMsgType.CLOSE, message="Server shutdown")
    app['websockets'].clear()
    app['distribute_thread'].join()
    app['collect_thread'].join()
    
async def on_startup(app):
    app['settings'] = {'prompt': 'a psychedelic landscape'}
    app['incoming_client_frames'] = Queue()
    app['processed_frames'] = Queue()
    app['shutdown'] = threading.Event()
    app['distribute_thread'] = threading.Thread(target=distribute_loop, args=(app,), daemon=True)
    app['collect_thread'] = threading.Thread(target=collect_loop, args=(app,), daemon=True)
    app['distribute_thread'].start()
    app['collect_thread'].start()

if __name__ == '__main__':
    app = web.Application()
    app['websockets'] = set()
    app.router.add_get('/', index)
    app.router.add_get('/ws', websocket_handler)
    app.router.add_get('/set', set_parameters)
    app.router.add_get('/prompts', get_prompts)
    app.router.add_get('/health', health)
    app.router.add_static('/audio', path='data/audio', name='audio')
    app.on_shutdown.append(on_shutdown)
    app.on_startup.append(on_startup)

    ssl_context = load_ssl_context()
    web.run_app(app, access_log=None, port=8443, ssl_context=ssl_context)

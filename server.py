import asyncio
import json
import logging
import cv2
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack, RTCConfiguration, RTCIceServer
from aiortc.contrib.media import MediaRelay
import threading
import queue
import numpy as np
from diffusion_processor import DiffusionProcessor
import time
import ssl

# Add these imports
from osc_socket import OscSocket
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pc")

pcs = set()
relay = MediaRelay()

# Modify the Settings dataclass to include a running flag
@dataclass
class Settings:
    prompt: str = "a psychedelic landscape"
    osc_host: str = "0.0.0.0"
    osc_port: int = 8888
    running: bool = True

# Modify the global variable
settings = Settings()

# Add this function to handle OSC messages
def osc_worker(settings):
    osc = OscSocket(settings.osc_host, settings.osc_port)
    while settings.running:
        msg = osc.recv()
        if msg is None:
            continue
        if msg.address == "/prompt":
            prompt = ' '.join(msg.params)
            settings.prompt = prompt
            print(f"OSC: Updated prompt to '{prompt}'")
    osc.close()  # Close the OSC socket when exiting

# Modify the filter_worker function to use settings
def filter_worker(input_queue, output_queue, settings):
    processor = DiffusionProcessor(local_files_only=False)
    while True:
        img = input_queue.get()
        if img is None:
            break
        
        img = np.float32(img) / 255
        filtered_img = processor.run(
            images=[img],
            prompt=settings.prompt,
            num_inference_steps=2,
            strength=0.7
        )[0]
        filtered_img = np.uint8(filtered_img * 255)
        
        try:
            output_queue.put(filtered_img, block=False)
        except queue.Full:
            output_queue.get()
            output_queue.put(filtered_img, block=False)

class VideoTransformTrack(VideoStreamTrack):
    """
    A video stream track that transforms frames using a diffusion model.
    """
    def __init__(self, track, settings):
        super().__init__()  # Don't forget this!
        self.track = track
        self.frame_num = 0
        self.input_queue = queue.Queue(maxsize=1)
        self.output_queue = queue.Queue(maxsize=1)
        self.filter_thread = threading.Thread(target=filter_worker, args=(self.input_queue, self.output_queue, settings))
        self.filter_thread.start()
        self.latest_frame = None
        self.settings = settings

    async def recv(self):
        frame = await self.track.recv()

        # Convert frame to numpy array
        img = frame.to_ndarray(format="rgb24")
        
        # Measure the time taken for resizing
        start_time = time.time()
        img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_LANCZOS4)
        resize_time_ms = (time.time() - start_time) * 1000  # Convert to milliseconds

        try:
            self.input_queue.put(img, block=False)
        except queue.Full:
            pass  # Skip this frame if the queue is full

        try:
            filtered_img = self.output_queue.get(block=False)
            new_frame = frame.from_ndarray(filtered_img, format="rgb24")
            new_frame.pts = frame.pts
            new_frame.time_base = frame.time_base
            self.latest_frame = new_frame
        except queue.Empty:
            pass
        
        if self.frame_num % 120 == 0:
            print(f"Resize time: {resize_time_ms:.2f} ms")
            print(f"Frame {self.frame_num} input resolution: {frame.width}x{frame.height}")
            print(f"Frame {self.frame_num} resized resolution: {img.shape[1]}x{img.shape[0]}")
            try:
                print(f"Frame {self.frame_num} output resolution: {self.latest_frame.width}x{self.latest_frame.height}")
            except AttributeError:
                print(f"Frame {self.frame_num} output resolution: None")

        self.frame_num += 1
        return self.latest_frame or frame

    def stop(self):
        self.input_queue.put(None)  # Signal the worker to stop
        self.filter_thread.join()

async def index(request):
    content = open('index.html', 'r').read()
    return web.Response(content_type='text/html', text=content)

async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params['sdp'], type=params['type'])

    # Create RTCConfiguration with the same STUN server as the frontend
    ice_servers = [
        RTCIceServer(urls="stun:stun.l.google.com:19302"),
    ]
    config = RTCConfiguration(iceServers=ice_servers)

    pc = RTCPeerConnection(configuration=config)
    pcs.add(pc)

    @pc.on('iceconnectionstatechange')
    async def on_iceconnectionstatechange():
        logger.info('ICE connection state is %s', pc.iceConnectionState)
        logger.info('Local candidates: %s', pc.localDescription.sdp)
        logger.info('Remote candidates: %s', pc.remoteDescription.sdp)
        if pc.iceConnectionState == 'failed':
            await pc.close()
            pcs.discard(pc)

    @pc.on('icegatheringstatechange')
    def on_icegatheringstatechange():
        logger.info('ICE gathering state is %s', pc.iceGatheringState)

    @pc.on('signalingstatechange')
    def on_signalingstatechange():
        logger.info('Signaling state is %s', pc.signalingState)

    @pc.on('track')
    def on_track(track):
        logger.info('Track %s received', track.kind)

        if track.kind == 'video':
            local_video = VideoTransformTrack(relay.subscribe(track), settings)
            pc.addTrack(local_video)

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    response = {'sdp': pc.localDescription.sdp, 'type': pc.localDescription.type}
    return web.Response(content_type='application/json', text=json.dumps(response))

async def on_shutdown(app):
    print("Shutting down...")
    
    # Close peer connections and stop video transform tracks
    coros = []
    for pc in pcs:
        coros.append(pc.close())
        for sender in pc.getSenders():
            if isinstance(sender.track, VideoTransformTrack):
                sender.track.stop()
    await asyncio.gather(*coros)
    pcs.clear()

    # Stop the OSC worker thread
    print("Stopping OSC worker thread...")
    settings.running = False
    app['osc_thread'].join()
    print("Application shut down successfully.")

async def set_prompt(request):
    try:
        new_prompt = request.query['prompt']
        settings.prompt = new_prompt
        return web.Response(text=f"Prompt updated to: {settings.prompt}")
    except KeyError:
        return web.Response(status=400, text="Invalid request: 'prompt' parameter is required")

if __name__ == '__main__':
    app = web.Application()
    app.router.add_get('/', index)
    app.router.add_post('/offer', offer)
    app.router.add_get('/prompt', set_prompt)
    app.on_shutdown.append(on_shutdown)
    
    ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ssl_context.load_cert_chain('cert.pem', 'key.pem')
    
    # Start the OSC worker thread
    osc_thread = threading.Thread(target=osc_worker, args=(settings,))
    osc_thread.start()
    app['osc_thread'] = osc_thread  # Store the thread in the app for later access
    
    web.run_app(app, access_log=None, port=8443, ssl_context=ssl_context)
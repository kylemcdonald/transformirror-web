import argparse
import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pyglet
from fastapi import Body, FastAPI
from fastapi.responses import HTMLResponse
from pyglet.gl import *
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer

from diffusion_processor import DiffusionProcessor


DEFAULT_CONFIG = {
    "width": 1280,
    "height": 704,
    "camera_device": "/dev/video0",
    "camera_backend": "ffmpeg",
    "camera_fps": 30,
    "display_index": 0,
    "fullscreen": True,
    "mirror": True,
    "prompt": "a cinematic mirror portrait, luminous, detailed, surreal",
    "seed": 0,
    "strength": 0.7,
    "blend": 1.0,
    "steps": 2,
    "osc_host": "0.0.0.0",
    "osc_port": 9000,
    "http_host": "0.0.0.0",
    "http_port": 8080,
}

RESOLUTION_STEP = 32
MAX_RESOLUTION_DIMENSION = 1280
MIN_RESOLUTION_DIMENSION = 32

CAMERA_MODES = [
    (1280, 720),
    (1600, 896),
    (1920, 1080),
    (2560, 1440),
    (3840, 2160),
    (4096, 2160),
]


def clamp(value, low, high):
    return max(low, min(high, value))


def clamp_resolution_dimension(value):
    value = int(float(value))
    value = int(clamp(value, MIN_RESOLUTION_DIMENSION, MAX_RESOLUTION_DIMENSION))
    value = (value // RESOLUTION_STEP) * RESOLUTION_STEP
    return max(MIN_RESOLUTION_DIMENSION, value)


def normalize_resolution(width, height):
    return clamp_resolution_dimension(width), clamp_resolution_dimension(height)


def choose_camera_mode(target_width, target_height):
    target_aspect = target_width / target_height
    candidates = []
    for camera_width, camera_height in CAMERA_MODES:
        if camera_width / camera_height >= target_aspect:
            crop_height = camera_height
            crop_width = int(round(crop_height * target_aspect))
        else:
            crop_width = camera_width
            crop_height = int(round(crop_width / target_aspect))

        if crop_width <= camera_width and crop_height <= camera_height:
            if crop_width >= target_width and crop_height >= target_height:
                candidates.append((camera_width * camera_height, camera_width, camera_height))

    if candidates:
        _, camera_width, camera_height = min(candidates)
        return camera_width, camera_height

    return CAMERA_MODES[-1]


def center_crop_and_resize(frame, target_width, target_height):
    frame_height, frame_width = frame.shape[:2]
    target_aspect = target_width / target_height
    frame_aspect = frame_width / frame_height

    if frame_aspect >= target_aspect:
        crop_height = frame_height
        crop_width = int(round(crop_height * target_aspect))
    else:
        crop_width = frame_width
        crop_height = int(round(crop_width / target_aspect))

    crop_width = min(crop_width, frame_width)
    crop_height = min(crop_height, frame_height)
    x = max(0, (frame_width - crop_width) // 2)
    y = max(0, (frame_height - crop_height) // 2)
    cropped = frame[y : y + crop_height, x : x + crop_width]

    if crop_width == target_width and crop_height == target_height:
        return cropped.copy(), (crop_width, crop_height)

    interpolation = cv2.INTER_AREA if crop_width >= target_width and crop_height >= target_height else cv2.INTER_LINEAR
    resized = cv2.resize(cropped, (target_width, target_height), interpolation=interpolation)
    return resized, (crop_width, crop_height)


def parse_camera_device(value):
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.startswith("/dev/video"):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


class RuntimeState:
    def __init__(self, config, config_path):
        self.config = config
        self.config_path = config_path
        self.lock = threading.RLock()
        self.frame_lock = threading.RLock()
        self.stop_event = threading.Event()

        width, height = normalize_resolution(config["width"], config["height"])
        self.config["width"] = width
        self.config["height"] = height
        self.width = width
        self.height = height
        self.resolution_generation = 0

        self.prompt = str(config["prompt"])
        self.seed = int(config["seed"])
        self.strength = float(config["strength"])
        self.blend = float(config["blend"])
        self.steps = int(config["steps"])

        self.raw_frame = None
        self.raw_frame_id = 0
        self.processed_frame = None
        self.processed_frame_id = 0

        self.model_ready = False
        self.camera_ready = False
        self.last_error = ""
        self.camera_fps = 0.0
        self.display_fps = 0.0
        self.diffusion_ms = 0.0
        self.diffusion_fps = 0.0
        self.last_frame_age_ms = 0.0
        self.camera_source_width = 0
        self.camera_source_height = 0
        self.camera_crop_width = 0
        self.camera_crop_height = 0
        self.display_width = width
        self.display_height = height
        self.last_screenshot = ""
        self.started_at = time.time()

    def controls(self):
        with self.lock:
            return {
                "prompt": self.prompt,
                "seed": self.seed,
                "strength": self.strength,
                "blend": self.blend,
                "steps": self.steps,
            }

    def resolution(self):
        with self.lock:
            return self.width, self.height, self.resolution_generation

    def parse_resolution_update(self, updates):
        width = updates.get("width")
        height = updates.get("height")
        resolution = updates.get("resolution")

        if resolution is not None and (width is None or height is None):
            if isinstance(resolution, str):
                clean = resolution.lower().replace(",", "x").replace(" ", "")
                if "x" in clean:
                    parts = clean.split("x", 1)
                    width = width if width is not None else parts[0]
                    height = height if height is not None else parts[1]
            elif isinstance(resolution, (list, tuple)) and len(resolution) >= 2:
                width = width if width is not None else resolution[0]
                height = height if height is not None else resolution[1]

        if width is None and height is None:
            return None

        with self.lock:
            current_width = self.width
            current_height = self.height

        return normalize_resolution(
            current_width if width is None else width,
            current_height if height is None else height,
        )

    def persist_resolution(self):
        data = dict(self.config)
        data["width"] = self.width
        data["height"] = self.height
        temp_path = self.config_path.with_suffix(self.config_path.suffix + ".tmp")
        temp_path.write_text(json.dumps(data, indent=2) + "\n")
        temp_path.replace(self.config_path)

    def set_resolution(self, width, height):
        width, height = normalize_resolution(width, height)
        with self.lock:
            if width == self.width and height == self.height:
                return {}

            self.width = width
            self.height = height
            self.config["width"] = width
            self.config["height"] = height
            self.resolution_generation += 1
            generation = self.resolution_generation
            self.model_ready = False
            self.camera_ready = False
            self.diffusion_ms = 0.0
            self.diffusion_fps = 0.0
            self.last_frame_age_ms = 0.0

            with self.frame_lock:
                self.raw_frame = None
                self.processed_frame = None
                self.raw_frame_id += 1
                self.processed_frame_id += 1

            self.persist_resolution()

        changed = {"width": width, "height": height, "resolution_generation": generation}
        print(f"resolution updated: {width}x{height}", flush=True)
        return changed

    def set_camera_status(self, source_width, source_height, crop_width, crop_height):
        with self.lock:
            self.camera_source_width = int(source_width)
            self.camera_source_height = int(source_height)
            self.camera_crop_width = int(crop_width)
            self.camera_crop_height = int(crop_height)

    def clear_error(self):
        with self.lock:
            self.last_error = ""

    def update_controls(self, **updates):
        changed = {}
        resolution = self.parse_resolution_update(updates)
        if resolution is not None:
            changed.update(self.set_resolution(*resolution))

        with self.lock:
            if "prompt" in updates and updates["prompt"] is not None:
                self.prompt = str(updates["prompt"])
                changed["prompt"] = self.prompt
            if "seed" in updates and updates["seed"] is not None:
                self.seed = int(updates["seed"])
                changed["seed"] = self.seed
            if "strength" in updates and updates["strength"] is not None:
                self.strength = clamp(float(updates["strength"]), 0.0, 1.0)
                changed["strength"] = self.strength
            if "blend" in updates and updates["blend"] is not None:
                self.blend = clamp(float(updates["blend"]), 0.0, 1.0)
                changed["blend"] = self.blend
            if "steps" in updates and updates["steps"] is not None:
                self.steps = int(clamp(int(updates["steps"]), 1, 8))
                changed["steps"] = self.steps
        if changed:
            print(f"controls updated: {changed}", flush=True)
        return changed

    def snapshot(self):
        with self.lock, self.frame_lock:
            return {
                "controls": {
                    "prompt": self.prompt,
                    "seed": self.seed,
                    "strength": self.strength,
                    "blend": self.blend,
                    "steps": self.steps,
                },
                "config": {
                    "width": self.width,
                    "height": self.height,
                    "camera_device": self.config["camera_device"],
                    "camera_backend": self.config["camera_backend"],
                    "camera_fps": self.config["camera_fps"],
                    "display_index": self.config["display_index"],
                    "osc_port": self.config["osc_port"],
                    "http_port": self.config["http_port"],
                },
                "status": {
                    "model_ready": self.model_ready,
                    "camera_ready": self.camera_ready,
                    "last_error": self.last_error,
                    "uptime_s": round(time.time() - self.started_at, 1),
                },
                "stats": {
                    "camera_fps": round(self.camera_fps, 2),
                    "display_fps": round(self.display_fps, 2),
                    "diffusion_ms": round(self.diffusion_ms, 1),
                    "diffusion_fps": round(self.diffusion_fps, 2),
                    "last_frame_age_ms": round(self.last_frame_age_ms, 1),
                    "raw_frame_id": self.raw_frame_id,
                    "processed_frame_id": self.processed_frame_id,
                    "last_screenshot": self.last_screenshot,
                    "camera_source_width": self.camera_source_width,
                    "camera_source_height": self.camera_source_height,
                    "camera_crop_width": self.camera_crop_width,
                    "camera_crop_height": self.camera_crop_height,
                    "resolution_generation": self.resolution_generation,
                },
            }

    def set_error(self, message):
        with self.lock:
            self.last_error = str(message)
        print(f"error: {message}", flush=True)

    def set_display_size(self, width, height):
        with self.lock:
            self.display_width = int(width)
            self.display_height = int(height)

    def current_display_frame(self):
        with self.lock:
            blend = self.blend
            display_width = self.display_width
            display_height = self.display_height
        with self.frame_lock:
            raw = None if self.raw_frame is None else self.raw_frame.copy()
            processed = None if self.processed_frame is None else self.processed_frame.copy()

        if raw is None:
            return None
        if processed is None or blend <= 0:
            frame = raw
        elif blend >= 1:
            frame = processed
        else:
            frame = cv2.addWeighted(raw, 1.0 - blend, processed, blend, 0)

        src_h, src_w = frame.shape[:2]
        scale = min(display_width / src_w, display_height / src_h)
        draw_w = max(1, int(src_w * scale))
        draw_h = max(1, int(src_h * scale))
        resized = cv2.resize(frame, (draw_w, draw_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((display_height, display_width, 3), dtype=np.uint8)
        x = (display_width - draw_w) // 2
        y = (display_height - draw_h) // 2
        canvas[y : y + draw_h, x : x + draw_w] = resized
        return canvas

    def save_screenshot(self, path):
        frame = self.current_display_frame()
        if frame is None:
            self.set_error("screenshot requested before a frame was available")
            return False
        out_path = Path(path).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(out_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        if ok:
            with self.lock:
                self.last_screenshot = str(out_path)
            print(f"screenshot saved: {out_path}", flush=True)
        else:
            self.set_error(f"failed to save screenshot: {out_path}")
        return ok


class CameraThread(threading.Thread):
    def __init__(self, state):
        super().__init__(daemon=True)
        self.state = state
        self.config = state.config

    def open_camera(self, camera_width, camera_height):
        device = parse_camera_device(self.config["camera_device"])
        self.configure_v4l2(device)
        cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
        cap.set(cv2.CAP_PROP_FPS, int(self.config["camera_fps"]))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def configure_v4l2(self, device):
        if not (isinstance(device, str) and device.startswith("/dev/video")):
            return
        subprocess.run(
            ["v4l2-ctl", "-d", device, "--set-ctrl=exposure_dynamic_framerate=0"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def open_ffmpeg(self, camera_width, camera_height):
        device = str(self.config["camera_device"])
        self.configure_v4l2(device)
        fps = int(self.config["camera_fps"])
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-fflags",
            "nobuffer",
            "-flags",
            "low_delay",
            "-f",
            "v4l2",
            "-input_format",
            "mjpeg",
            "-framerate",
            str(fps),
            "-video_size",
            f"{camera_width}x{camera_height}",
            "-i",
            device,
            "-an",
            "-sn",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ]
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def run_ffmpeg(self):
        mirror = bool(self.config["mirror"])
        target_width, target_height, generation = self.state.resolution()
        camera_width, camera_height = choose_camera_mode(target_width, target_height)
        frame_bytes = camera_width * camera_height * 3
        proc = None
        count = 0
        last_count = 0
        last_t = time.perf_counter()

        while not self.state.stop_event.is_set():
            current_width, current_height, current_generation = self.state.resolution()
            current_camera_width, current_camera_height = choose_camera_mode(current_width, current_height)
            if current_generation != generation or (current_camera_width, current_camera_height) != (camera_width, camera_height):
                generation = current_generation
                target_width = current_width
                target_height = current_height
                camera_width = current_camera_width
                camera_height = current_camera_height
                frame_bytes = camera_width * camera_height * 3
                if proc is not None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    proc = None

            if proc is None or proc.poll() is not None:
                if proc is not None:
                    proc.kill()
                try:
                    proc = self.open_ffmpeg(camera_width, camera_height)
                    self.state.camera_ready = True
                    self.state.clear_error()
                    print(
                        f"ffmpeg camera opened {self.config['camera_device']} "
                        f"{camera_width}x{camera_height}@{self.config['camera_fps']} "
                        f"for {target_width}x{target_height}",
                        flush=True,
                    )
                except Exception as exc:
                    self.state.camera_ready = False
                    self.state.set_error(f"ffmpeg camera failed: {exc}")
                    time.sleep(1.0)
                    continue

            frame_data = proc.stdout.read(frame_bytes)
            if len(frame_data) != frame_bytes:
                self.state.camera_ready = False
                self.state.set_error("ffmpeg camera stream ended; reconnecting")
                proc.kill()
                proc = None
                time.sleep(0.2)
                continue

            frame = np.frombuffer(frame_data, dtype=np.uint8).reshape(camera_height, camera_width, 3)
            if mirror:
                frame = cv2.flip(frame, 1)
            frame, crop_size = center_crop_and_resize(frame, target_width, target_height)
            self.state.set_camera_status(camera_width, camera_height, crop_size[0], crop_size[1])

            with self.state.frame_lock:
                self.state.raw_frame = frame.copy()
                self.state.raw_frame_id += 1

            count += 1
            now = time.perf_counter()
            if now - last_t >= 2.0:
                self.state.camera_fps = (count - last_count) / (now - last_t)
                last_count = count
                last_t = now

        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                proc.kill()

    def run(self):
        if self.config.get("camera_backend") == "ffmpeg":
            self.run_ffmpeg()
            return

        mirror = bool(self.config["mirror"])
        target_width, target_height, generation = self.state.resolution()
        camera_width, camera_height = choose_camera_mode(target_width, target_height)
        cap = None
        count = 0
        last_count = 0
        last_t = time.perf_counter()

        while not self.state.stop_event.is_set():
            current_width, current_height, current_generation = self.state.resolution()
            current_camera_width, current_camera_height = choose_camera_mode(current_width, current_height)
            if current_generation != generation or (current_camera_width, current_camera_height) != (camera_width, camera_height):
                generation = current_generation
                target_width = current_width
                target_height = current_height
                camera_width = current_camera_width
                camera_height = current_camera_height
                if cap is not None:
                    cap.release()
                    cap = None

            if cap is None or not cap.isOpened():
                if cap is not None:
                    cap.release()
                cap = self.open_camera(camera_width, camera_height)
                if not cap.isOpened():
                    self.state.camera_ready = False
                    self.state.set_error(f"camera unavailable: {self.config['camera_device']}")
                    time.sleep(1.0)
                    continue
                self.state.camera_ready = True
                self.state.clear_error()
                print(
                    f"camera opened {self.config['camera_device']} "
                    f"{camera_width}x{camera_height}@{self.config['camera_fps']} "
                    f"for {target_width}x{target_height}",
                    flush=True,
                )

            ok, frame = cap.read()
            if not ok:
                self.state.camera_ready = False
                self.state.set_error("camera read failed; reconnecting")
                cap.release()
                cap = None
                time.sleep(0.2)
                continue

            if mirror:
                frame = cv2.flip(frame, 1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame, crop_size = center_crop_and_resize(frame, target_width, target_height)
            self.state.set_camera_status(camera_width, camera_height, crop_size[0], crop_size[1])

            with self.state.frame_lock:
                self.state.raw_frame = frame
                self.state.raw_frame_id += 1

            count += 1
            now = time.perf_counter()
            if now - last_t >= 2.0:
                self.state.camera_fps = (count - last_count) / (now - last_t)
                last_count = count
                last_t = now

        if cap is not None:
            cap.release()


class InferenceThread(threading.Thread):
    def __init__(self, state):
        super().__init__(daemon=True)
        self.state = state
        self.config = state.config

    def run(self):
        processor = None
        active_generation = None
        width = None
        height = None
        last_raw_id = -1
        count = 0
        last_count = 0
        last_t = time.perf_counter()

        while not self.state.stop_event.is_set():
            current_width, current_height, current_generation = self.state.resolution()
            if processor is None or current_generation != active_generation:
                if processor is not None:
                    del processor
                    processor = None
                    try:
                        import torch

                        torch.cuda.empty_cache()
                    except Exception:
                        pass

                width = current_width
                height = current_height
                active_generation = current_generation
                last_raw_id = -1
                count = 0
                last_count = 0
                last_t = time.perf_counter()
                with self.state.lock:
                    self.state.model_ready = False
                    self.state.diffusion_fps = 0.0
                    self.state.diffusion_ms = 0.0

                warmup = f"1x{height}x{width}x3"
                print(f"loading diffusion model for {width}x{height}", flush=True)
                try:
                    processor = DiffusionProcessor(warmup=warmup, local_files_only=True, gpu_id=0)
                except Exception as exc:
                    self.state.set_error(f"model load failed for {width}x{height}: {exc}")
                    time.sleep(1.0)
                    continue

                self.state.model_ready = True
                self.state.clear_error()

            with self.state.frame_lock:
                raw_id = self.state.raw_frame_id
                raw = None if self.state.raw_frame is None else self.state.raw_frame.copy()

            if raw is None or raw_id == last_raw_id:
                time.sleep(0.002)
                continue

            if raw.shape[1] != width or raw.shape[0] != height:
                time.sleep(0.002)
                continue

            last_raw_id = raw_id
            controls = self.state.controls()
            image = raw.astype(np.float32) / 255.0

            started = time.perf_counter()
            try:
                result = processor.run(
                    images=[image],
                    prompt=controls["prompt"],
                    num_inference_steps=controls["steps"],
                    strength=controls["strength"],
                    seed=controls["seed"],
                )
                processed = np.clip(result[0] * 255.0, 0, 255).astype(np.uint8)
            except Exception as exc:
                self.state.set_error(f"inference failed: {exc}")
                time.sleep(0.1)
                continue

            elapsed_ms = (time.perf_counter() - started) * 1000.0
            with self.state.frame_lock:
                self.state.processed_frame = processed
                self.state.processed_frame_id += 1
                self.state.diffusion_ms = elapsed_ms
                self.state.last_frame_age_ms = (time.perf_counter() - started) * 1000.0

            count += 1
            now = time.perf_counter()
            if now - last_t >= 2.0:
                self.state.diffusion_fps = (count - last_count) / (now - last_t)
                last_count = count
                last_t = now
                print(
                    f"diffusion {self.state.diffusion_ms:.1f} ms, "
                    f"{self.state.diffusion_fps:.2f} fps",
                    flush=True,
                )


class OscThread(threading.Thread):
    def __init__(self, state):
        super().__init__(daemon=True)
        self.state = state
        self.server = None

    def run(self):
        dispatcher = Dispatcher()
        dispatcher.set_default_handler(self.handle)
        for address in (
            "/prompt",
            "/seed",
            "/strength",
            "/blend",
            "/passthrough",
            "/resolution",
            "/screenshot",
            "/steps",
            "/width",
            "/height",
            "/transformirror/prompt",
            "/transformirror/seed",
            "/transformirror/strength",
            "/transformirror/blend",
            "/transformirror/passthrough",
            "/transformirror/resolution",
            "/transformirror/screenshot",
            "/transformirror/steps",
            "/transformirror/width",
            "/transformirror/height",
        ):
            dispatcher.map(address, self.handle)

        host = self.state.config["osc_host"]
        port = int(self.state.config["osc_port"])
        self.server = ThreadingOSCUDPServer((host, port), dispatcher)
        print(f"OSC listening on {host}:{port}", flush=True)
        self.server.serve_forever()

    def stop(self):
        if self.server is not None:
            self.server.shutdown()

    def handle(self, address, *args):
        if not args:
            return
        key = address.strip("/").split("/")[-1]
        value = args[0]
        try:
            if key == "prompt":
                self.state.update_controls(prompt=value)
            elif key == "seed":
                self.state.update_controls(seed=value)
            elif key == "strength":
                self.state.update_controls(strength=value)
            elif key == "blend":
                self.state.update_controls(blend=value)
            elif key == "passthrough":
                enabled = str(value).lower() in ("1", "true", "yes", "on")
                self.state.update_controls(blend=0.0 if enabled else 1.0)
            elif key == "resolution":
                if len(args) >= 2:
                    self.state.update_controls(width=args[0], height=args[1])
                else:
                    self.state.update_controls(resolution=value)
            elif key == "width":
                self.state.update_controls(width=value)
            elif key == "height":
                self.state.update_controls(height=value)
            elif key == "screenshot":
                self.state.save_screenshot(str(value))
            elif key == "steps":
                self.state.update_controls(steps=value)
        except Exception as exc:
            self.state.set_error(f"bad OSC {address}: {exc}")


def create_http_app(state):
    app = FastAPI()
    web_root = Path(__file__).with_name("web")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTMLResponse((web_root / "index.html").read_text())

    @app.get("/api/state")
    def get_state():
        return state.snapshot()

    @app.post("/api/state")
    def set_state(payload: dict = Body(...)):
        allowed = {
            k: payload[k]
            for k in ("prompt", "seed", "strength", "blend", "steps", "width", "height", "resolution")
            if k in payload
        }
        state.update_controls(**allowed)
        return state.snapshot()

    @app.post("/api/resolution")
    def set_resolution(payload: dict = Body(...)):
        state.update_controls(**payload)
        return state.snapshot()

    @app.post("/api/screenshot")
    def screenshot(payload: dict = Body(...)):
        state.save_screenshot(payload["path"])
        return state.snapshot()

    return app


class HttpThread(threading.Thread):
    def __init__(self, state):
        super().__init__(daemon=True)
        self.state = state

    def run(self):
        import uvicorn

        host = self.state.config["http_host"]
        port = int(self.state.config["http_port"])
        print(f"HTTP listening on http://{host}:{port}", flush=True)
        uvicorn.run(create_http_app(self.state), host=host, port=port, log_level="warning")


class DisplayApp:
    def __init__(self, state):
        self.state = state
        self.config = state.config
        self.width = int(self.config["width"])
        self.height = int(self.config["height"])
        self.texture = None
        self.last_raw_id = -1
        self.last_processed_id = -1
        self.last_blend = None
        self.last_generation = -1
        self.draw_count = 0
        self.last_draw_t = time.perf_counter()

        self.window = self.create_window()
        self.window.event(self.on_draw)
        self.window.event(self.on_key_press)

    def create_window(self):
        display = pyglet.display.get_display()
        screens = display.get_screens()
        screen = None
        if screens:
            idx = int(clamp(int(self.config["display_index"]), 0, len(screens) - 1))
            screen = screens[idx]
            print(f"using display {idx}: {screen.width}x{screen.height}", flush=True)
            self.state.set_display_size(screen.width, screen.height)

        try:
            return pyglet.window.Window(
                fullscreen=bool(self.config["fullscreen"]),
                screen=screen,
                vsync=True,
                caption="Transformirror",
            )
        except Exception as exc:
            print(f"fullscreen window failed, opening windowed: {exc}", flush=True)
            return pyglet.window.Window(
                width=self.width,
                height=self.height,
                vsync=True,
                caption="Transformirror",
            )

    def choose_frame(self):
        width, height, generation = self.state.resolution()
        with self.state.frame_lock:
            raw_id = self.state.raw_frame_id
            processed_id = self.state.processed_frame_id
            raw = None if self.state.raw_frame is None else self.state.raw_frame.copy()
            processed = None if self.state.processed_frame is None else self.state.processed_frame.copy()
        blend = self.state.controls()["blend"]

        needs_update = (
            raw_id != self.last_raw_id
            or processed_id != self.last_processed_id
            or blend != self.last_blend
            or generation != self.last_generation
        )
        if not needs_update or raw is None:
            return None

        self.last_raw_id = raw_id
        self.last_processed_id = processed_id
        self.last_blend = blend
        self.last_generation = generation
        self.width = width
        self.height = height

        if processed is None:
            return raw
        if blend <= 0:
            return raw
        if blend >= 1:
            return processed
        return cv2.addWeighted(raw, 1.0 - blend, processed, blend, 0)

    def update_texture(self, _dt):
        width, height, generation = self.state.resolution()
        if generation != self.last_generation and self.texture is not None:
            self.texture.delete()
            self.texture = None
            self.last_generation = generation
            self.width = width
            self.height = height

        frame = self.choose_frame()
        if frame is None:
            return
        frame_height, frame_width = frame.shape[:2]
        image = pyglet.image.ImageData(
            frame_width,
            frame_height,
            "RGB",
            frame.tobytes(),
            pitch=frame_width * 3,
        )
        if self.texture is not None:
            self.texture.delete()
        self.texture = image.get_texture().get_transform(flip_y=True)
        self.width = frame_width
        self.height = frame_height

    def fitted_rect(self):
        ww, wh = self.window.width, self.window.height
        scale = min(ww / self.width, wh / self.height)
        draw_w = self.width * scale
        draw_h = self.height * scale
        return (ww - draw_w) / 2, (wh - draw_h) / 2, draw_w, draw_h

    def on_draw(self):
        self.window.clear()
        if self.texture is not None:
            x, y, w, h = self.fitted_rect()
            self.texture.blit(x, y, width=w, height=h)

        self.draw_count += 1
        now = time.perf_counter()
        if now - self.last_draw_t >= 2.0:
            self.state.display_fps = self.draw_count / (now - self.last_draw_t)
            self.draw_count = 0
            self.last_draw_t = now

    def on_key_press(self, symbol, _modifiers):
        if symbol == pyglet.window.key.ESCAPE:
            self.state.stop_event.set()
            pyglet.app.exit()
        elif symbol == pyglet.window.key.R:
            self.state.update_controls(blend=0.0)
        elif symbol == pyglet.window.key.P:
            self.state.update_controls(blend=1.0)

    def run(self):
        frame_interval = 1.0 / 60.0
        while not self.state.stop_event.is_set() and not self.window.has_exit:
            started = time.perf_counter()
            self.window.switch_to()
            self.window.dispatch_events()
            self.update_texture(frame_interval)
            self.on_draw()
            self.window.flip()
            elapsed = time.perf_counter() - started
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)


def load_config(path, args):
    config = dict(DEFAULT_CONFIG)
    if path.exists():
        config.update(json.loads(path.read_text()))
    for key in ("width", "height", "camera_fps", "display_index", "osc_port", "http_port"):
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value
    if args.camera_device is not None:
        config["camera_device"] = args.camera_device
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="live_config.json")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--camera-fps", type=int)
    parser.add_argument("--camera-device")
    parser.add_argument("--display-index", type=int)
    parser.add_argument("--osc-port", type=int)
    parser.add_argument("--http-port", type=int)
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    config_path = Path(args.config)
    config = load_config(config_path, args)
    print(f"config: {json.dumps(config, sort_keys=True)}", flush=True)

    state = RuntimeState(config, config_path)

    def handle_signal(signum, _frame):
        print(f"received signal {signum}; shutting down", flush=True)
        state.stop_event.set()
        pyglet.app.exit()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    camera = CameraThread(state)
    inference = InferenceThread(state)
    osc = OscThread(state)
    http = HttpThread(state)

    camera.start()
    inference.start()
    osc.start()
    http.start()

    display = DisplayApp(state)
    try:
        display.run()
    finally:
        state.stop_event.set()
        osc.stop()


if __name__ == "__main__":
    main()

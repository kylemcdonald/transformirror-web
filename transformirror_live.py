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
    "height": 720,
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


def clamp(value, low, high):
    return max(low, min(high, value))


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
    def __init__(self, config):
        self.config = config
        self.lock = threading.RLock()
        self.frame_lock = threading.RLock()
        self.stop_event = threading.Event()

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
        self.display_width = int(config["width"])
        self.display_height = int(config["height"])
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

    def update_controls(self, **updates):
        changed = {}
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
                    "width": self.config["width"],
                    "height": self.config["height"],
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

    def open_camera(self):
        device = parse_camera_device(self.config["camera_device"])
        self.configure_v4l2(device)
        cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.config["width"]))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.config["height"]))
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

    def open_ffmpeg(self):
        device = str(self.config["camera_device"])
        self.configure_v4l2(device)
        width = int(self.config["width"])
        height = int(self.config["height"])
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
            f"{width}x{height}",
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
        width = int(self.config["width"])
        height = int(self.config["height"])
        mirror = bool(self.config["mirror"])
        frame_bytes = width * height * 3
        proc = None
        count = 0
        last_count = 0
        last_t = time.perf_counter()

        while not self.state.stop_event.is_set():
            if proc is None or proc.poll() is not None:
                if proc is not None:
                    proc.kill()
                try:
                    proc = self.open_ffmpeg()
                    self.state.camera_ready = True
                    print(
                        f"ffmpeg camera opened {self.config['camera_device']} "
                        f"{width}x{height}@{self.config['camera_fps']}",
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

            frame = np.frombuffer(frame_data, dtype=np.uint8).reshape(height, width, 3)
            if mirror:
                frame = cv2.flip(frame, 1)

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

        width = int(self.config["width"])
        height = int(self.config["height"])
        mirror = bool(self.config["mirror"])
        cap = None
        count = 0
        last_count = 0
        last_t = time.perf_counter()

        while not self.state.stop_event.is_set():
            if cap is None or not cap.isOpened():
                if cap is not None:
                    cap.release()
                cap = self.open_camera()
                if not cap.isOpened():
                    self.state.camera_ready = False
                    self.state.set_error(f"camera unavailable: {self.config['camera_device']}")
                    time.sleep(1.0)
                    continue
                self.state.camera_ready = True
                print(
                    f"camera opened {self.config['camera_device']} "
                    f"{width}x{height}@{self.config['camera_fps']}",
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

            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            if mirror:
                frame = cv2.flip(frame, 1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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
        width = int(self.config["width"])
        height = int(self.config["height"])
        warmup = f"1x{height}x{width}x3"
        print(f"loading diffusion model for {width}x{height}", flush=True)

        try:
            processor = DiffusionProcessor(warmup=warmup, local_files_only=True, gpu_id=0)
        except Exception as exc:
            self.state.set_error(f"model load failed: {exc}")
            return

        self.state.model_ready = True
        last_raw_id = -1
        count = 0
        last_count = 0
        last_t = time.perf_counter()

        while not self.state.stop_event.is_set():
            with self.state.frame_lock:
                raw_id = self.state.raw_frame_id
                raw = None if self.state.raw_frame is None else self.state.raw_frame.copy()

            if raw is None or raw_id == last_raw_id:
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
            "/screenshot",
            "/steps",
            "/transformirror/prompt",
            "/transformirror/seed",
            "/transformirror/strength",
            "/transformirror/blend",
            "/transformirror/passthrough",
            "/transformirror/screenshot",
            "/transformirror/steps",
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
        allowed = {k: payload[k] for k in ("prompt", "seed", "strength", "blend", "steps") if k in payload}
        state.update_controls(**allowed)
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
        )
        if not needs_update or raw is None:
            return None

        self.last_raw_id = raw_id
        self.last_processed_id = processed_id
        self.last_blend = blend

        if processed is None:
            return raw
        if blend <= 0:
            return raw
        if blend >= 1:
            return processed
        return cv2.addWeighted(raw, 1.0 - blend, processed, blend, 0)

    def update_texture(self, _dt):
        frame = self.choose_frame()
        if frame is None:
            return
        image = pyglet.image.ImageData(
            self.width,
            self.height,
            "RGB",
            frame.tobytes(),
            pitch=self.width * 3,
        )
        if self.texture is not None:
            self.texture.delete()
        self.texture = image.get_texture().get_transform(flip_y=True)

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

    state = RuntimeState(config)

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

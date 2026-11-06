"""Odaiba live installation: two cameras -> SDXL Turbo img2img -> LED wall.

Input: one DeckLink HDMI signal (3840x2160) with the cameras stacked:
  camera 1 (outward-facing) at (0, 0, 3840, 1080)
  camera 2 (crowd-facing)   at (0, 1080, 3840, 1080)

From each camera we take the centered 8:1 crop (3840x480). Only one camera is
shown at a time, with a slow crossfade between them (camera 2 is mirrored):
  camera 1 for camera_hold_seconds
  crossfade to camera 2 over camera_fade_seconds
  camera 2 for camera_hold_seconds
  crossfade to camera 1 over camera_fade_seconds

The blended 3840x480 frame goes through img2img. The result is the full 6144x768 LED wall (scaled 1.6x), split across the
3840x2160 output that the LED processor samples:
  image x 0..1920    -> output (0, 0)   to (3072, 768)    left half of the wall
  image x 1920..3840 -> output (0, 768) to (3072, 1536)   right half of the wall

Until the model is ready (about a minute after start) the raw composite is shown.
Prompts come from prompts-odaiba.txt (one per line), shown in order for
prompt_cycle_seconds each. Settings come from odaiba_config.json. Both files
are reloaded when they change.
There is no audio and no web interface.

Keys: Esc quits, R toggles raw/processed, G toggles guides, S toggles status.

Sending SIGUSR1 saves input/output JPEG pairs to stills/ for camera 1, camera 2
and the 50-50 blend, each as the crossfade reaches it (up to one full cycle).
"""

import argparse
import json
import os
import signal
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pyglet
from pyglet import shapes
from pyglet.gl import GL_LINEAR, GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_TEXTURE_MIN_FILTER, glBindTexture, glTexParameteri

from led_preview import Capture

DEFAULT_CONFIG = {
    "prompts_file": "prompts-odaiba.txt",
    "prompt_cycle_seconds": 15,
    "camera_hold_seconds": 60,
    "camera_fade_seconds": 60,
    # Used only if the prompts file is missing or empty.
    "prompt": "a vivid oil painting of a city at night, glowing lights, rich color",
    "strength": 0.7,
    "steps": 2,
    "seed": 0,
    "mirror_crowd_camera": True,
    "show_status": True,
    "screen": None,
}

WIDTH, HEIGHT = 3840, 480  # centered crop of each camera, and diffusion input/output
CAMERA_H = 1080
PANEL_W, PANEL_H = 3072, 768  # each half of the 6144x768 wall on the output
OUTPUT_W, OUTPUT_H = 3840, 2160
STILLS_DIR = Path("stills")
STILL_TARGETS = {"camera1": 0.0, "camera2": 1.0, "blend50": 0.5}
STILL_TOLERANCE = 0.004  # wider than the blend change between two diffusion frames


class Config:
    def __init__(self, path):
        self.path = Path(path)
        self.mtime = None
        self.values = dict(DEFAULT_CONFIG)
        self.lock = threading.Lock()
        self.prompts = []
        self.prompts_path = None
        self.prompts_mtime = None
        self.prompt_index = 0
        self.prompt_started = time.monotonic()
        self.started = time.monotonic()
        self.reload()

    def reload(self):
        self.reload_config()
        self.reload_prompts()

    def reload_prompts(self):
        with self.lock:
            path = self.path.parent / str(self.values["prompts_file"])
        try:
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            if self.prompts:
                print(f"prompts file missing: {path}", flush=True)
            with self.lock:
                self.prompts = []
                self.prompts_path = path
                self.prompts_mtime = None
            return
        if path == self.prompts_path and mtime == self.prompts_mtime:
            return
        prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        with self.lock:
            self.prompts = prompts
            self.prompts_path = path
            self.prompts_mtime = mtime
            self.prompt_index %= max(1, len(prompts))
        print(f"loaded {len(prompts)} prompts from {path.name}", flush=True)

    def current_prompt(self):
        """Returns (prompt, index, count), advancing every prompt_cycle_seconds."""
        with self.lock:
            if not self.prompts:
                return str(self.values["prompt"]), 0, 0
            now = time.monotonic()
            cycle = max(1.0, float(self.values["prompt_cycle_seconds"]))
            if now - self.prompt_started >= cycle:
                self.prompt_index = (self.prompt_index + 1) % len(self.prompts)
                self.prompt_started = now
                print(f"prompt {self.prompt_index + 1}/{len(self.prompts)}: {self.prompts[self.prompt_index]}", flush=True)
            return self.prompts[self.prompt_index], self.prompt_index, len(self.prompts)

    def reload_config(self):
        try:
            mtime = self.path.stat().st_mtime
        except FileNotFoundError:
            return
        if mtime == self.mtime:
            return
        self.mtime = mtime
        try:
            data = json.loads(self.path.read_text())
        except Exception as exc:
            print(f"config not reloaded ({exc})", flush=True)
            return
        with self.lock:
            self.values = {**DEFAULT_CONFIG, **data}
        print(f"config: {json.dumps(self.values, ensure_ascii=False)}", flush=True)

    def get(self):
        with self.lock:
            return dict(self.values)

    def camera_blend(self):
        """Where the crossfade is now: 0 is camera 1, 1 is camera 2."""
        cfg = self.get()
        hold = max(0.0, float(cfg["camera_hold_seconds"]))
        fade = max(0.0, float(cfg["camera_fade_seconds"]))
        t = (time.monotonic() - self.started) % max(1.0, 2 * (hold + fade))
        if t < hold:
            return 0.0
        t -= hold
        if t < fade:
            return t / fade
        t -= fade
        if t < hold:
            return 1.0
        return 1.0 - (t - hold) / fade


def blend_cameras(rows, blend, mirror_crowd):
    """rows starts at camera 1's crop and ends with camera 2's crop."""
    cam1 = rows[:HEIGHT]
    cam2 = rows[CAMERA_H:]
    cam2 = np.ascontiguousarray(cam2[:, ::-1] if mirror_crowd else cam2)
    if blend <= 0.0:
        return cam1
    if blend >= 1.0:
        return cam2
    return cv2.addWeighted(cam1, 1.0 - blend, cam2, blend, 0.0)


def save_jpeg(path, frame):
    cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])


class Inference(threading.Thread):
    def __init__(self, capture, config, stop):
        super().__init__(daemon=True)
        self.capture = capture
        self.config = config
        self.stop = stop
        self.lock = threading.Lock()
        self.output = None
        self.output_id = 0
        self.ready = False
        self.status = "loading model"
        self.ms = 0.0
        self.fps = 0.0
        self.stills_pending = {}

    def request_stills(self):
        self.stills_pending = dict(STILL_TARGETS)
        print("stills requested", flush=True)

    def save_stills(self, blend, frame, output, prompt):
        for name, target in list(self.stills_pending.items()):
            if abs(blend - target) <= STILL_TOLERANCE:
                STILLS_DIR.mkdir(exist_ok=True)
                save_jpeg(STILLS_DIR / f"{name}-input.jpg", frame)
                save_jpeg(STILLS_DIR / f"{name}-output.jpg", output)
                del self.stills_pending[name]
                print(f"saved {name} stills at blend {blend:.3f}: {prompt}", flush=True)

    def run(self):
        from diffusion_processor import DiffusionProcessor

        while not self.stop.is_set():
            try:
                processor = DiffusionProcessor(warmup=f"1x{HEIGHT}x{WIDTH}x3", local_files_only=True, gpu_id=0)
                break
            except Exception as exc:
                self.status = f"model load failed: {exc}"
                print(self.status, flush=True)
                time.sleep(5.0)
        else:
            return

        self.ready = True
        self.status = "running"
        last_id = -1
        count = 0
        last_t = time.perf_counter()
        while not self.stop.is_set():
            rows, frame_id = self.capture.latest()
            if rows is None or frame_id == last_id:
                time.sleep(0.002)
                continue
            last_id = frame_id
            cfg = self.config.get()
            blend = self.config.camera_blend()
            frame = blend_cameras(rows, blend, cfg["mirror_crowd_camera"])
            image = frame.astype(np.float32) / 255.0
            prompt = self.config.current_prompt()[0]
            started = time.perf_counter()
            try:
                result = processor.run(
                    images=[image],
                    prompt=prompt,
                    num_inference_steps=int(cfg["steps"]),
                    strength=float(cfg["strength"]),
                    seed=int(cfg["seed"]),
                )
            except Exception as exc:
                self.status = f"inference failed: {exc}"
                print(self.status, flush=True)
                time.sleep(0.5)
                continue
            out = np.clip(result[0] * 255.0, 0, 255).astype(np.uint8)
            with self.lock:
                self.output = out
                self.output_id += 1
            self.save_stills(blend, frame, out, prompt)
            self.ms = (time.perf_counter() - started) * 1000.0
            self.status = "running"
            count += 1
            now = time.perf_counter()
            if now - last_t >= 10.0:
                self.fps = count / (now - last_t)
                print(f"diffusion {self.ms:.1f} ms, {self.fps:.2f} fps, capture {self.capture.fps:.1f} fps", flush=True)
                count = 0
                last_t = now

    def latest(self):
        with self.lock:
            return self.output, self.output_id


class Display:
    def __init__(self, capture, inference, config, screen_index):
        self.capture = capture
        self.inference = inference
        self.config = config
        self.show_raw = False
        self.show_guides = False
        self.uploaded = None  # (source, id) currently in the texture
        self.draw_count = 0
        self.draw_fps = 0.0
        self.last_draw_t = time.perf_counter()

        screens = pyglet.display.get_display().get_screens()
        if screen_index is not None:
            screen = screens[screen_index]
        else:
            matches = [s for s in screens if (s.width, s.height) == (OUTPUT_W, OUTPUT_H)]
            screen = matches[0] if matches else screens[-1]
        print(f"screens: {[(s.x, s.y, s.width, s.height) for s in screens]}; using {screen.x},{screen.y}", flush=True)
        self.window = pyglet.window.Window(fullscreen=True, screen=screen, caption="Odaiba", vsync=True)
        self.window.set_mouse_visible(False)
        self.window.push_handlers(on_draw=self.on_draw, on_key_press=self.on_key_press)

        self.texture = pyglet.image.Texture.create(WIDTH, HEIGHT)
        glBindTexture(GL_TEXTURE_2D, self.texture.id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        self.halves = [
            self.texture.get_region(0, 0, WIDTH // 2, HEIGHT),
            self.texture.get_region(WIDTH // 2, 0, WIDTH // 2, HEIGHT),
        ]
        self.label = pyglet.text.Label(
            "", font_size=28, x=40, y=40, anchor_y="bottom", color=(200, 200, 200, 255), multiline=True, width=3600
        )

    def upload(self, frame):
        image = pyglet.image.ImageData(WIDTH, HEIGHT, "RGB", frame.tobytes(), pitch=-WIDTH * 3)
        self.texture.blit_into(image, 0, 0, 0)

    def update_texture(self):
        cfg = self.config.get()
        if not self.show_raw and self.inference.ready:
            output, output_id = self.inference.latest()
            if output is not None:
                if self.uploaded != ("processed", output_id):
                    self.upload(output)
                    self.uploaded = ("processed", output_id)
                return
        rows, frame_id = self.capture.latest()
        if rows is not None and self.uploaded != ("raw", frame_id):
            self.upload(blend_cameras(rows, self.config.camera_blend(), cfg["mirror_crowd_camera"]))
            self.uploaded = ("raw", frame_id)

    def placements(self):
        sx = self.window.width / OUTPUT_W
        sy = self.window.height / OUTPUT_H
        return [(0, (OUTPUT_H - (i + 1) * PANEL_H) * sy, PANEL_W * sx, PANEL_H * sy) for i in range(2)]

    def on_draw(self):
        self.window.clear()
        self.update_texture()
        placements = self.placements()
        if self.uploaded is not None:
            for region, (x, y, w, h) in zip(self.halves, placements):
                region.blit(x, y, width=w, height=h)
        if self.show_guides:
            for (x, y, w, h), color in zip(placements, [(255, 60, 60), (60, 160, 255)]):
                shapes.Box(x, y, w, h, thickness=4, color=color).draw()
        if self.config.get()["show_status"]:
            source = self.uploaded[0] if self.uploaded else "none"
            self.label.text = (
                f"capture: {self.capture.status} {self.capture.fps:.1f} fps   "
                f"model: {self.inference.status} {self.inference.ms:.0f} ms   "
                f"showing: {source}   camera 2 blend {self.config.camera_blend():.0%}   "
                f"display {self.draw_fps:.0f} fps\n"
                f"{self.prompt_label()}   [R] raw  [G] guides  [Esc] quit"
            )
            self.label.draw()

        self.draw_count += 1
        now = time.perf_counter()
        if now - self.last_draw_t >= 1.0:
            self.draw_fps = self.draw_count / (now - self.last_draw_t)
            self.draw_count = 0
            self.last_draw_t = now

    def prompt_label(self):
        prompt, index, count = self.config.current_prompt()
        return f"prompt {index + 1}/{count}: {prompt}" if count else f"prompt: {prompt}"

    def on_key_press(self, symbol, _modifiers):
        if symbol == pyglet.window.key.ESCAPE:
            pyglet.app.exit()
        elif symbol == pyglet.window.key.R:
            self.show_raw = not self.show_raw
        elif symbol == pyglet.window.key.G:
            self.show_guides = not self.show_guides


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(Path(__file__).with_name("odaiba_config.json")))
    args = parser.parse_args()
    os.chdir(Path(__file__).resolve().parent)

    config = Config(args.config)
    capture_args = SimpleNamespace(
        device=0,
        threads=4,
        signal_width=3840,
        signal_height=2160,
        # Capture crops a centered block of rows: 300..1860 spans both camera crops.
        camera_height=2 * CAMERA_H,
        crop_height=CAMERA_H + HEIGHT,
    )
    capture = Capture(capture_args)
    stop = threading.Event()
    inference = Inference(capture, config, stop)
    capture.start()
    inference.start()

    display = Display(capture, inference, config, config.get()["screen"])
    pyglet.clock.schedule_interval(lambda _dt: config.reload(), 1.0)
    signal.signal(signal.SIGTERM, lambda *_: pyglet.app.exit())
    signal.signal(signal.SIGUSR1, lambda *_: inference.request_stills())
    try:
        pyglet.app.run(1 / 60)
    finally:
        stop.set()
        capture.stop.set()


if __name__ == "__main__":
    main()

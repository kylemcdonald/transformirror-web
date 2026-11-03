"""Realtime LED wall layout preview for the Odaiba installation.

Input: one DeckLink HDMI signal (3840x2160). The top half holds two cameras
side by side: camera 1 at (0, 0, 1920, 1080), camera 2 at (1920, 0, 1920, 1080).

From each camera we take the centered 4:1 crop (1920x480), scale it to
3072x768 and draw it on the 3840x2160 output, which the LED processor samples:

    camera 1 -> (0, 0)   to (3072, 768)   left half of the 6144x768 wall
    camera 2 -> (0, 768) to (3072, 1536)  right half of the wall

Everything below y=1536 is not sampled by the wall, so status text goes there.

Keys: Esc quits, G toggles guide outlines, S toggles the status text.
"""

import argparse
import ctypes
import signal
import subprocess
import threading
import time

import numpy as np
import pyglet
from pyglet import shapes
from pyglet.gl import GL_LINEAR, GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_TEXTURE_MIN_FILTER, glBindTexture, glTexParameteri


def die_with_parent():
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
    except Exception:
        pass


class Capture(threading.Thread):
    """Reads the centered 4:1 band of both cameras as one RGB frame."""

    def __init__(self, args):
        super().__init__(daemon=True)
        self.args = args
        self.band_y = (args.camera_height - args.crop_height) // 2  # 300
        self.width = args.signal_width  # 3840: both cameras side by side
        self.height = args.crop_height  # 480
        self.lock = threading.Lock()
        self.frame = None
        self.frame_id = 0
        self.fps = 0.0
        self.status = "starting"
        self.stop = threading.Event()

    def command(self):
        a = self.args
        return [
            "gst-launch-1.0",
            "-q",
            "decklinkvideosrc",
            f"device-number={a.device}",
            "connection=hdmi",
            "mode=auto",
            "drop-no-signal-frames=true",
            "!",
            "queue",
            "max-size-buffers=2",
            "max-size-bytes=0",
            "max-size-time=0",
            "leaky=downstream",
            "!",
            # Normalizes to the expected signal size even if the mode changes.
            "videoconvertscale",
            f"n-threads={a.threads}",
            "!",
            f"video/x-raw,format=RGB,width={a.signal_width},height={a.signal_height}",
            "!",
            "videocrop",
            f"top={self.band_y}",
            f"bottom={a.signal_height - self.band_y - a.crop_height}",
            "!",
            "fdsink",
            "fd=1",
            "sync=false",
        ]

    def run(self):
        frame_bytes = self.width * self.height * 3
        count = 0
        last_t = time.perf_counter()
        while not self.stop.is_set():
            proc = subprocess.Popen(
                self.command(), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, preexec_fn=die_with_parent
            )
            self.status = "waiting for signal"
            print("decklink capture started", flush=True)
            while not self.stop.is_set():
                data = proc.stdout.read(frame_bytes)
                if len(data) != frame_bytes:
                    break
                frame = np.frombuffer(data, dtype=np.uint8).reshape(self.height, self.width, 3)
                with self.lock:
                    self.frame = frame
                    self.frame_id += 1
                self.status = "live"
                count += 1
                now = time.perf_counter()
                if now - last_t >= 1.0:
                    self.fps = count / (now - last_t)
                    count = 0
                    last_t = now
            proc.kill()
            proc.wait()
            if not self.stop.is_set():
                self.status = "no signal or input busy; retrying"
                self.fps = 0.0
                print(self.status, flush=True)
                time.sleep(2.0)

    def latest(self):
        with self.lock:
            return self.frame, self.frame_id


class PreviewApp:
    def __init__(self, args, capture):
        self.args = args
        self.capture = capture
        self.last_frame_id = -1
        self.show_guides = args.guides
        self.show_status = True
        self.draw_count = 0
        self.draw_fps = 0.0
        self.last_draw_t = time.perf_counter()

        display = pyglet.display.get_display()
        screens = display.get_screens()
        screen = self.pick_screen(screens)
        print(f"screens: {[(s.x, s.y, s.width, s.height) for s in screens]}", flush=True)
        print(f"using screen at {screen.x},{screen.y} {screen.width}x{screen.height}", flush=True)
        if args.windowed:
            self.window = pyglet.window.Window(1920, 1080, caption="LED preview", resizable=True, vsync=True)
        else:
            self.window = pyglet.window.Window(fullscreen=True, screen=screen, caption="LED preview", vsync=True)
        self.window.set_mouse_visible(False)
        self.window.push_handlers(on_draw=self.on_draw, on_key_press=self.on_key_press)

        # One texture holds both cameras' bands side by side (3840x480).
        self.texture = pyglet.image.Texture.create(capture.width, capture.height)
        glBindTexture(GL_TEXTURE_2D, self.texture.id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        half = capture.width // 2
        self.regions = [
            self.texture.get_region(0, 0, half, capture.height),
            self.texture.get_region(half, 0, capture.width - half, capture.height),
        ]

        self.status_label = pyglet.text.Label(
            "",
            font_size=28,
            x=40,
            y=40,
            anchor_y="bottom",
            color=(200, 200, 200, 255),
            multiline=True,
            width=3600,
        )

    def pick_screen(self, screens):
        if self.args.screen is not None:
            return screens[self.args.screen]
        for screen in screens:
            if (screen.width, screen.height) == (self.args.output_width, self.args.output_height):
                return screen
        return screens[-1]

    def placements(self):
        """(x, y, w, h) in GL coordinates (origin bottom-left) for each camera."""
        a = self.args
        sx = self.window.width / a.output_width
        sy = self.window.height / a.output_height
        out = []
        for index in range(2):
            top = index * a.panel_height  # 0 for camera 1, 768 for camera 2
            out.append(
                (
                    0,
                    (a.output_height - top - a.panel_height) * sy,
                    a.panel_width * sx,
                    a.panel_height * sy,
                )
            )
        return out

    def update_texture(self):
        frame, frame_id = self.capture.latest()
        if frame is None or frame_id == self.last_frame_id:
            return
        self.last_frame_id = frame_id
        # Negative pitch: rows are top-to-bottom in the numpy frame.
        image = pyglet.image.ImageData(
            frame.shape[1], frame.shape[0], "RGB", frame.tobytes(), pitch=-frame.shape[1] * 3
        )
        self.texture.blit_into(image, 0, 0, 0)

    def draw_guides(self, placements):
        colors = [(255, 60, 60), (60, 160, 255)]
        for (x, y, w, h), color in zip(placements, colors):
            shapes.Box(x, y, w, h, thickness=4, color=color).draw()

    def on_draw(self):
        self.window.clear()
        self.update_texture()
        placements = self.placements()
        if self.last_frame_id >= 0:
            for region, (x, y, w, h) in zip(self.regions, placements):
                region.blit(x, y, width=w, height=h)
        if self.show_guides:
            self.draw_guides(placements)
        if self.show_status:
            a = self.args
            self.status_label.text = (
                f"capture: {self.capture.status}   {self.capture.fps:.1f} fps   display {self.draw_fps:.1f} fps\n"
                f"camera 1 center {a.camera_width}x{a.crop_height} -> (0,0)-({a.panel_width},{a.panel_height})   "
                f"camera 2 -> (0,{a.panel_height})-({a.panel_width},{2 * a.panel_height})   "
                f"window {self.window.width}x{self.window.height}   [G] guides  [S] status  [Esc] quit"
            )
            self.status_label.draw()

        self.draw_count += 1
        now = time.perf_counter()
        if now - self.last_draw_t >= 1.0:
            self.draw_fps = self.draw_count / (now - self.last_draw_t)
            self.draw_count = 0
            self.last_draw_t = now

    def on_key_press(self, symbol, _modifiers):
        if symbol == pyglet.window.key.ESCAPE:
            pyglet.app.exit()
        elif symbol == pyglet.window.key.G:
            self.show_guides = not self.show_guides
        elif symbol == pyglet.window.key.S:
            self.show_status = not self.show_status


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--signal-width", type=int, default=3840)
    parser.add_argument("--signal-height", type=int, default=2160)
    parser.add_argument("--camera-width", type=int, default=1920)
    parser.add_argument("--camera-height", type=int, default=1080)
    parser.add_argument("--crop-height", type=int, default=480, help="4:1 crop of a 1920-wide camera")
    parser.add_argument("--output-width", type=int, default=3840)
    parser.add_argument("--output-height", type=int, default=2160)
    parser.add_argument("--panel-width", type=int, default=3072)
    parser.add_argument("--panel-height", type=int, default=768)
    parser.add_argument("--screen", type=int, help="pyglet screen index (default: the 3840x2160 one)")
    parser.add_argument("--windowed", action="store_true")
    parser.add_argument("--guides", action="store_true", help="start with guide outlines visible")
    args = parser.parse_args()

    capture = Capture(args)
    capture.start()
    app = PreviewApp(args, capture)
    signal.signal(signal.SIGTERM, lambda *_: pyglet.app.exit())
    try:
        pyglet.app.run(1 / 60)
    finally:
        capture.stop.set()


if __name__ == "__main__":
    main()

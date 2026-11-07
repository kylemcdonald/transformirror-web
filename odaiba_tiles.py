"""Renders the Odaiba wall at full 7680x960 from 960x960 tiles.

Stylizer, Wall and MotionPicker are what odaiba_live.py runs. Run as a script, this
is the offline experiment they came from.

Takes the centered 3840x480 crop of each stacked camera in a recording, scales it
2x to 7680x960, runs SDXL Turbo img2img on 960x960 tiles and writes one video per
camera per variant:

  tiled      All 8 tiles every frame, hard seams. Every tile gets the same seed and
             so the same noise.
  feathered  All 8 tiles every frame. Each tile is processed with a margin of its
             neighbors, the windows are matched to each other in the overlaps, and
             the overlaps are crossfaded. Noise comes from one wall-sized field, so
             overlapping windows see the same noise.
  motion     Feathered for the first frame. After that, only the tile that changed
             most since its own last update is processed, and it is matched to the
             frozen tiles on either side of it.

Windows always see the camera, never the stylized wall. Feeding the wall back in as
context looked like a way to join tiles up, but it is a feedback loop: within 90
frames figures grew and new objects appeared.

Motion is the change since a tile was last updated, not since the previous frame.
Water shimmer and lamp flicker are large from frame to frame, but they do not build
up, so they do not hold the update on one tile while real changes wait elsewhere.
Everything is causal, so a live version could make the same choices.
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderTiny, AutoPipelineForImage2Image

CROP_W, CROP_H = 3840, 480
CAMERA_H = 1080
CAMERAS = {"top": (CAMERA_H - CROP_H) // 2, "bottom": CAMERA_H + (CAMERA_H - CROP_H) // 2}
WALL_W, WALL_H = 2 * CROP_W, 2 * CROP_H
TILE = 960
TILES = WALL_W // TILE
LATENT_SCALE = 8
MOTION_W, MOTION_H = WALL_W // 8, WALL_H // 8
VARIANTS = ("tiled", "feathered", "motion")
DEFAULT_OUT = "data/odaiba-tiles"


class Stylizer:
    """SDXL Turbo img2img, split up so tiles can share latents and noise.

    With 2 steps at strength 0.7 the img2img pipeline runs a single UNet step from
    the second timestep, so that one step is all this does.
    """

    def __init__(self, prompt, seed, steps=2, strength=0.7, device="cuda"):
        self.device = torch.device(device)
        pipe = AutoPipelineForImage2Image.from_pretrained(
            "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16", local_files_only=True
        )
        pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesdxl", torch_dtype=torch.float16, local_files_only=True)
        pipe.to(self.device)
        pipe.set_progress_bar_config(disable=True)
        self.unet = pipe.unet.to(memory_format=torch.channels_last)
        self.vae = pipe.vae
        self.seed = seed
        self.pipe = pipe

        strength = min(max(1 / steps, strength), 1)
        start = steps - min(int(steps * strength), steps)
        if start != steps - 1:
            raise ValueError("only a single denoising step is supported")
        pipe.scheduler.set_timesteps(steps, device=self.device)
        self.timestep = pipe.scheduler.timesteps[start]
        self.sigma = float(pipe.scheduler.sigmas[start])

        self.set_prompt(prompt)

    def set_prompt(self, prompt):
        with torch.inference_mode():
            self.embeds, _, self.pooled, _ = self.pipe.encode_prompt(
                prompt=prompt, device=self.device, num_images_per_prompt=1, do_classifier_free_guidance=False
            )

    def noise(self, width):
        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        shape = (4, WALL_H // LATENT_SCALE, width // LATENT_SCALE)
        return torch.randn(shape, generator=generator, device=self.device, dtype=torch.float16)

    def encode(self, images):
        """images: Bx3xHxW in 0..1."""
        return self.vae.encode(images * 2 - 1).latents * self.vae.config.scaling_factor

    def decode(self, latents):
        return (self.vae.decode(latents / self.vae.config.scaling_factor).sample / 2 + 0.5).clamp(0, 1)

    def denoise(self, latents, noise):
        batch, _, height, width = latents.shape
        noisy = latents + self.sigma * noise
        size = (height * LATENT_SCALE, width * LATENT_SCALE)
        time_ids = torch.tensor([[*size, 0, 0, *size]], device=self.device, dtype=torch.float16).expand(batch, -1)
        predicted = self.unet(
            noisy / (self.sigma**2 + 1) ** 0.5,
            self.timestep,
            encoder_hidden_states=self.embeds.expand(batch, -1, -1),
            added_cond_kwargs={"text_embeds": self.pooled.expand(batch, -1), "time_ids": time_ids},
            return_dict=False,
        )[0]
        return noisy - self.sigma * predicted


def feather_masks(margin, device):
    """Per-tile weights along the wall. They sum to 1 at every x.

    Each seam gets a smoothstep ramp 2*margin wide, centered on the tile edge.
    """
    x = np.arange(WALL_W) + 0.5
    masks = []
    for tile in range(TILES):
        left, right = tile * TILE, (tile + 1) * TILE
        mask = np.ones(WALL_W, np.float32)
        if margin == 0:
            mask[:] = (x >= left) & (x < right)
        else:
            if tile > 0:
                mask *= np.clip((x - left + margin) / (2 * margin), 0, 1)
            if tile < TILES - 1:
                mask *= np.clip((right + margin - x) / (2 * margin), 0, 1)
            mask = np.where(mask >= 1, 1, mask * mask * (3 - 2 * mask))
        masks.append(mask)
    return torch.from_numpy(np.stack(masks)).to(device)


def smooth_rows(offset, radius=48):
    """offset: 3xH. Keeps the correction low-frequency down the height of the wall."""
    for _ in range(2):
        offset = F.avg_pool1d(offset[None], 2 * radius + 1, stride=1, padding=radius, count_include_pad=False)[0]
    return offset


class Wall:
    """The stylized wall, updated one window at a time."""

    def __init__(self, stylizer, margin, shared_noise):
        self.stylizer = stylizer
        self.margin = margin
        self.window = TILE + 2 * margin
        self.masks = feather_masks(margin, stylizer.device)
        # Each window slices one wall-sized field, or every tile reuses one field.
        self.noise = stylizer.noise(WALL_W if shared_noise else self.window)
        self.shared_noise = shared_noise
        self.windows = torch.zeros(TILES, 3, WALL_H, self.window, device=stylizer.device)
        self.pixels = torch.zeros(3, WALL_H, WALL_W, device=stylizer.device)

    def window_x(self, tile):
        return min(max(tile * TILE - self.margin, 0), WALL_W - self.window)

    def seam_offsets(self, tiles, xs, pixels):
        """Returns the left and right edge offsets of each window, each Bx3xH.

        Every window lands on its own overall exposure and color, and a crossfade
        only softens that step. So at each seam, find the difference in the overlap
        and remove it: windows updated together meet halfway, and a window next to
        the frozen wall moves all the way to it. The median ignores whatever is
        moving through the overlap.
        """
        left = torch.zeros(len(tiles), 3, WALL_H, device=pixels.device)
        right = torch.zeros_like(left)
        index = {tile: i for i, tile in enumerate(tiles)}

        def band(tile, seam):
            start = seam * TILE - self.margin - self.window_x(tile)
            window = pixels[index[tile]] if tile in index else self.windows[tile]
            return window[:, :, start : start + 2 * self.margin]

        for seam in range(1, TILES):
            before, after = seam - 1, seam
            if before not in index and after not in index:
                continue
            offset = smooth_rows((band(after, seam) - band(before, seam)).median(-1).values)
            share = 0.5 if before in index and after in index else 1.0
            if before in index:
                right[index[before]] = share * offset
            if after in index:
                left[index[after]] = -share * offset
        for i, tile in enumerate(tiles):
            if tile == 0:
                left[i] = right[i]
            if tile == TILES - 1:
                right[i] = left[i]
        return left, right

    def apply_offsets(self, tiles, xs, pixels):
        left, right = self.seam_offsets(tiles, xs, pixels)
        for i, (tile, x) in enumerate(zip(tiles, xs)):
            columns = torch.arange(x, x + self.window, device=pixels.device) + 0.5
            ramp = ((columns - tile * TILE) / TILE).clamp(0, 1)
            pixels[i] += left[i][:, :, None] * (1 - ramp) + right[i][:, :, None] * ramp
        return pixels.clamp_(0, 1)

    def update(self, frame, tiles):
        """frame: 3xHxW on the GPU in 0..1. Blends the given tiles into the wall."""
        xs = [self.window_x(tile) for tile in tiles]
        ls = LATENT_SCALE
        lw = self.window // ls
        latents = self.stylizer.encode(torch.stack([frame[:, :, x : x + self.window] for x in xs]))
        if self.shared_noise:
            noise = torch.stack([self.noise[:, :, x // ls : x // ls + lw] for x in xs])
        else:
            noise = self.noise.expand(len(tiles), -1, -1, -1)
        pixels = self.stylizer.decode(self.stylizer.denoise(latents, noise)).float()
        if self.margin:
            pixels = self.apply_offsets(tiles, xs, pixels)

        # Always rebuild the wall from the latest window of every tile. Crossfading
        # onto the wall instead would compound with each update, and the ramp would
        # erode into a hard edge that eats into the frozen neighbor.
        self.windows[tiles] = pixels
        self.pixels.zero_()
        for tile in range(TILES):
            x = self.window_x(tile)
            self.pixels[:, :, x : x + self.window] += self.masks[tile, x : x + self.window] * self.windows[tile]


class MotionPicker:
    """Picks the tile that has changed most since its own last update."""

    def __init__(self, threshold):
        self.threshold = threshold
        self.reference = None

    def reset(self, frame):
        """Every tile has just been updated from this frame."""
        self.reference = self.small(frame)

    def small(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (MOTION_W, MOTION_H), interpolation=cv2.INTER_AREA).astype(np.float32)
        return cv2.GaussianBlur(gray, (0, 0), 1.5).reshape(MOTION_H, TILES, MOTION_W // TILES)

    def pick(self, frame):
        """frame: the 3840x480 crop. Returns (tile or None, score per tile)."""
        small = self.small(frame)
        if self.reference is None:
            self.reference = small
            return None, np.zeros(TILES)
        scores = np.maximum(np.abs(small - self.reference) - self.threshold, 0).sum((0, 2))
        tile = int(scores.argmax())
        if scores[tile] <= 0:
            return None, scores
        self.reference[:, tile] = small[:, tile]
        return tile, scores


def read_frames(path, y, limit):
    command = ["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"crop={CROP_W}:{CROP_H}:0:{y}", "-pix_fmt", "rgb24", "-f", "rawvideo"]
    if limit:
        command += ["-frames:v", str(limit)]
    process = subprocess.Popen([*command, "-"], stdout=subprocess.PIPE)
    size = CROP_W * CROP_H * 3
    try:
        while len(data := process.stdout.read(size)) == size:
            yield np.frombuffer(data, np.uint8).reshape(CROP_H, CROP_W, 3)
    finally:
        process.stdout.close()
        process.wait()


def open_writer(path, fps):
    size = f"{WALL_W}x{WALL_H}"
    return subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", size, "-r", str(fps), "-i", "-"]
        + ["-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
        stdin=subprocess.PIPE,
    )


def to_gpu(frame, device):
    """Scales an RGB uint8 frame 2x with bicubic. Returns 3xHxW half in 0..1."""
    pixels = torch.from_numpy(np.ascontiguousarray(frame)).to(device).permute(2, 0, 1)[None].float() / 255
    return F.interpolate(pixels, scale_factor=2, mode="bicubic")[0].clamp_(0, 1).half()


def render(stylizer, args, camera, variant, fps):
    margin = 0 if variant == "tiled" else args.margin
    wall = Wall(stylizer, margin, shared_noise=variant != "tiled")
    picker = MotionPicker(args.threshold)
    path = args.out_dir / f"{camera}-{variant}.mp4"
    writer = open_writer(path, fps)
    all_tiles = list(range(TILES))
    picks = []
    tile_ms = []
    started = time.perf_counter()
    count = 0
    for frame in read_frames(args.input, CAMERAS[camera], args.frames):
        tile, _ = picker.pick(frame) if variant == "motion" else (None, None)
        gpu_frame = to_gpu(frame, stylizer.device)
        torch.cuda.synchronize()
        step_started = time.perf_counter()
        if variant != "motion" or count == 0:
            wall.update(gpu_frame, all_tiles)
        elif tile is not None:
            wall.update(gpu_frame, [tile])
        torch.cuda.synchronize()
        if variant == "motion" and count > 0:
            picks.append(tile)
            tile_ms.append((time.perf_counter() - step_started) * 1000)
        pixels = (wall.pixels * 255).round_().to(torch.uint8).permute(1, 2, 0).contiguous().cpu().numpy()
        writer.stdin.write(pixels.tobytes())
        count += 1
    writer.stdin.close()
    writer.wait()
    print(f"{path}: {count} frames in {time.perf_counter() - started:.0f} s", flush=True)
    if variant == "motion":
        updates = np.bincount([p for p in picks if p is not None], minlength=TILES).tolist()
        summary = {"updates_per_tile": updates, "median_update_ms": float(np.median(tile_ms[10:] or tile_ms)), "picks": picks}
        path.with_suffix(".json").write_text(json.dumps(summary))
        print(f"  updates per tile {updates}, median update {summary['median_update_ms']:.0f} ms", flush=True)


def write_preview(args, fps):
    """One 3840-wide comparison: for each camera, source then each variant."""
    inputs = []
    filters = []
    for camera in CAMERAS:
        y = CAMERAS[camera]
        filters.append(f"[0:v]crop={CROP_W}:{CROP_H}:0:{y}[{camera}0]")
        for variant in VARIANTS:
            path = args.out_dir / f"{camera}-{variant}.mp4"
            inputs += ["-i", str(path)]
            filters.append(f"[{len(inputs) // 2}:v]scale={CROP_W}:{CROP_H}:flags=area[{camera}{variant}]")
    rows = "".join(f"[{camera}{name}]" for camera in CAMERAS for name in ("0", *VARIANTS))
    filters.append(f"{rows}vstack=inputs={len(CAMERAS) * (len(VARIANTS) + 1)}[out]")
    path = args.out_dir / "preview.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(args.input), *inputs, "-filter_complex", ";".join(filters), "-map", "[out]"]
        + ["-r", str(fps), "-shortest", "-c:v", "hevc_nvenc", "-cq", "22", "-tag:v", "hvc1", "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )
    print(path, flush=True)


def print_link(path):
    url = os.environ.get("AGENT_FILES_URL")
    if url:
        print(f"{url}/#{quote(str(path.resolve()))}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT))
    parser.add_argument("--prompt", default=None, help="defaults to the first line of prompts-odaiba.txt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--margin", type=int, default=64, help="pixels of neighbor on each side of a tile, a multiple of 8")
    parser.add_argument("--threshold", type=float, default=3.0, help="luma change ignored by the motion score")
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--cameras", nargs="+", choices=list(CAMERAS), default=list(CAMERAS))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    args = parser.parse_args()
    if args.margin % LATENT_SCALE:
        parser.error("--margin must be a multiple of 8")

    prompt = args.prompt or Path("prompts-odaiba.txt").read_text(encoding="utf-8").splitlines()[0]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "prompt.txt").write_text(f"{prompt}\nseed {args.seed}\n", encoding="utf-8")
    probe = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", str(args.input)],
        text=True,
    )
    fps = probe.strip()
    stylizer = Stylizer(prompt, args.seed)
    with torch.inference_mode():
        for camera in args.cameras:
            for variant in args.variants:
                render(stylizer, args, camera, variant, fps)
    if set(args.cameras) == set(CAMERAS) and set(args.variants) == set(VARIANTS):
        write_preview(args, fps)
        print_link(args.out_dir / "preview.mp4")
    print_link(args.out_dir)


if __name__ == "__main__":
    main()

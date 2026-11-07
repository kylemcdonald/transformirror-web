"""Renders every Odaiba prompt on the center of both cameras, as a page for review.

Takes a 480x480 crop from the center of each camera in the first frame of a
recording, scales it to 960x960 and runs SDXL Turbo img2img on it with each line of
prompts-odaiba.txt. Writes index.html and its images to one folder. The page is laid
out to be printed to PDF.
"""

import argparse
import html
from pathlib import Path

import cv2
import torch

from odaiba_tiles import CAMERAS, CROP_H, CROP_W, Stylizer, print_link, read_frames, to_gpu

SIZE = CROP_H
DEFAULT_OUT = "data/odaiba-prompt-sheet"
JPEG_QUALITY = 92
STYLE = """
@page { margin: 10mm; }
body { margin: 16px; background: #fff; color: #000; font: 11pt/1.35 system-ui, sans-serif; }
main { max-width: 1400px; margin: auto; }
div { display: grid; grid-template-columns: 2fr 2fr 1fr; gap: 8px; margin-bottom: 8px; break-inside: avoid; }
img { display: block; width: 100%; }
p { margin: 0; }
@media print { body { margin: 0; font-size: 9pt; } main { max-width: none; } }
"""


def save(path, pixels):
    """pixels: 3xHxW on the GPU in 0..1."""
    image = (pixels.float() * 255).round().to(torch.uint8).permute(1, 2, 0).contiguous().cpu().numpy()
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])


def row(names, text=""):
    images = "".join(f'<img src="{name}" alt="">' for name in names)
    return f"<div>{images}<p>{html.escape(text)}</p></div>"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT))
    parser.add_argument("--prompts", type=Path, default=Path("prompts-odaiba.txt"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    prompts = [line.strip() for line in args.prompts.read_text(encoding="utf-8").splitlines() if line.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stylizer = Stylizer(prompts[0], args.seed)
    x = (CROP_W - SIZE) // 2
    crops = [next(read_frames(args.input, y, 1))[:, x : x + SIZE] for y in CAMERAS.values()]
    frames = torch.stack([to_gpu(crop, stylizer.device) for crop in crops])
    noise = stylizer.noise(2 * SIZE).expand(len(crops), -1, -1, -1)

    for camera, frame in zip(CAMERAS, frames):
        save(args.out_dir / f"input-{camera}.jpg", frame)
    rows = [row([f"input-{camera}.jpg" for camera in CAMERAS])]
    with torch.inference_mode():
        latents = stylizer.encode(frames)
        for number, prompt in enumerate(prompts, 1):
            stylizer.set_prompt(prompt)
            names = [f"{number:02}-{camera}.jpg" for camera in CAMERAS]
            for name, pixels in zip(names, stylizer.decode(stylizer.denoise(latents, noise))):
                save(args.out_dir / name, pixels)
            rows.append(row(names, prompt))

    page = f'<!doctype html><html lang="en"><meta charset="utf-8"><title>Odaiba prompts</title><style>{STYLE}</style><main>{"".join(rows)}</main></html>\n'
    (args.out_dir / "index.html").write_text(page, encoding="utf-8")
    print(f"{len(prompts)} prompts, {len(prompts) * len(crops)} images", flush=True)
    print_link(args.out_dir / "index.html")


if __name__ == "__main__":
    main()

import argparse
import os

import cv2
import numpy as np

from diffusion_processor import DiffusionProcessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="a surreal portrait of a person made of flowers")
    parser.add_argument("--output", default="artifacts/stablefast-once.jpg")
    parser.add_argument("--input")
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--use-compel", action="store_true")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    if args.input:
        img_bgr = cv2.imread(args.input, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"Failed to read input image from {args.input}")
        img_bgr = cv2.resize(img_bgr, (args.width, args.height), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    else:
        # Fallback synthetic image for deterministic testing.
        x = np.linspace(0, 255, args.width, dtype=np.uint8)
        y = np.linspace(0, 255, args.height, dtype=np.uint8)
        xv, yv = np.meshgrid(x, y)
        img = np.stack([xv, yv, np.full_like(xv, 96)], axis=-1)

    processor = DiffusionProcessor(
        local_files_only=False,
        use_compel=args.use_compel,
        warmup="1x768x768x3",
    )
    out = processor(img, args.prompt)

    out = np.clip(out, 0, 255).astype(np.uint8)
    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(args.output, out_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise RuntimeError(f"Failed to write output image to {args.output}")

    print(args.output)


if __name__ == "__main__":
    main()

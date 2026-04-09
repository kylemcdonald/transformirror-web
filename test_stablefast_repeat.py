import argparse
import os

import cv2
import numpy as np

from diffusion_processor import DiffusionProcessor


def load_input_image(path, width, height):
    if path:
        img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"Failed to read input image from {path}")
        img_bgr = cv2.resize(img_bgr, (width, height), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    x = np.linspace(0, 255, width, dtype=np.uint8)
    y = np.linspace(0, 255, height, dtype=np.uint8)
    xv, yv = np.meshgrid(x, y)
    return np.stack([xv, yv, np.full_like(xv, 96)], axis=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="a psychedelic landscape")
    parser.add_argument("--input")
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", default="1x768x768x3")
    parser.add_argument("--use-compel", action="store_true")
    parser.add_argument("--output-dir", default="artifacts/repeat")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    img = load_input_image(args.input, args.width, args.height)
    processor = DiffusionProcessor(
        local_files_only=False,
        use_compel=args.use_compel,
        warmup=args.warmup,
    )

    stats = []
    for i in range(args.iterations):
        out = processor(img, args.prompt)
        out = np.clip(out, 0, 255).astype(np.uint8)
        mean = float(out.mean())
        min_v = int(out.min())
        max_v = int(out.max())
        stats.append((i, mean, min_v, max_v))

        out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        out_path = os.path.join(args.output_dir, f"iter-{i:03d}.jpg")
        ok = cv2.imwrite(out_path, out_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ok:
            raise RuntimeError(f"Failed to write output image to {out_path}")
        print(f"iter={i} mean={mean:.2f} min={min_v} max={max_v} output={out_path}", flush=True)

    means = [mean for _, mean, _, _ in stats]
    print(
        f"summary iterations={len(stats)} mean_min={min(means):.2f} mean_max={max(means):.2f} mean_avg={sum(means)/len(means):.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()

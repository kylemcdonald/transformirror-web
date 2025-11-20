import os
from pathlib import Path
from typing import List

import cv2
import numpy as np

from diffusion_processor import DiffusionProcessor

ROOT_DIR = Path(__file__).resolve().parent
SAMPLES_DIR = ROOT_DIR / "samples"
RESULTS_DIR = ROOT_DIR / "results"
PROMPTS_FILE = ROOT_DIR / "alt-prompts.txt"


def load_prompts(prompts_path: Path) -> List[str]:
    if not prompts_path.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_path}")
    with prompts_path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def list_image_paths(samples_dir: Path) -> List[Path]:
    if not samples_dir.exists():
        raise FileNotFoundError(f"Samples directory not found: {samples_dir}")
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    return sorted(
        [
            path
            for path in samples_dir.iterdir()
            if path.is_file() and path.suffix.lower() in image_extensions
        ]
    )


def load_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return np.float32(image) / 255.0


def save_image(image: np.ndarray, output_path: Path) -> None:
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), image)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Creating diffusion processor...")
    processor = DiffusionProcessor(local_files_only=True, gpu_id=0, use_compel=True)

    print(f"Listing images in {SAMPLES_DIR}...")
    image_paths = list_image_paths(SAMPLES_DIR)
    if not image_paths:
        print("No images found in samples directory. Exiting.")
        return

    print(f"Loading prompts from {PROMPTS_FILE}...")
    prompts = load_prompts(PROMPTS_FILE)
    if not prompts:
        print("No prompts found. Exiting.")
        return

    for image_path in image_paths:
        try:
            original_image = load_image(image_path)
        except Exception as exc:
            print(f"Skipping {image_path.name}: {exc}")
            continue

        for prompt_index, prompt in enumerate(prompts):
            padded_index = f"{prompt_index:02d}"
            output_name = f"{image_path.stem}_{padded_index}{image_path.suffix}"
            output_path = RESULTS_DIR / output_name

            print(f"Processing {image_path.name} with prompt #{prompt_index}: {prompt[:60]}...")
            try:
                result = processor(original_image, prompt)
            except Exception as exc:
                print(f"Failed to process {image_path.name} with prompt {prompt_index}: {exc}")
                continue

            save_image(result, output_path)
            print(f"Saved result to {output_path}")


if __name__ == "__main__":
    main()


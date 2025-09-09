import torch
from PIL import Image
import os
from pathlib import Path
from diffusion_processor import DiffusionProcessor
import numpy as np

def load_and_resize_images(folder_path, target_size=(1024, 1024)):
    """Load and resize all images from the given folder"""
    images = []
    for file in sorted(os.listdir(folder_path)):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, file)
            img = Image.open(img_path)
            img = img.convert('RGB')
            img = img.resize(target_size, Image.Resampling.LANCZOS)
            # Convert to numpy array and normalize to [0, 1]
            img_array = np.array(img).astype(np.float32) / 255.0
            images.append(img_array)
    return np.stack(images)

def load_prompts(prompt_file):
    """Load prompts from text file"""
    with open(prompt_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def create_grid_image(images):
    """Create a horizontal grid of images"""
    # Assuming images is a list of PIL Images
    total_width = sum(img.width for img in images)
    max_height = max(img.height for img in images)
    
    grid_img = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    
    for img in images:
        grid_img.paste(img, (x_offset, 0))
        x_offset += img.width
    
    return grid_img

def main():
    # Initialize paths
    reference_folder = "references"
    prompts_file = "prompts.txt"
    output_folder = "images"
    os.makedirs(output_folder, exist_ok=True)

    # Load and resize reference images
    reference_images = load_and_resize_images(reference_folder)
    
    # Save unprocessed grid image
    unprocessed_pil_images = []
    for img_array in reference_images:
        # Convert numpy array back to PIL Image
        img_array = (img_array * 255).astype(np.uint8)
        unprocessed_pil_images.append(Image.fromarray(img_array))
    
    # Create and save unprocessed grid
    unprocessed_grid = create_grid_image(unprocessed_pil_images)
    unprocessed_path = os.path.join(output_folder, "unprocessed.jpg")
    unprocessed_grid.save(unprocessed_path, "JPEG", quality=95)
    print("Saved unprocessed reference images grid")
    
    # Load and initialize the model with proper warmup
    processor = DiffusionProcessor(warmup="4x1024x1024x3")  # This handles warmup in init
    
    # Convert to numpy array format expected by the processor
    reference_images = reference_images.astype(np.float32)  # Already normalized from load_and_resize_images

    # Load prompts
    prompts = load_prompts(prompts_file)

    # Process each prompt
    for idx, prompt in enumerate(prompts):
        # Process the batch of 4 images with the current prompt using run()
        outputs = processor.run(
            images=reference_images,
            prompt=prompt,
            num_inference_steps=2,
            strength=0.7,
            seed=0
        )
        
        # Convert outputs to PIL images
        output_images = []
        for output in outputs:
            # outputs are already numpy arrays from run()
            img_array = (output * 255).astype(np.uint8)
            output_images.append(Image.fromarray(img_array))
        
        # Create grid image
        grid_image = create_grid_image(output_images)
        
        # Save the grid as jpg instead of png
        output_path = os.path.join(output_folder, f"output_{idx:03d}.jpg")
        grid_image.save(output_path, "JPEG", quality=95)  # High quality JPEG
        print(f"Processed and saved result for prompt {idx + 1}/{len(prompts)}")

if __name__ == "__main__":
    main() 
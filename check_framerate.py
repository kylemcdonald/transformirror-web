import numpy as np
import time
from diffusion_processor import DiffusionProcessor

# Initialize the DiffusionProcessor
processor = DiffusionProcessor(local_files_only=False)

# Create a random 1024x1024x3 image
input_image = np.random.rand(1024, 1024, 3).astype(np.float32)

# Run the test 10 times
for i in range(10):
    start_time = time.time()
    
    batch_size = 1
    
    # Perform image-to-image generation
    output_image = processor.run(
        images=[input_image] * batch_size,
        prompt="a beautiful landscape",
        num_inference_steps=2,
        strength=0.7
    )
    
    end_time = time.time()
    duration_ms = (end_time - start_time) * 1000
    
    print(f"Run {i+1}: {duration_ms/batch_size:.2f} ms")

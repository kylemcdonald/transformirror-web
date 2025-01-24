import re
import numpy as np
import time
from fixed_seed import fix_seed
import cv2

from sfast.compilers.stable_diffusion_pipeline_compiler import (
    compile,
    CompilationConfig,
)

from diffusers.utils.logging import disable_progress_bar
from diffusers import AutoPipelineForImage2Image, AutoencoderTiny
import torch
import warnings

from compel import Compel, ReturnedEmbeddingsType
from fixed_size_dict import FixedSizeDict

def build_pipe(local_files_only):
    base_model = "stabilityai/sdxl-turbo"
    vae_model = "madebyollin/taesdxl"

    local_files_only = False

    pipe = AutoPipelineForImage2Image.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        variant="fp16",
        local_files_only=local_files_only,
    )

    pipe.vae = AutoencoderTiny.from_pretrained(
        vae_model,
        torch_dtype=torch.float16,
        local_files_only=local_files_only
    )
    
    return pipe

class DiffusionProcessor:
    def __init__(self, warmup="1x1280x720x3", local_files_only=True, gpu_id=0, use_compel=True):
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)

        self.device = torch.device(f"cuda:{gpu_id}")
        with torch.cuda.device(self.device):
            
            disable_progress_bar()
            self.pipe = build_pipe(local_files_only)
            fix_seed(self.pipe)

            print(f"{self.device}: model loaded")

            config = CompilationConfig.Default()
            config.enable_xformers = True
            config.enable_triton = True
            config.enable_cuda_graph = True
            self.pipe = compile(self.pipe, config=config)

            print(f"{self.device}: model compiled")

            self.pipe.to(device=self.device, dtype=torch.float16)
            self.pipe.set_progress_bar_config(disable=True)

            print(f"{self.device}: model moved", flush=True)
            
            if use_compel:
                self.compel = Compel(
                    tokenizer=[self.pipe.tokenizer, self.pipe.tokenizer_2],
                    text_encoder=[self.pipe.text_encoder, self.pipe.text_encoder_2],
                    returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                    requires_pooled=[False, True],
                    device=self.device
                )
                self.prompt_cache = FixedSizeDict(32)
                print(f"{self.device}: prepared compel")
            else:
                self.compel = None

            self.generator = torch.Generator(device=self.device).manual_seed(0)
            
            if warmup:
                warmup_shape = [int(e) for e in warmup.split("x")]
                images = np.zeros(warmup_shape, dtype=np.float32)
                for i in range(2):
                    print(f"{self.device}: warmup {warmup} {i+1}/2")
                    start_time = time.time()
                    self.run(
                        images,
                        prompt="warmup",
                        num_inference_steps=2,
                        strength=1.0
                    )
                print(f"{self.device}: warmup finished", flush=True)
            
            self.call_durations = []
            self.last_print_time = time.time()

    def embed_prompt(self, prompt):
        if prompt not in self.prompt_cache:
            with torch.no_grad():
                print(f"{self.device}: embedding prompt", prompt)
                self.prompt_cache[prompt] = self.compel(prompt)
        return self.prompt_cache[prompt]
    
    def meta_embed_prompt(self, prompt):
        pattern = r'\("(.*?)"\s*,\s*"(.*?)"\)\.blend\((.*?),(.*?)\)'
        match = re.search(pattern, prompt)
        if not match:
            return self.embed_prompt(prompt)
        str1, str2, t1, t2 = match.groups()
        t1 = float(t1)
        t2 = float(t2)
        cond1, pool1 = self.embed_prompt(str1)
        cond2, pool2 = self.embed_prompt(str2)
        cond = cond1 * t1 + cond2 * t2
        pool = pool1 * t1 + pool2 * t2
        return cond, pool
    
    def run(self, images, prompt, num_inference_steps, strength, seed=None):
        with torch.cuda.device(self.device):
            strength = min(max(1 / num_inference_steps, strength), 1)
            if seed is not None:
                self.generator = torch.Generator().manual_seed(seed)
            kwargs = {}
            if self.compel is not None:
                conditioning, pooled = self.meta_embed_prompt(prompt)
                batch_size = len(images)
                conditioning_batch = conditioning.expand(batch_size, -1, -1)
                pooled_batch = pooled.expand(batch_size, -1)
                kwargs["prompt_embeds"] = conditioning_batch
                kwargs["pooled_prompt_embeds"] = pooled_batch
            else:
                kwargs["prompt"] = [prompt] * len(images)
            return self.pipe(
                image=images,
                generator=self.generator,
                num_inference_steps=num_inference_steps,
                guidance_scale=0,
                strength=strength,
                output_type="np",
                **kwargs
            ).images

    def __call__(self, img, prompt):
        start_time = time.time()
        
        img = cv2.resize(img, (720, 1280), interpolation=cv2.INTER_LINEAR)

        img = np.float32(img) / 255
        filtered_img = self.run(
            images=[img],
            seed=0,
            prompt=prompt.decode("utf-8"),
            num_inference_steps=2,
            strength=0.7
        )[0]
        filtered_img = np.uint8(filtered_img * 255)
        
        end_time = time.time()
        duration = (end_time - start_time) * 1000  # Convert to milliseconds
        self.call_durations.append(duration)
        
        current_time = time.time()
        if current_time - self.last_print_time >= 5:
            mean_duration = np.mean(self.call_durations)
            min_duration = np.min(self.call_durations)
            max_duration = np.max(self.call_durations)
            print(f"diffusion mean: {mean_duration:.1f}ms, min: {min_duration:.1f}ms, max: {max_duration:.1f}ms")
            self.call_durations.clear()
            self.last_print_time = current_time
        
        return filtered_img

from diffusers import DiffusionPipeline, DDIMScheduler, StableDiffusionPipeline
from transformers.models.paligemma.convert_paligemma_weights_to_hf import device

from model.pipeline import SelfRectificationPipeline
import torch
from model.KVInjection import register_kv_injection
from PIL import Image
from utils.io import load_image, save_image
from utils.device import device


model_path = './pretrained/stable-diffusion-v1-4'
num_inference_steps = 50

scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                          clip_sample=False,
                          set_alpha_to_one=False)
pipe: SelfRectificationPipeline = SelfRectificationPipeline.from_pretrained(model_path, scheduler=scheduler)
register_kv_injection(pipe, num_inference_steps)
for name in pipe.unet.register_dict.keys():
    if not (name.startswith('.down') or name.startswith('.mid') or name.startswith('.up')):
        print(name)
pipe.pipeline.to(device)
print(f'running on device:{device}')

texture_ref = 'images/aug/203.jpg'
# target_image = load_image('images/tgts/203-1.jpg', True, device)
texture_ref = load_image(texture_ref, True, device)
noised_ref = pipe.invert(texture_ref, num_inference_steps, save_kv=False, use_injection=False)
image = pipe.sampling(noised_ref, num_inference_steps=num_inference_steps, save_kv=False, use_injection=False)

save_image(image, 'result.jpg')

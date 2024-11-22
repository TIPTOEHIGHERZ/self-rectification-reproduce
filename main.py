from diffusers import DiffusionPipeline, DDIMScheduler, StableDiffusionPipeline
import torch

from model.pipeline import SelfRectificationPipeline
from model.KVInjection import register_kv_injection, KVInjectionAgent
from PIL import Image
from utils.io import load_image, save_image
from utils.device import device


model_path = './pretrained/stable-diffusion-v1-4'
num_inference_steps = 50

scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                          clip_sample=False,
                          set_alpha_to_one=False)
pipe: SelfRectificationPipeline = SelfRectificationPipeline.from_pretrained(model_path, scheduler=scheduler)
_, count = register_kv_injection(pipe,
                                 KVInjectionAgent(20, num_inference_steps, 10, 10),
                                 num_inference_steps)
print(pipe.unet.register_count)
assert hasattr(pipe.unet, 'kv_injection_agent')
pipe.pipeline.to(device)
print(f'running on device:{device}')

texture_ref = ['images/aug/203.jpg',
               'images/aug/203-1.jpg',
               'images/aug/203-2.jpg',
               'images/aug/203-3.jpg']
target_image = load_image('images/tgts/203-1.jpg', True, device)
texture_ref = load_image(texture_ref, True, device)
image_coarse = pipe(target_image, texture_ref, num_inference_steps=num_inference_steps)
save_image(image_coarse, 'result_coarse.jpg')
register_kv_injection(pipe,
                      KVInjectionAgent(5, num_inference_steps, 0, activate_layer_num=None),
                      num_inference_steps)
image_fine = pipe(image_coarse, texture_ref, num_inference_steps=num_inference_steps)
save_image(image_fine, 'result_fine.jpg')

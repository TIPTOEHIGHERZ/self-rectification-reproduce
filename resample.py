from diffusers import DiffusionPipeline, DDIMScheduler, StableDiffusionPipeline
import torch

from model.pipeline import SelfRectificationPipeline
from model.KVInjection import register_kv_injection, KVInjectionAgent
from PIL import Image
from utils.io import load_image, save_image
from utils.device import device
from utils.functionals import get_mask


model_path = './pretrained/stable-diffusion-v1-4'
num_inference_steps = 50

scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                          clip_sample=False,
                          set_alpha_to_one=False)
pipe: SelfRectificationPipeline = SelfRectificationPipeline.from_pretrained(model_path, scheduler=scheduler)
_, count = register_kv_injection(pipe,
                                 KVInjectionAgent(20, num_inference_steps, 10, 10),
                                 num_inference_steps)

image = load_image('images/aug/203.jpg', to_batch=True).cuda()
pipe.to('cuda')
latents = pipe.invert(image, num_inference_steps, '')
image_resample = pipe.sampling(latents, num_inference_steps=num_inference_steps, prompt='a leaf')
save_image(image_resample, 'image_resample.jpg')

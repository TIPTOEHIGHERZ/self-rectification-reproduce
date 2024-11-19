import torch
from diffusers import DiffusionPipeline, StableDiffusionPipeline, DDIMScheduler
from model.pipeline import SelfRectificationPipeline
import os


os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'

# TODO modify scheduler
model_path = './pretrained/stable-diffusion-v1-4'
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                          clip_sample=False,
                          set_alpha_to_one=False)
pipeline: SelfRectificationPipeline = SelfRectificationPipeline.from_pretrained(model_path, scheduler=scheduler)

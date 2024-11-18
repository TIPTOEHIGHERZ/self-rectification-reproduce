import torch
from diffusers import DiffusionPipeline, StableDiffusionPipeline, DDIMScheduler
from model.pipeline import SelfRectificationPipeline
import os


os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'

# TODO modify scheduler
model_path = './pretrained/stable-diffusion-v1-4'
pipeline: SelfRectificationPipeline = SelfRectificationPipeline.from_pretrained(model_path, scheduler=None)

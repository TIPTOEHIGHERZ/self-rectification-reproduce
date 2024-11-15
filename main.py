import torch
from diffusers import DiffusionPipeline
from model.pipeline import SelfRectificationPipeline

# TODO modify scheduler
pipeline: SelfRectificationPipeline = SelfRectificationPipeline.from_pretrained('./pretrained/stable-diffusion-v1-4')

from diffusers import DiffusionPipeline, DDIMScheduler, StableDiffusionPipeline
from model.pipeline import SelfRectificationPipeline
from model.KVInjection import register_kv_saver
from PIL import Image


model_path = './pretrained/stable-diffusion-v1-4'

scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                          clip_sample=False,
                          set_alpha_to_one=False)
pipe: SelfRectificationPipeline = SelfRectificationPipeline.from_pretrained(model_path, scheduler=scheduler)
counts = register_kv_saver(pipe)



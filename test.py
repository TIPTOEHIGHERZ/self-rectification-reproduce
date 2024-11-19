from diffusers import DiffusionPipeline
from model.pipeline import SelfRectificationPipeline
from model.KVInjection import KVSaver


model_path = './pretrained/stable-diffusion-v1-4'

pipe: SelfRectificationPipeline = DiffusionPipeline.from_pretrained(model_path)
KVSaver.register_kv_saver(pipe)

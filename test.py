from diffusers import DiffusionPipeline, DDIMScheduler, StableDiffusionPipeline
from model.pipeline import SelfRectificationPipeline
from model.KVInjection import KVSaver
from PIL import Image


model_path = './pretrained/stable-diffusion-v1-4'

scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                          clip_sample=False,
                          set_alpha_to_one=False)
pipe: SelfRectificationPipeline = SelfRectificationPipeline.from_pretrained(model_path, scheduler=scheduler)
counts = KVSaver.register_kv_saver(pipe)

# todo test whether still functional
image: Image.Image = pipe.pipeline(prompt='a horse standing in mud', height=512, width=512)
image.show()
image.save('./result.jpg')

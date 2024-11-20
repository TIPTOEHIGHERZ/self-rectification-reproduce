from diffusers import DiffusionPipeline, DDIMScheduler, StableDiffusionPipeline
from model.pipeline import SelfRectificationPipeline
import torch
from model.KVInjection import register_kv_injection
from PIL import Image
from utils.io import load_image, save_image


model_path = './pretrained/stable-diffusion-v1-4'
num_inference_steps = 5

scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                          clip_sample=False,
                          set_alpha_to_one=False)
pipe: SelfRectificationPipeline = SelfRectificationPipeline.from_pretrained(model_path, scheduler=scheduler)


# generate function test
image = torch.rand([1, 3, 512, 512])
image = pipe.sampling(image, 50, '')
save_image(image, 'result.jpg')
# image = pipe.pipeline('', 512, 512, num_inference_steps=50).images
# save_image(image[0], 'result.jpg')


# counts = register_kv_injection(pipe, num_inference_steps)
# print(counts)
# print(pipe.unet.register_dict.keys())

# target_image = load_image('./images/tgts/203-1.jpg').unsqueeze(0)
# inversion_reference = target_image

# pipe.invert(inversion_reference, num_inference_steps, '', desc='Inversion reference inverting', save_kv=True, use_injection=False)
# pipe.invert(target_image, num_inference_steps, '', desc='Target image inverting', save_kv=False, use_injection=True)

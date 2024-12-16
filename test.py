from diffusers import DiffusionPipeline, DDIMScheduler, StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel
from model.pipeline import SelfRectificationPipeline
import torch
from torchvision.transforms.transforms import ToTensor
from torch.optim.adam import Adam
from model.KVInjection import register_kv_injection
from PIL import Image
from utils.io import load_image, save_image
from utils.device import device
import tqdm
import pathlib
import os


def sample_interval(image: torch.Tensor, image_name, interval, input_size, resample_times, use_norm):
    images = pipeline.multi_resample(image, num_inference_steps, interval, resample_times, use_norm)
    images = torch.nn.functional.interpolate(images, input_size)
    image_name = pathlib.Path(image_name)
    pre_fix = 'o_use_norm-' if use_norm else 'no_norm-'
    dir_name = os.path.join('result', f'{interval[0]}-{interval[1]}-' + image_name.stem)
    os.makedirs(dir_name, exist_ok=True)
    for i, image in enumerate(images):
        modified_name = os.path.join(dir_name, pre_fix + image_name.stem + f'-{i + 1}' + image_name.suffix)
        save_image(image.unsqueeze(0).detach(), modified_name)

    return

def init_image(pt: str):
    image = Image.open(pt)
    to_tensor = ToTensor()
    image: torch.Tensor = to_tensor(image)
    image = image.unsqueeze(0)

    return image


def sample(image: torch.Tensor, image_name, intervals: list, input_size, resample_times, use_norm):
    for interval in intervals:
        sample_interval(image, image_name, interval, input_size, resample_times, use_norm)

    return


model_path = './pretrained/stable-diffusion-v1-4'
num_inference_steps = 100
device = 'cuda:1'
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                          clip_sample=False,
                          set_alpha_to_one=False)
pipeline: SelfRectificationPipeline = SelfRectificationPipeline.from_pretrained(model_path, scheduler=scheduler)
pipeline.to(device)
# intervals = [[0, 100], [90, 100], [80, 90], [80, 90], [70, 80], [60, 70], [50, 60], [40, 50], [30, 40], [20, 30], [10, 20], [0, 10]]
min_period = 10
# ranges = range(min_period, num_inference_steps, min_period)
intervals = list()
for p in range(min_period, num_inference_steps + 1, min_period):
    for j in range(0, num_inference_steps - p + 1, min_period):
        assert j + p <= num_inference_steps, f'{j} + {p} is greater than {num_inference_steps}'
        intervals.append([j, j + p])

print(intervals)
resample_times = 3

images = ['real.jpg', 'gen.jpg']

for image_name in images:
    image = init_image(image_name)
    input_size = image.shape[-2:]
    image = torch.nn.functional.interpolate(image, [512, 512])
    image = image.to(device)

    sample(image, image_name=image_name, intervals=intervals, input_size=input_size, resample_times=resample_times, use_norm=False)
    sample(image, image_name=image_name, intervals=intervals, input_size=input_size, resample_times=resample_times, use_norm=True)


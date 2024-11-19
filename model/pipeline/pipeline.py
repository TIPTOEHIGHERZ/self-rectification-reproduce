import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel, DDIMScheduler, DiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
import logging
from logging import Logger
import os
import tqdm


class SelfRectificationPipeline:
    def __init__(self, pipeline: StableDiffusionPipeline=None):
        self.logger = self.get_logger(self.__class__.__name__)
        self.pipeline = None
        # self.unet: UNet2DConditionModel = pipeline.unet
        # self.scheduler: DDIMScheduler = pipeline.scheduler
        # self.vae: AutoencoderKL = pipeline.vae

        if pipeline is not None:
            self.pipeline: StableDiffusionPipeline = pipeline
            self.text_encoder: CLIPTextModel = pipeline.text_encoder
            self.tokenizer: CLIPTokenizer = pipeline.tokenizer
            self.unet: UNet2DConditionModel = pipeline.unet
            self.scheduler: DDIMScheduler = pipeline.scheduler
            self.vae: AutoencoderKL = pipeline.vae

        return

    def init(self, pipeline: StableDiffusionPipeline):
        self.pipeline: StableDiffusionPipeline = pipeline
        self.unet: UNet2DConditionModel = pipeline.unet
        self.scheduler: DDIMScheduler = pipeline.scheduler
        self.vae: AutoencoderKL = pipeline.vae

        return

    @staticmethod
    def from_pretrained(model_path: str, **kwargs):
        for key, value in kwargs.copy().items():
            # kwargs set to None isn't allowed to pass to from_pretrained
            if value is None:
                kwargs.pop(key)

        pipeline = DiffusionPipeline.from_pretrained(model_path)
        pipeline = SelfRectificationPipeline(pipeline)

        return pipeline

    @staticmethod
    def get_logger(logger_name: str) -> Logger:
        return logging.getLogger(logger_name)

    def add_noise(self,
                  sample: torch.Tensor,
                  timestep,
                  encoder_hidden_states: torch.Tensor):
        num_train_steps, num_inference_steps = len(self.scheduler.alphas), self.scheduler.num_inference_steps
        next_step = timestep + num_train_steps // num_inference_steps

        noise_pred = self.unet(sample, timestep, encoder_hidden_states)
        alpha = self.scheduler.alphas_cumprod

        alpha_t = alpha[timestep]
        beta_t = 1 - alpha_t
        alpha_next = alpha[next_step]
        beta_next = 1 - alpha_next

        x_0 = (sample - beta_t.sqrt() * noise_pred) / alpha_t.sqrt()
        x_next = alpha_next.sqrt() * x_0 + beta_next.sqrt() * noise_pred

        return x_next

    @torch.no_grad()
    def invert(self,
               sample: torch.Tensor,
               prompt='',
               use_clamp=False,
               verbose=False,
               desc=''):
        batch_size = sample.shape[0]
        if isinstance(prompt, str):
            prompt = [prompt] * batch_size
        elif len(prompt) == 1:
            prompt = prompt * batch_size
        elif len(prompt) != batch_size:
            raise ValueError(f'Prompts should have the same number as the sample!,'
                             f'{batch_size} samples accept, but {len(prompt)} are given.')

        timesteps = reversed(self.scheduler.timesteps)
        iteration = tqdm.tqdm(timesteps, desc=desc) if verbose else timesteps
        device = sample.device

        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )
        encoder_hidden_states = self.text_encoder(tokens['input_ids'].to(device))[0]
        for timestep in iteration:
            sample = self.add_noise(sample, timestep, encoder_hidden_states)

        if use_clamp:
            noised_latents = torch.clamp(noised_latents, -1., 1.)
        return

    def denoising_process(self,
                          image: torch.Tensor,
                          time_steps,
                          encoder_hidden_states,
                          desc=''):
        latents = self.vae.encode(image, return_dict=False)

        self.scheduler.set_timesteps(time_steps)
        time_steps = reversed(self.scheduler.timesteps)

        x_t = image
        x_states = [x_t]
        iteration = tqdm.tqdm(reversed(self.scheduler.timesteps), desc=desc)
        for t in iteration:
            x_next = self.add_noise(x_t, t, encoder_hidden_states)

        return x_t, x_states

    def structure_preserve_invert(self,
                                  inversion_ref: torch.Tensor,
                                  inversion_target: torch.Tensor,
                                  time_steps,
                                  encoder_hidden_states,
                                  eta=0.):
        pass
        return

    def predict_x_prev(self, x_t: torch.Tensor, t, noise_pred: torch.Tensor, eta=0.):
        batch_size = noise_pred.shape[0]

        alpha = self.scheduler.alphas_cumprod

        train_steps = alpha.shape[0]

        t_prev = t - train_steps // self.scheduler.num_inference_steps
        alpha_t = alpha[t].repeat(batch_size).reshape(-1, [1] * (noise_pred.ndim - 1))
        alpha_t_prev = alpha[t_prev].repeat(batch_size).reshape(-1, [1] * (noise_pred.ndim - 1))

        x_start = (x_t - (1 - alpha_t).sqrt() * noise_pred / alpha_t.sqrt())
        sigma = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t)) * torch.sqrt(1 - (alpha_t / alpha_t_prev))
        direct_x_t = (1 - alpha_t_prev - sigma ** 2.0).sqrt() * noise_pred
        random_noise = sigma * torch.randn_like(x_t, device=x_t.device)

        x_t_prev = alpha_t_prev.sqrt() * x_start + direct_x_t + random_noise

        return  x_t_prev

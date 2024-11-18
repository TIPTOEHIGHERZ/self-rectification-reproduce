import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel, DDIMScheduler, DiffusionPipeline
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
        noise_pred = self.unet(sample, timestep, encoder_hidden_states)
        # latents_start =

    @torch.no_grad()
    def invert(self, latents, num_inference_steps=None, use_clamp=False):
        """
        add noise
        :param latents: latents without noise
        :param num_inference_steps: steps to take
        :param use_clamp: whether clamp the noised image
        :return:
        """
        num_inference_steps = num_inference_steps if num_inference_steps \
                                                     is not None else self.scheduler.num_inference_steps
        noised_latents = latents
        for i in range(num_inference_steps):
            noised_latents += torch.randn_like(latents)

        if use_clamp:
            noised_latents = torch.clamp(noised_latents, -1., 1.)
        return noised_latents

    def denoising_process(self,
                          image: torch.Tensor,
                          time_steps,
                          encoder_hidden_states,
                          eta=0.,
                          desc=''):
        latents = self.vae.encode(image, return_dict=False)

        self.scheduler.set_timesteps(time_steps)
        time_steps = self.scheduler.timesteps

        x_t = image
        x_states = [x_t]
        for t in tqdm.trange(time_steps, desc=desc):
            noise_pred = self.unet(latents, time_steps, encoder_hidden_states).sample
            x_t = self.predict_x_prev(x_t, t, noise_pred, eta)
            x_states.append(x_t)

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

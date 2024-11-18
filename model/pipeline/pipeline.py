import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel, DDIMScheduler, DiffusionPipeline
import logging
import os
import tqdm


class SelfRectificationPipeline:
    def __init__(self, pipeline: StableDiffusionPipeline=None):
        self.logger = None
        self.get_logger(self.__class__.__name__)
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
            if value is None:
                kwargs.pop(key)

        pipeline = DiffusionPipeline.from_pretrained(model_path)
        pipeline = SelfRectificationPipeline(pipeline)

        return pipeline

    def get_logger(self, logger_name: str):
        self.logger = logging.getLogger(logger_name)

        return

    def invert(self):
        pass

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

    def noising_process(self):
        pass

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

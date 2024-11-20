import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel, DDIMScheduler, DiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
import logging
from logging import Logger
import os
import tqdm


def check_prompt(prompt, batch_size):
    if isinstance(prompt, str):
        prompt = [prompt] * batch_size
    elif len(prompt) == 1:
        prompt = prompt * batch_size
    elif len(prompt) != batch_size:
        raise ValueError(f'Prompts should have the same number as the sample!,'
                         f'{batch_size} samples accept, but {len(prompt)} are given.')

    return prompt


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

        pipeline = DiffusionPipeline.from_pretrained(model_path, **kwargs)
        pipeline = SelfRectificationPipeline(pipeline)

        return pipeline

    @staticmethod
    def get_logger(logger_name: str) -> Logger:
        return logging.getLogger(logger_name)
    
    @torch.no_grad()
    def add_noise(self,
                  sample: torch.Tensor,
                  timestep,
                  encoder_hidden_states: torch.Tensor,
                  cross_attention_kwargs):
        num_train_steps, num_inference_steps = len(self.scheduler.alphas), self.scheduler.num_inference_steps
        next_step = min(timestep + num_train_steps // num_inference_steps, num_train_steps - 1)

        noise_pred = self.unet(sample, timestep, encoder_hidden_states, cross_attention_kwargs=cross_attention_kwargs).sample
        alpha = self.scheduler.alphas_cumprod

        alpha_t = alpha[timestep]
        beta_t = 1 - alpha_t
        alpha_next = alpha[next_step]
        beta_next = 1 - alpha_next

        x_0 = (sample - beta_t.sqrt() * noise_pred) / alpha_t.sqrt()
        x_next = alpha_next.sqrt() * x_0 + beta_next.sqrt() * noise_pred

        return x_next

    @torch.no_grad()
    def remove_noise(self, 
                     sample: torch.Tensor, 
                     timestep, 
                     encoder_hidden_states: torch.Tensor, 
                     eta=0.,
                     guidance_scale=7.5,
                     cross_attention_kwargs=None):
        num_train_steps, num_inference_steps = len(self.scheduler.alphas), self.scheduler.num_inference_steps
        pre_step = max(timestep - num_train_steps // num_inference_steps, 0)

        alpha = self.scheduler.alphas_cumprod
        alpha_t = alpha[timestep]
        beta_t = 1 - alpha_t
        alpha_pre = alpha[pre_step]
        beta_pre = 1 - alpha_pre
        sigma_t = eta * (beta_pre / beta_t).sqrt() * (1 - alpha_t / alpha_pre).sqrt()

        latens_input = torch.concat([sample] * 2)
        encoder_hidden_states = torch.concat([encoder_hidden_states] * 2)
        noise_pred = self.unet(latens_input, timestep, encoder_hidden_states, cross_attention_kwargs=cross_attention_kwargs).sample
        noise_uncond, noise_cond = noise_pred.chunk(2)
        noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        x_pre = self.scheduler.step(noise_pred, timestep, sample).prev_sample
        # x_0 = (sample - beta_t.sqrt() * noise_pred) / alpha_t.sqrt()
        # x_pre = alpha_pre.sqrt() * x_0 + (beta_pre - sigma_t ** 2) * noise_pred + \
        #         sigma_t * torch.randn_like(sample)
        return x_pre

    @torch.no_grad()
    def invert(self,
               image: torch.Tensor,
               num_inference_steps,
               prompt='',
               verbose=True,
               desc='DDIM Inverting',
               **cross_attention_kwargs):
        batch_size = image.shape[0]
        device = image.device

        self.scheduler.set_timesteps(num_inference_steps)
        prompt = check_prompt(prompt, batch_size)

        latents = self.vae.encode(image, return_dict=False)[0].mode()
        timesteps = reversed(self.scheduler.timesteps)
        iteration = tqdm.tqdm(timesteps, desc=desc) if verbose else timesteps

        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )
        encoder_hidden_states = self.text_encoder(tokens['input_ids'].to(device))[0]
        for timestep in iteration:
            latents = self.add_noise(latents, timestep, encoder_hidden_states, cross_attention_kwargs)

        return

    @torch.no_grad()
    def sampling(self,
                 image: torch.Tensor,
                 num_inference_steps,
                 prompt='',
                 verbose=True,
                 desc='DDIM Sampling',
                 eta=0.,
                 **cross_attention_kwargs):
        device = image.device
        batch_size = image.shape[0]
        prompt = check_prompt(prompt, batch_size)

        self.scheduler.set_timesteps(num_inference_steps)
        latents = self.vae.encode(image, return_dict=False)[0].mode()
        latents = torch.randn([1, 4, 512 // self.pipeline.vae_scale_factor, 512 // self.pipeline.vae_scale_factor])
        iteration = tqdm.tqdm(self.scheduler.timesteps, desc=desc) if verbose else self.scheduler.timesteps

        # tokens = self.tokenizer(
        #     prompt,
        #     padding="max_length",
        #     max_length=77,
        #     return_tensors="pt"
        # )
        # encoder_hidden_states = self.text_encoder(tokens['input_ids'].to(device))[0]
        encoder_hidden_states = self.pipeline.encode_prompt(prompt, device, 1, True)[0]
        for timestep in iteration:
            latents = self.remove_noise(latents, timestep, encoder_hidden_states, eta, 
                                        cross_attention_kwargs=cross_attention_kwargs)

        result = self.vae.decode(latents / self.vae.config.scaling_factor).sample
        result = result.clamp(-1, -1)
        result = (result + 1) / 2

        return result

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

        return x_t_prev

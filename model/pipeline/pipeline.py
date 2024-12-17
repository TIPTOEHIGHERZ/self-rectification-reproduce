import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel, DDIMScheduler, DiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
import logging
from logging import Logger
import os
from typing import Union
from diffusers.models.attention_processor import Attention
from model.KVInjection import KVSaver
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
    def __init__(self, pipeline: StableDiffusionPipeline=None, device='cpu'):
        self.logger = self.get_logger(self.__class__.__name__)
        self.pipeline = None
        self.device = device
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

    def to(self, device: str):
        self.pipeline.to(device)
        self.device = device
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
    def image2latents(self, image: torch.Tensor):
        # normalize
        image = image * 2 - 1
        latents = self.vae.encode(image, return_dict=False)[0].mean
        latents *= self.vae.config.scaling_factor
        # latents *= 0.18215

        return latents

    def latents2image(self, latents: torch.Tensor):
        # denormalize
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        # image = self.vae.decode(latents / 0.18215, return_dict=False)[0]
        image = image.clamp(-1, 1)
        image = (image + 1) / 2

        return image

    @torch.no_grad()
    def add_noise(self,
                  sample: torch.Tensor,
                  timestep,
                  encoder_hidden_states: torch.Tensor,
                  cross_attention_kwargs,
                  return_process=False):
        num_train_steps, num_inference_steps = len(self.scheduler.alphas), self.scheduler.num_inference_steps
        next_step = min(timestep + num_train_steps // num_inference_steps, num_train_steps - 1)

        noise_pred = self.unet(sample, timestep, encoder_hidden_states,
                               cross_attention_kwargs=cross_attention_kwargs, return_dict=False)[0]
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
                     ground_truth: torch.Tensor = None,
                     cross_attention_kwargs=None):
        num_train_steps, num_inference_steps = len(self.scheduler.alphas), self.scheduler.num_inference_steps
        pre_step = max(timestep - num_train_steps // num_inference_steps, 0)

        alpha = self.scheduler.alphas_cumprod
        alpha_t = alpha[timestep]
        beta_t = 1 - alpha_t
        alpha_pre = alpha[pre_step]
        beta_pre = 1 - alpha_pre
        sigma_t = eta * (beta_pre / beta_t).sqrt() * (1 - alpha_t / alpha_pre).sqrt()

        # latens_input = torch.concat([sample] * 2)
        # encoder_hidden_states = torch.concat([encoder_hidden_states] * 2)
        latens_input = sample
        noise_pred = self.unet(latens_input, timestep, encoder_hidden_states,
                               cross_attention_kwargs=cross_attention_kwargs, return_dict=False)[0]
        # noise_uncond, noise_cond = noise_pred.chunk(2)
        # noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        # x_pre = self.scheduler.step(noise_pred, timestep, sample).prev_sample
        x_0 = (sample - beta_t.sqrt() * noise_pred) / alpha_t.sqrt()
        x_pre = alpha_pre.sqrt() * x_0 + (beta_pre - sigma_t ** 2).sqrt() * noise_pred + \
                sigma_t * torch.randn_like(sample)
        return x_pre
    
    @torch.no_grad()
    def prompt2embeddings(self, prompt: Union[str, list[str]]) -> torch.Tensor:
        if isinstance(prompt, str):
            prompt = [prompt]
        
        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )
        embeddings = self.text_encoder(tokens['input_ids'].to(self.device))[0]
        return embeddings

    @torch.no_grad()
    def invert(self,
               image: torch.Tensor,
               num_inference_steps,
               prompt='',
               verbose=True,
               desc='DDIM Inverting',
               is_latents=False,
               interval=None,
               save_period=None,
               **cross_attention_kwargs):
        assert interval is None or (len(interval) == 2 and interval[1] >= interval[0])
        if interval is None:
            interval = [0, num_inference_steps]

        batch_size = image.shape[0]
        device = image.device
        self.scheduler.set_timesteps(num_inference_steps)
        prompt = check_prompt(prompt, batch_size)
        latents = self.image2latents(image) if not is_latents else image
        timesteps = reversed(self.scheduler.timesteps[interval[0]: interval[1]])
        iteration = tqdm.tqdm(timesteps, desc=desc) if verbose else timesteps
        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )
        encoder_hidden_states = self.text_encoder(tokens['input_ids'].to(device))[0]
        latents_list = list()
        for i, timestep in enumerate(iteration):
            latents = self.add_noise(latents, timestep, encoder_hidden_states, cross_attention_kwargs)
            if save_period and i % save_period == 0:
                latents_list.append(latents.cpu())

        if save_period:
            return latents, latents_list

        return latents

    def resample(self, image: torch.Tensor, num_inference_steps: int, interval, resample_times=1):
        assert isinstance(interval[0], int)
        # interval = [interval] * resample_times if isinstance(interval[0], int) else interval
        latents = self.invert(image, num_inference_steps=num_inference_steps)
        latents = self.sampling(latents, num_inference_steps=num_inference_steps, return_latents=True, interval=[0 , interval[0]])

        for t in range(resample_times - 1):
            latents = self.sampling(latents, num_inference_steps=num_inference_steps, return_latents=True, interval=interval)
            latents = self.invert(latents, num_inference_steps=num_inference_steps, is_latents=True, interval=interval)
        
        return self.sampling(latents, num_inference_steps=num_inference_steps, interval=[interval[0], num_inference_steps])
    
    def multi_resample(self, image: torch.Tensor, num_inference_steps: int, interval, resample_times=1, use_norm=False):
        assert isinstance(interval[0], int)
        latents = self.invert(image, num_inference_steps=num_inference_steps)
        latents = self.sampling(latents, num_inference_steps=num_inference_steps, return_latents=True, interval=[0 , interval[0]])

        std = torch.std(latents, dim=(-1, -2, -3), keepdim=True)
        m = torch.mean(latents, dim=(-1, -2, -3), keepdim=True)

        latents_list = [latents]
        for t in range(resample_times - 1):
            latents = self.sampling(latents, num_inference_steps=num_inference_steps, return_latents=True, interval=interval)
            latents = self.invert(latents, num_inference_steps=num_inference_steps, is_latents=True, interval=interval)
            latents_list.append(latents)

        latents = torch.concat(latents_list, dim=0)
        latents = self.invert(latents, num_inference_steps=num_inference_steps, is_latents=True, interval=[0, interval[0]])

        # std = torch.std(latents, dim=(-1, -2, -3), keepdim=True)
        # m = torch.mean(latents, dim=(-1, -2, -3), keepdim=True)
        # print(m, '\n', std)
        std_ = torch.std(latents, dim=(-1, -2, -3), keepdim=True)
        m_ = torch.mean(latents, dim=(-1, -2, -3), keepdim=True)
        if use_norm:
            # latents = (latents - m) / std
            latents = (latents - (m_ - m)) / (std_ / std)
        images = self.sampling(latents, num_inference_steps=num_inference_steps)
        return images

    @torch.no_grad()
    def sampling(self,
                 latents: torch.Tensor=None,
                 width=512,
                 height=512,
                 num_inference_steps=50,
                 prompt='',
                 verbose=True,
                 desc='DDIM Sampling',
                 eta=0.,
                 return_latents=False,
                 interval=None,
                 mask=None,
                 cond_noised_latents_list: list[torch.Tensor] = None,
                 **cross_attention_kwargs):
        assert interval is None or (len(interval) == 2 and interval[1] >= interval[0])
        assert not((mask is None) ^ (cond_noised_latents_list is None)), 'mask and list should be None or not None in the same time'

        if interval is None:
            interval = [0, num_inference_steps]

        device = self.vae.device
        batch_size = latents.shape[0] if latents is not None else 1
        prompt = check_prompt(prompt, batch_size)

        self.scheduler.set_timesteps(num_inference_steps)
        # latents = self.vae.encode(image, return_dict=False)[0].mode()
        latents = torch.randn([1, 4, width // self.pipeline.vae_scale_factor, height // self.pipeline.vae_scale_factor], device=device) \
        if latents is None else latents

        timesteps = self.scheduler.timesteps[interval[0]: interval[1]]
        iteration = tqdm.tqdm(timesteps, desc=desc) if verbose else timesteps

        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            return_tensors="pt"
        )
        encoder_hidden_states = self.text_encoder(tokens['input_ids'].to(device))[0]
        # encoder_hidden_states, _ = self.pipeline.encode_prompt(prompt, device, 1, True)
        for i, timestep in enumerate(iteration):
            latents = self.remove_noise(latents, timestep, encoder_hidden_states, eta,
                                        cross_attention_kwargs=cross_attention_kwargs)
            if mask is not None:
                # 用invert过程中的状态override掉采样得到的噪声，让其更加贴近给定的target
                latents[mask] = cond_noised_latents_list[i].to(self.device)[mask]

        # image = self.vae.decode(latents / self.vae.config.scaling_factor).sample
        # image = image.clamp(-1, 1)
        # image = (image + 1) / 2
        if return_latents:
            return latents

        image = self.latents2image(latents)
        # image = self.latents2image(latents)
        # do_denormalize = [True] * image.shape[0]
        # image = self.pipeline.image_processor.postprocess(image, do_denormalize=do_denormalize, output_type='pt')

        return image

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

    @torch.no_grad()
    def structure_preserving_invert(self,
                                    target_image: torch.Tensor,
                                    inversion_reference: torch.Tensor = None,
                                    num_inference_steps=50):
        inversion_reference = target_image if inversion_reference is None else inversion_reference

        self.invert(inversion_reference, num_inference_steps, desc='KV preserve invert',
                    prompt='', save_kv=True, use_injection=False, save_period=1)
        cond_noised_latents, latents_list = self.invert(target_image, num_inference_steps,
                                                        desc='Noised latents generate invert',
                                                        prompt='', save_kv=False, use_injection=True, save_period=1)

        return cond_noised_latents, latents_list

    @torch.no_grad()
    def fine_texture_sampling(self,
                              cond_noised_latents: torch.Tensor,
                              texture_reference: torch.Tensor,
                              num_inference_steps=50,
                              mask=None,
                              cond_noised_latents_list=None):
        self.invert(texture_reference, num_inference_steps, prompt='', desc='Sampling invert',
                    save_kv=True, use_injection=False, save_period=1)
        # denoising process!
        image = self.sampling(cond_noised_latents,
                              num_inference_steps=num_inference_steps,
                              prompt='',
                              desc='Sampling',
                              save_kv=False,
                              use_injection=True,
                              mask=mask,
                              cond_noised_latents_list=cond_noised_latents_list)

        return image

    def check_kv_empty(self):
        for name, module in self.unet.named_modules():
            if isinstance(module, Attention) and hasattr(module, 'save_kv'):
                assert module.save_kv.idx == 0, f'{name}\'s save_kv is not empty'

    @torch.no_grad()
    def __call__(self,
                 target_image: torch.Tensor,
                 texture_reference: torch.Tensor,
                 inversion_reference: torch.Tensor = None,
                 num_inference_steps=50,
                 mask=None):
        for _, module in self.unet.named_modules():
            if isinstance(module, Attention):
                if not hasattr(module, 'kv_saver'):
                    raise AttributeError('check if KV-Injection is registered')
        self.unet.kv_injection_agent.check_inference_steps(num_inference_steps)

        inversion_reference = target_image if inversion_reference is None else inversion_reference

        cond_noised_latents, cond_noised_latents_list = self.structure_preserving_invert(target_image, inversion_reference, num_inference_steps)

        self.check_kv_empty()
        image = self.fine_texture_sampling(cond_noised_latents, texture_reference, num_inference_steps,
                                           mask=mask)

        return image

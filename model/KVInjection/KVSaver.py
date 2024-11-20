import torch
import torch.nn as nn
from typing import Union, Optional
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from diffusers.models.attention_processor import Attention
import logging


REGISTER_BLOCK_NAMES = ['down_blocks', 'up_blocks', 'mid_block']


class KVInjection:
    def __init__(self, num_inference_steps):
        self.k = [None] * num_inference_steps
        self.v = [None] * num_inference_steps
        self.count = 0
        self.num_inference_steps = num_inference_steps

        return

    def append(self, k: torch.Tensor, v: torch.Tensor):
        self.k[self.count] = k
        self.v[self.count] = v
        self.count += 1

        if self.count == self.num_inference_steps:
            self.count = 0

        return

    def pop(self):
        k = self.k[self.count]
        v= self.v[self.count]
        self.count -= 1

        if self.count < 0:
            self.count = self.num_inference_steps - 1

        return k, v


def register_kv_injection(model: Union[StableDiffusionPipeline, UNet2DConditionModel], num_inference_steps: int, register_name=''):
    if isinstance(model, UNet2DConditionModel):
        unet = model
    else:
        unet = model.unet

    # dict to save register information
    register_dict = dict()
    count = 0

    def register_forward(attn: Attention, count=0):
        def forward(hidden_states: torch.Tensor,
                    encoder_hidden_states: Optional[torch.Tensor] = None,
                    attention_mask: Optional[torch.Tensor] = None,
                    **cross_attention_kwargs):
            self: Attention = attn
            assert hasattr(self, 'kv_injection'), 'seems kv_injection is not registered'
            # todo find ways to pass parameters to here
            save_kv = cross_attention_kwargs.pop('save_kv', True)
            use_injection = cross_attention_kwargs.pop('use_injection', False)

            residual = hidden_states
            in_dim = hidden_states.ndim
            if in_dim == 4:
                b, c, h, w = hidden_states.shape
                dim = h * w
                hidden_states = hidden_states.view(*hidden_states.shape[:2], -1).transpose(-1, -2)
            else:
                assert hidden_states.ndim == 3, 'check hidden_states\'s dim!'
                b, c, dim = hidden_states.shape

            if self.group_norm is not None:
                hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif attn.norm_cross:
                encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

            q = self.to_q(hidden_states)
            if use_injection:
                k, v = self.kv_injection.pop()
            elif encoder_hidden_states is None:
                # cross attention
                k = self.to_k(hidden_states)
                v = self.to_v(hidden_states)
            else:
                k = self.to_k(encoder_hidden_states)
                v = self.to_v(encoder_hidden_states)

            attention_mask = self.prepare_attention_mask(attention_mask, c // self.heads, b * self.heads)

            # todo check if is right
            k = self.head_to_batch_dim(k)
            q = self.head_to_batch_dim(q)
            v = self.head_to_batch_dim(v)
            # k = k.reshape(b * self.heads, c // self.heads, -1)
            # k = k.transpose(-1, -2)
            # q = q.reshape(b * self.heads, c // self.heads, -1)
            # attn_matric = torch.softmax(torch.matmul(k, q) / math.sqrt(dim), dim=-1)
            attn_score = self.get_attention_scores(q, k, attention_mask)
            out = torch.bmm(attn_score, v)
            out = self.batch_to_head_dim(out)
            out = self.to_out[0](out)
            out = self.to_out[1](out)

            if in_dim == 4:
                out = out.transpose(-1, -2).reshape(b, c, h, w)

            if self.residual_connection:
                out = out + residual

            out = out / self.rescale_output_factor

            if save_kv:
                self.kv_injection.append(k, v)
            return out

        # register new forward function and kv_injection
        # todo through global variable to be access from outside?
        attn.kv_injection = KVInjection(num_inference_steps)
        attn.forward = forward
        # Attention's child can't have attention, just return
        return count + 1

    for name, module in unet.named_modules():
        if isinstance(module, Attention):
            count = register_forward(module, count)
            register_dict[f'{register_name}.{name}'] = module

    # access from the outside
    unet.register_dict = register_dict
    return count


def reset_inference_steps(attn: nn.Module, count=0):
    for name, child in attn.named_children():
        if isinstance(child, Attention) and hasattr(child, 'inference_steps'):
            child.inference_steps = 0
            return count + 1
        else:
           count = reset_inference_steps(child, count)

    return count

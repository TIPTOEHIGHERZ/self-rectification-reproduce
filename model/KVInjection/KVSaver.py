import torch
import torch.nn as nn
from typing import Union, Optional
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from diffusers.models.attention_processor import Attention
import logging


REGISTER_BLOCK_NAMES = ['down_blocks', 'up_blocks', 'mid_block']


def batch_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int):
    # q [b1, c, h, w] k, v [b2, c, h, w]
    # perform attention on each batch of them
    b1, l, d = q.shape
    b2 = k.shape[0]

    q = q.reshape(b1, l, heads, d // heads)
    k = k.reshape(b2, l, heads, d // heads)
    v = v.reshape(b2, l, heads, d // heads)

    q = q.permute(2, 1, 0, 3).reshape(heads, l * b1, d // heads)
    k = k.permute(2, 1, 0, 3).reshape(heads, l * b2, d // heads)
    v = v.permute(2, 1, 0, 3).reshape(heads, l * b2, d // heads)

    attn_score = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / ((d // heads) ** 0.5), dim=-1)
    out = torch.matmul(attn_score, v)

    # reset dim
    out = out.reshape(heads, l, b1, d // heads).permute(2, 1, 0, 3).reshape(b1, l, d)

    return out


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
            print('full')
            self.count = 0

        return

    def pop(self):
        k = self.k[-self.count - 1]
        v = self.v[-self.count - 1]
        self.count += 1

        if self.count == self.num_inference_steps:
            print('clear')
            self.count = 0

        return k, v

    def __len__(self):
        return len(self.k)


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
            save_kv = save_kv and encoder_hidden_states is None
            use_injection = use_injection and encoder_hidden_states is None

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

            if encoder_hidden_states is not None and self.norm_cross:
                encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

            q = self.to_q(hidden_states)
            if use_injection:
                k, v = self.kv_injection.pop()
                k, v = k.repeat(b, *[1] * (k.ndim - 1)), v.repeat(b, *[1] * (v.ndim - 1))
            elif encoder_hidden_states is None:
                # cross attention
                k = self.to_k(hidden_states)
                v = self.to_v(hidden_states)
            else:
                k = self.to_k(encoder_hidden_states)
                v = self.to_v(encoder_hidden_states)

            if save_kv:
                self.kv_injection.append(k, v)
            attention_mask = self.prepare_attention_mask(attention_mask, c // self.heads, b * self.heads)

            # todo check if is right
            if use_injection:
                out = batch_attention(q, k, v, self.heads)
            else:
                k = self.head_to_batch_dim(k)
                q = self.head_to_batch_dim(q)
                v = self.head_to_batch_dim(v)
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

            return out

        # register new forward function and kv_injection
        # todo through global variable to be access from outside?
        attn.kv_injection = KVInjection(num_inference_steps)
        attn.forward = forward
        # Attention's child can't have attention, just return
        return count + 1

    for name, module in unet.named_modules():
        if isinstance(module, Attention):
            # TODO only register to attn2 not attn1
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

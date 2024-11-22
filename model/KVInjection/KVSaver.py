import torch
import torch.nn as nn
from typing import Union, Optional
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from diffusers.models.attention_processor import Attention
import logging


REGISTER_BLOCK_NAMES = ['down_blocks', 'up_blocks', 'mid_block']


def batch_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int):
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


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int):
    b, l1, d = q.shape
    l2 = k.shape[1]
    q = q.reshape(b, l1, heads, d // heads).permute(0, 2, 1, 3)
    k = k.reshape(b, l2, heads, d // heads).permute(0, 2, 1, 3)
    v = v.reshape(b, l2, heads, d // heads).permute(0, 2, 1, 3)

    attn_score = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / (d // heads) ** 0.5, dim=-1)
    out = torch.matmul(attn_score, v)
    out = out.permute(0, 2, 1, 3).reshape(b, l1, d)

    return out


class KVSaver:
    def __init__(self, num_inference_steps):
        self.k = [None] * num_inference_steps
        self.v = [None] * num_inference_steps
        self.idx = 0
        self.num_inference_steps = num_inference_steps

        return

    def append(self, k: torch.Tensor, v: torch.Tensor):
        if self.idx == self.num_inference_steps:
            raise IndexError(f'max saving size is {self.num_inference_steps}!')

        self.k[self.idx] = k.cpu()
        self.v[self.idx] = v.cpu()
        self.idx += 1

        return

    def pop(self):
        if self.idx == 0:
            raise IndexError('No element to pop!')

        k = self.k[self.idx - 1].cuda()
        v = self.v[self.idx - 1].cuda()
        self.idx -= 1

        return k, v

    def __len__(self):
        return self.idx


class KVInjectionAgent:
    def __init__(self, start_step: int, num_inference_steps: int, start_layer: int, activate_layer_num: int = None):
        self.start_step = start_step
        self.num_inference_steps = num_inference_steps
        self.curr_step = 0

        self.cur_layer = 0
        self.start_layer = start_layer
        self.activate_layer_num = activate_layer_num

        # will be set through register function
        self.total_layer = 0

        return

    def step(self, q, k, v, heads):
        self.cur_layer += 1
        return batch_attention(q, k, v, heads)

    def reset(self):
        self.cur_layer = 0
        self.curr_step = 0

    def __call__(self, q, k, v, heads, kv_saver: KVSaver, use_injection=False, save_kv=False):
        # load and save at the same time is not allowed!
        assert not (save_kv and use_injection), 'load and save at the same time is not allowed!'
        end_layer = self.total_layer if self.activate_layer_num is None else self.start_layer + self.activate_layer_num
        if end_layer > self.total_layer:
            logging.warning(f'activate layers too big, start_layer + activate_layer={end_layer} exceed total layers'
                            f'and was set to {self.total_layer}')
        reach_save_layer = ((self.curr_step >= self.start_step)
                            and (self.start_layer <= self.cur_layer < end_layer))
        reach_load_layer = (self.curr_step < self.num_inference_steps - self.start_step
                            and (self.start_layer <= self.cur_layer < end_layer))
        if reach_save_layer and save_kv:
            kv_saver.append(k, v)

        if reach_load_layer and use_injection:
            k, v = kv_saver.pop()
            out = batch_attention(q, k, v, heads)
        else:
            out = attention(q, k, v, heads)

        self.cur_layer += 1
        self.curr_step += self.cur_layer // self.total_layer
        self.cur_layer %= self.total_layer

        if self.curr_step == self.num_inference_steps:
            if use_injection:
                assert len(kv_saver) == 0, 'injection finished but kv_saver is not empty!'
            self.reset()

        return out

    def check_inference_steps(self, num_inference_steps: int):
        if self.num_inference_steps:
            return
        else:
            self.kv_saver = KVSaver(num_inference_steps)
            logging.warning(f'KV-Injection register length do not match, previous registered'
                            f'num_inference_steps: {self.num_inference_steps} was changed to {num_inference_steps}')
        return


def register_kv_injection(model: Union[StableDiffusionPipeline, UNet2DConditionModel],
                          kv_injection_agent: KVInjectionAgent,
                          num_inference_steps: int,
                          register_name=''):
    if isinstance(model, UNet2DConditionModel):
        unet = model
    else:
        unet = model.unet

    # dict to save register information
    register_dict = dict()
    register_count = dict()

    def register_forward(attn: Attention, count=0):
        def forward(hidden_states: torch.Tensor,
                    encoder_hidden_states: Optional[torch.Tensor] = None,
                    attention_mask: Optional[torch.Tensor] = None,
                    **cross_attention_kwargs):
            self: Attention = attn
            assert hasattr(self, 'kv_saver'), 'seems kv_injection is not registered'
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
            # if use_injection:
            #     k, v = self.kv_injection.pop()
            #     k, v = k.repeat(b, *[1] * (k.ndim - 1)), v.repeat(b, *[1] * (v.ndim - 1))
            if encoder_hidden_states is None:
                # cross attention
                k = self.to_k(hidden_states)
                v = self.to_v(hidden_states)
            else:
                k = self.to_k(encoder_hidden_states)
                v = self.to_v(encoder_hidden_states)

            # if save_kv:
            #     self.kv_injection.append(k, v)
            # attention_mask = self.prepare_attention_mask(attention_mask, c // self.heads, b * self.heads)

            # todo check if is right
            # if use_injection:
            #     out = batch_attention(q, k, v, self.heads)
            # else:
            #     k = self.head_to_batch_dim(k)
            #     q = self.head_to_batch_dim(q)
            #     v = self.head_to_batch_dim(v)
            #     attn_score = self.get_attention_scores(q, k, attention_mask)
            #     out = torch.bmm(attn_score, v)
            #     out = self.batch_to_head_dim(out)

            out = kv_injection_agent(q, k, v, self.heads, kv_saver=self.kv_saver,
                                     use_injection=use_injection, save_kv=save_kv)

            out = self.to_out[0](out)
            out - self.to_out[1](out)

            if in_dim == 4:
                out = out.transpose(-1, -2).reshape(b, c, h, w)

            if self.residual_connection:
                out = out + residual

            out = out / self.rescale_output_factor

            return out

        # register new forward function and kv_injection
        # todo through global variable to be access from outside?
        attn.kv_saver = KVSaver(num_inference_steps)
        attn.forward = forward
        # Attention's child can't have attention, just return
        return count + 1

    def register_blocks(block: nn.Module, n: str, count=0):
        for name, child in block.named_modules():
            if isinstance(child, Attention):
                # TODO only register to attn2 not attn1
                count = register_forward(child, count)
                register_dict[f'{register_name}.{n}.{name}'] = child

        return count

    for name, module in unet.named_children():
        if name in REGISTER_BLOCK_NAMES:
            register_count[name] = register_blocks(module, name)

    # access from the outside
    unet.register_dict = register_dict
    unet.register_count = register_count
    unet.kv_injection_agent = kv_injection_agent
    kv_injection_agent.total_layer = sum(register_count.values())
    return register_dict, register_count


def reset_inference_steps(attn: nn.Module, count=0):
    for name, child in attn.named_children():
        if isinstance(child, Attention) and hasattr(child, 'inference_steps'):
            child.inference_steps = 0
            return count + 1
        else:
           count = reset_inference_steps(child, count)

    return count

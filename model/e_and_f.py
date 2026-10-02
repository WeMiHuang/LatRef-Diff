from torch import nn
import torch as th
import torch.nn.functional as F
import numpy as np
import math
import math

import torch
from dataclasses import dataclass
from numbers import Number
from typing import NamedTuple, Tuple, Union

import numpy as np
import torch as th
from torch import nn
import torch.nn.functional as F
from choices import *
from config_base import BaseConfig
from .blocks import *
import math
import random
from inspect import isfunction
from einops import rearrange
from torch import nn, einsum

try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False

import os
_ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")

from .nn import (conv_nd, linear, normalization, timestep_embedding,
                 torch_checkpoint, zero_module)

class DownBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_dim, in_dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_dim, out_dim, 3, 1, 1)

        self.activ = nn.LeakyReLU(0.2, inplace=True)

        self.sc = nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=False)

    def forward(self, x):
        residual = F.avg_pool2d(self.sc(x), 2)
        out = self.conv2(self.activ(F.avg_pool2d(self.conv1(self.activ(x.clone())), 2)))
        out = residual + out
        return out / math.sqrt(2)

class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.linear = nn.Linear(in_dim, out_dim)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.linear(self.activ(x))


import torch
import torch.nn as nn

class AttributeBranch(nn.Module):
    def __init__(
        self,
        num_attributes,
        token_dim=512,
        model_dim=256,
        num_heads=4,
        num_layers=2,
        style_dim=256,
        w_dim=512,
        w_blocks=16
    ):
        super().__init__()
        self.token_dim = token_dim
        self.model_dim = model_dim
        self.token_count = 16
        self.total_tokens = 1 + self.token_count


        self.attribute_tokens = nn.Parameter(
            torch.randn(6, token_dim)
        )


        self.token_proj = nn.Linear(token_dim, model_dim)
        self.pos_emb = nn.Parameter(
            torch.randn(1, self.total_tokens, model_dim)
        )

        # ——— Transformer Encoder ———
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.style_head = nn.ModuleList([
            nn.Sequential(
                nn.Linear(model_dim, style_dim),
                nn.Dropout(0.1)
            )
            for _ in range(3)
        ])
        self.w_head = nn.ModuleList([nn.Linear(model_dim, w_dim * w_blocks)
            for _ in range(3)
        ])

    def forward(self, token_seq, tag,j):

        B = token_seq.size(0)


        attr_token = self.attribute_tokens[2*tag+j] \
                         .unsqueeze(0) \
                         .expand(B, 1, self.token_dim)  # (B,1,512)


        x = torch.cat([attr_token, token_seq], dim=1)  # (B, 17, 512)


        x = self.token_proj(x)                          # (B, 17, model_dim)
        x = x + self.pos_emb
        x = self.encoder(x)                             # (B, 17, model_dim)
        cls_token = x[:, 0]                             # (B, model_dim)


        style = self.style_head[tag](cls_token)              # (B, 256)
        w = self.w_head[tag](cls_token)                      # (B, 16*512)
        w = w.view(B, self.token_count, -1)             # (B, 16, 512)

        return style, w

class Extractors(nn.Module):
    def __init__(self, num_attributes=3):
        super().__init__()
        self.branches = AttributeBranch(num_attributes=num_attributes)

    def forward(self, x, tag,j):

        B = x.size(0)
        # (B, 512, 4, 4) → (B, 16, 512)
        token_seq = x.view(B, 512, -1).transpose(1, 2)

        style, w = self.branches(token_seq, tag,j)
        return style, w, (0, 0)


class Mapper(nn.Module):
    def __init__(
        self,
        num_attributes,
        seq_len=4,
        latent_dim=32,
        model_dim=256,
        num_heads=4,
        num_layers=3,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.model_dim = model_dim
        self.total_len = seq_len + 1  # +1 for image token

        # === z token embedding ===
        self.token_embed = nn.Linear(latent_dim, model_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, self.total_len, model_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            activation='gelu',
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        flat_dim = self.total_len * model_dim

        self.style_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(flat_dim, 256),
                #nn.LayerNorm(256),
                nn.Dropout(0.1)
            ) for _ in range(num_attributes)
        ])
        self.w_heads = nn.ModuleList([
            nn.Linear(flat_dim, 16 * 512)
            for _ in range(num_attributes)
        ])

    def forward(self, z, j, image):
        B, T, _ = z.shape
        assert T == self.seq_len


        z_tokens = self.token_embed(z)                      # (B, T, D)
        x = torch.cat([image, z_tokens], dim=1) + self.pos_emb

        x = self.transformer(x)
        x_flat = x.view(B, -1)


        style = self.style_heads[j](x_flat)
        w = self.w_heads[j](x_flat)


        return style, w.view(B, 16, 512)


class ImageE(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),  # 128x128
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),  # 64x64
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),  # 32x32
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),  # (B, 256)
            nn.Linear(256, 256)
        )


    def forward(self, image):
        assert image.shape[1:] == (3, 256, 256)

        img_token = self.image_encoder(image).unsqueeze(1)  # (B, 1, D)



        return img_token







def exists(val):
    return val is not None

def Normalize(in_channels):
    return th.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """
    def __init__(self, in_channels, n_heads=4, d_head=128,
                 depth=1, dropout=0., context_dim=512,
                 disable_self_attn=False, use_linear=False,
                 use_checkpoint=True):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels,
                                     inner_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim[d],
                                   disable_self_attn=disable_self_attn, checkpoint=use_checkpoint)
                for d in range(depth)]
        )
        if not use_linear:
            self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                                  in_channels,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
        self.use_linear = use_linear

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context[i])
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in

class MemoryEfficientCrossAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        print(f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
              f"{heads} heads.")
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        self.attention_op: Optional[Any] = None

    def forward(self, x, context=None, mask=None):
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # force cast to fp32 to avoid overflowing
        if _ATTN_PRECISION == "fp32":
            with th.autocast(enabled=False, device_type='cuda'):
                q, k = q.float(), k.float()
                sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        else:
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        del q, k

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -th.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)  #10*64*512,64=8*8

class CrossAttention3(nn.Module):
    def __init__(self, query_dim, context_dim, heads=1, dim_head=64, dropout=0.):
        super().__init__()
        #inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = query_dim ** -0.5
        self.heads = heads
        self.norm = Normalize(query_dim)

        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k = nn.Linear(context_dim, query_dim, bias=False)
        self.to_v = nn.Linear(context_dim, query_dim, bias=False)


    def forward(self, x, context=None, mask=None):
        #context,alpha=context
        h = self.heads
        x = self.norm(x)

        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()

        q = self.to_q(x)
        context=context.unsqueeze(1)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # force cast to fp32 to avoid overflowing
        if _ATTN_PRECISION == "fp32":
            with th.autocast(enabled=False, device_type='cuda'):
                q, k = q.float(), k.float()
                sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        else:
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        del q, k

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -th.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        out=out.permute(0, 2, 1)
        return out#,alpha) ##10*64*512,64=8*8

# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)

class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)

def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)

class CheckpointFunction(th.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        ctx.gpu_autocast_kwargs = {"enabled": th.is_autocast_enabled(),
                                   "dtype": th.get_autocast_gpu_dtype(),
                                   "cache_enabled": th.is_autocast_cache_enabled()}
        with th.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with th.enable_grad(), \
                th.cuda.amp.autocast(**ctx.gpu_autocast_kwargs):
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = th.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads

class BasicTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention
    }
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
                 disable_self_attn=False):
        super().__init__()
        attn_mode = "softmax-xformers" if XFORMERS_IS_AVAILBLE else "softmax"
        assert attn_mode in self.ATTENTION_MODES
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                              context_dim=context_dim if self.disable_self_attn else None)  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(query_dim=dim, context_dim=context_dim,
                              heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None):
        x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x
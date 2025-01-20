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
from .generator import  Generator
import math
import random
from inspect import isfunction
from einops import rearrange
from torch import nn, einsum
from .discriminator import Decoder
#from .e_and_f import Extractors

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

#AttentionBlock(512,use_checkpoint=False,num_heads=1,num_head_channels=-1,use_new_attention_order=False)

class Extractors_(nn.Module):   #风格提取器
    def __init__(self, ):
        super().__init__()
        self.num_tags = 3#len(hyperparameters.tags)   #3
        channels = [32, 64, 128, 256, 256, 512, 512]
        self.out=nn.Conv2d(channels[-1], 512 * (self.num_tags), 1, 1, 0)
        self.model = nn.Sequential(
            nn.Conv2d(3, channels[0], 1, 1, 0),
            *[DownBlock(channels[i], channels[i + 1]) for i in range(len(channels) - 1)],                                                             #8*512*1*1   #8*（256*3）
        )  #输出的向量的长度为风格特征维度*tag数量

        self.AdainResBlk2 = SpatialTransformer(512)
        self.token = nn.ParameterList([nn.Parameter(torch.zeros(1, 10, 512)) for i in range(7)])

        self.out = nn.ModuleList([nn.Conv2d(channels[-1], 512, 1, 1, 0) for i in range(7)])
    def forward(self, x,tag,tag_j_trg):  #8*3*256*256
        s_bg = self.model(x)

        s=self.AdainResBlk2(s_bg,self.token[tag*2+tag_j_trg].repeat(x.size(0),1,1))

        s=F.adaptive_avg_pool2d(s,1)


        s=self.out[tag](s).view(x.size(0), -1)   #8*3*256
        return s#,s_bg.squeeze(-1).squeeze(-1)#[:, i]  #8*256选第i个tag的风格特征返回

class Extractors__(nn.Module):   #风格提取器
    def __init__(self, ):
        super().__init__()
        self.num_tags = 3#len(hyperparameters.tags)   #3
        channels = [32, 64, 128, 256, 256, 512, 512]
        self.out=nn.Conv2d(channels[-1], 512 * (self.num_tags), 1, 1, 0)
        self.out_co = nn.Conv2d(128, 512 * (self.num_tags), 1, 1, 0)
        self.out_mid = nn.Conv2d(256, 512 * (self.num_tags), 1, 1, 0)
        self.model = nn.ModuleList([
            nn.Conv2d(3, channels[0], 1, 1, 0),
            *[DownBlock(channels[i], channels[i + 1]) for i in range(len(channels) - 1)],
            nn.AdaptiveAvgPool2d(1),]                                                             #8*512*1*1   #8*（256*3）
        )  #输出的向量的长度为风格特征维度*tag数量

    def forward(self, x,tag):  #8*3*256*256
        out=[]
        co=self.model[2](self.model[1](self.model[0](x)))
        co_s=self.out_co(F.adaptive_avg_pool2d(co,1)).view(x.size(0), self.num_tags, -1)[:,tag]
        out.append(co_s)



        mid=self.model[4](self.model[3](co))
        mid_s = self.out_mid(F.adaptive_avg_pool2d(mid, 1)).view(x.size(0), self.num_tags, -1)[:, tag]
        out.append(mid_s)

        s_bg=self.model[6](self.model[5](mid))
        s_bg = self.model[7](s_bg)
        s=self.out(s_bg).view(x.size(0), self.num_tags, -1)[:,tag]  #8*3*256
        out.append(s)

        return out#.squeeze(-1).squeeze(-1)#[:, i]  #8*256选第i个tag的风格特征返回

class Extractors(nn.Module):   #风格提取器
    def __init__(self, ):
        super().__init__()
        style_dim=512
        #self.out=nn.Linear(512, 512 * 3)
        #self.out2=nn.ModuleList([nn.Linear(512, 512 * 4),nn.Linear(512, 512 * 4),nn.Linear(512, 512 * 4)])
        self.out = nn.ModuleList([nn.Sequential(nn.Linear(style_dim, style_dim)
                                                , nn.ReLU()
                                                , nn.Linear(style_dim, style_dim)) for i in range(3)])

        self.out2 = nn.ModuleList([nn.Sequential(nn.Linear(style_dim, style_dim)
                                                 , nn.ReLU()
                                                 , nn.Linear(style_dim, style_dim * 4)) for i in range(3)])

    def forward(self, x,tag):  #8*3*256*256



        #s=self.out(x).view(x.size(0), 3, -1)[:,tag]    #8*3*256
        s = self.out[tag](x)  # 8*256
        cond = self.out2[tag](x).view(x.size(0), 4, 512)
        return s,cond#,s_bg.squeeze(-1).squeeze(-1)#[:, i]  #8*256选第i个tag的风格特征返回

class Mapper(nn.Module):
    def __init__(self, num_attributes): #[256,256,256]
        super().__init__()
        channels = [256, 256, 512]
        self.pre_model = nn.Sequential(
            nn.Linear(32, channels[0]),
            *[LinearBlock(channels[i], channels[i + 1]) for i in range(len(channels) - 1)]
        )

        channels = [512, 512, 512]
        self.post_models = nn.ModuleList([nn.Sequential(
            *[LinearBlock(channels[i], channels[i + 1]) for i in range(len(channels) - 1)],
            nn.Linear(channels[-1], 512*5),
            ) for i in range(num_attributes)
        ])

    def forward(self, z, j):
        z = self.pre_model(z)  #8*256
        z = self.post_models[j](z)
        return z[:,:512],z[:,512:].view(z.size(0),4,512)  #8*256

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
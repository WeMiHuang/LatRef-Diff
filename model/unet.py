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
from .e_and_f import Extractors,ImageE

from torchvision.utils import make_grid, save_image

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

class InstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

        # weight and bias are dynamically assigned
        self.weight = nn.Parameter(th.ones(1, num_features, 1))
        self.bias = nn.Parameter(th.zeros(1, num_features, 1))

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, -1)
        bias_in = x.mean(-1, keepdim=True)
        weight_in = x.std(-1, keepdim=True)

        out = (x - bias_in) / (weight_in + self.eps) * self.weight + self.bias
        return out.view(N, C, H, W)

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'

class UpBlockIN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1)

        self.in1 = InstanceNorm2d(in_dim)
        self.in2 = InstanceNorm2d(out_dim)

        self.activ = nn.LeakyReLU(0.2, inplace=True)

        self.sc = nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=False)

    def forward(self, x):
        residual = F.interpolate(self.sc(x), scale_factor=2, mode='nearest')
        out = self.conv2(self.activ(self.in2(self.conv1(F.interpolate(self.activ(self.in1(x)), scale_factor=2, mode='nearest')))))
        out = residual + out
        return out / math.sqrt(2)

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

@dataclass
class BeatGANsUNetConfig(BaseConfig):
    image_size: int = 64
    in_channels: int = 3
    # base channels, will be multiplied
    model_channels: int = 64
    # output of the unet
    # suggest: 3
    # you only need 6 if you also model the variance of the noise prediction (usually we use an analytical variance hence 3)
    out_channels: int = 3
    # how many repeating resblocks per resolution
    # the decoding side would have "one more" resblock
    # default: 2
    num_res_blocks: int = 2
    # you can also set the number of resblocks specifically for the input blocks
    # default: None = above
    num_input_res_blocks: int = None
    # number of time embed channels and style channels
    embed_channels: int = 512
    # at what resolutions you want to do self-attention of the feature maps
    # attentions generally improve performance
    # default: [16]
    # beatgans: [32, 16, 8]
    attention_resolutions: Tuple[int] = (16, )
    # number of time embed channels
    time_embed_channels: int = None
    # dropout applies to the resblocks (on feature maps)
    dropout: float = 0.1
    channel_mult: Tuple[int] = (1, 2, 4, 8)
    input_channel_mult: Tuple[int] = None
    conv_resample: bool = True
    # always 2 = 2d conv
    dims: int = 2
    # don't use this, legacy from BeatGANs
    num_classes: int = None
    use_checkpoint: bool = False
    # number of attention heads
    num_heads: int = 1
    # or specify the number of channels per attention head
    num_head_channels: int = -1
    # what's this?
    num_heads_upsample: int = -1
    # use resblock for upscale/downscale blocks (expensive)
    # default: True (BeatGANs)
    resblock_updown: bool = True
    # never tried
    use_new_attention_order: bool = False
    resnet_two_cond: bool = False
    resnet_cond_channels: int = None
    # init the decoding conv layers with zero weights, this speeds up training
    # default: True (BeattGANs)
    resnet_use_zero_module: bool = True
    # gradient checkpoint the attention operation
    attn_checkpoint: bool = False

    def make_model(self):
        return BeatGANsUNetModel(self)



class BeatGANsUNetModel(nn.Module):
    def __init__(self, conf: BeatGANsUNetConfig):
        super().__init__()
        self.conf = conf

        if conf.num_heads_upsample == -1:
            self.num_heads_upsample = conf.num_heads

        self.dtype = th.float32

        self.time_emb_channels = conf.time_embed_channels or conf.model_channels
        self.time_embed = nn.Sequential(
            linear(self.time_emb_channels, conf.embed_channels),
            nn.SiLU(),
            linear(conf.embed_channels, conf.embed_channels),
        )

        if conf.num_classes is not None:
            self.label_emb = nn.Embedding(conf.num_classes,
                                          conf.embed_channels)

        ch = input_ch = int(conf.channel_mult[0] * conf.model_channels)
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(
                conv_nd(conf.dims, conf.in_channels, ch, 3, padding=1))
        ])

        kwargs = dict(
            use_condition=True,
            two_cond=conf.resnet_two_cond,
            use_zero_module=conf.resnet_use_zero_module,
            # style channels for the resnet block
            cond_emb_channels=conf.resnet_cond_channels,
        )

        self._feature_size = ch

        # input_block_chans = [ch]
        input_block_chans = [[] for _ in range(len(conf.channel_mult))]
        input_block_chans[0].append(ch)

        # number of blocks at each resolution
        self.input_num_blocks = [0 for _ in range(len(conf.channel_mult))]
        self.input_num_blocks[0] = 1
        self.output_num_blocks = [0 for _ in range(len(conf.channel_mult))]

        ds = 1
        resolution = conf.image_size
        for level, mult in enumerate(conf.input_channel_mult
                                     or conf.channel_mult):
            for _ in range(conf.num_input_res_blocks or conf.num_res_blocks):
                layers = [
                    ResBlockConfig(
                        ch,
                        conf.embed_channels,
                        conf.dropout,
                        out_channels=int(mult * conf.model_channels),
                        dims=conf.dims,
                        use_checkpoint=conf.use_checkpoint,
                        **kwargs,
                    ).make_model()
                ]
                ch = int(mult * conf.model_channels)
                if resolution in conf.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=conf.use_checkpoint
                            or conf.attn_checkpoint,
                            num_heads=conf.num_heads,
                            num_head_channels=conf.num_head_channels,
                            use_new_attention_order=conf.
                            use_new_attention_order,
                        ))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                # input_block_chans.append(ch)
                input_block_chans[level].append(ch)
                self.input_num_blocks[level] += 1
                # print(input_block_chans)
            if level != len(conf.channel_mult) - 1:
                resolution //= 2
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlockConfig(
                            ch,
                            conf.embed_channels,
                            conf.dropout,
                            out_channels=out_ch,
                            dims=conf.dims,
                            use_checkpoint=conf.use_checkpoint,
                            down=True,
                            **kwargs,
                        ).make_model() if conf.
                        resblock_updown else Downsample(ch,
                                                        conf.conv_resample,
                                                        dims=conf.dims,
                                                        out_channels=out_ch)))
                ch = out_ch
                # input_block_chans.append(ch)
                input_block_chans[level + 1].append(ch)
                self.input_num_blocks[level + 1] += 1
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlockConfig(
                ch,
                conf.embed_channels,
                conf.dropout,
                dims=conf.dims,
                use_checkpoint=conf.use_checkpoint,
                **kwargs,
            ).make_model(),
            AttentionBlock(
                ch,
                use_checkpoint=conf.use_checkpoint or conf.attn_checkpoint,
                num_heads=conf.num_heads,
                num_head_channels=conf.num_head_channels,
                use_new_attention_order=conf.use_new_attention_order,
            ),
            ResBlockConfig(
                ch,
                conf.embed_channels,
                conf.dropout,
                dims=conf.dims,
                use_checkpoint=conf.use_checkpoint,
                **kwargs,
            ).make_model(),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(conf.channel_mult))[::-1]:
            for i in range(conf.num_res_blocks + 1):
                # print(input_block_chans)
                # ich = input_block_chans.pop()
                try:
                    ich = input_block_chans[level].pop()
                except IndexError:
                    # this happens only when num_res_block > num_enc_res_block
                    # we will not have enough lateral (skip) connecions for all decoder blocks
                    ich = 0
                # print('pop:', ich)
                layers = [
                    ResBlockConfig(
                        # only direct channels when gated
                        channels=ch + ich,
                        emb_channels=conf.embed_channels,
                        dropout=conf.dropout,
                        out_channels=int(conf.model_channels * mult),
                        dims=conf.dims,
                        use_checkpoint=conf.use_checkpoint,
                        # lateral channels are described here when gated
                        has_lateral=True if ich > 0 else False,
                        lateral_channels=None,
                        **kwargs,
                    ).make_model()
                ]
                ch = int(conf.model_channels * mult)
                if resolution in conf.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=conf.use_checkpoint
                            or conf.attn_checkpoint,
                            num_heads=self.num_heads_upsample,
                            num_head_channels=conf.num_head_channels,
                            use_new_attention_order=conf.
                            use_new_attention_order,
                        ))
                if level and i == conf.num_res_blocks:
                    resolution *= 2
                    out_ch = ch
                    layers.append(
                        ResBlockConfig(
                            ch,
                            conf.embed_channels,
                            conf.dropout,
                            out_channels=out_ch,
                            dims=conf.dims,
                            use_checkpoint=conf.use_checkpoint,
                            up=True,
                            **kwargs,
                        ).make_model() if (
                            conf.resblock_updown
                        ) else Upsample(ch,
                                        conf.conv_resample,
                                        dims=conf.dims,
                                        out_channels=out_ch))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self.output_num_blocks[level] += 1
                self._feature_size += ch

        # print(input_block_chans)
        # print('inputs:', self.input_num_blocks)
        # print('outputs:', self.output_num_blocks)

        if conf.resnet_use_zero_module:
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                zero_module(
                    conv_nd(conf.dims,
                            input_ch,
                            conf.out_channels,
                            3,
                            padding=1)),
            )
        else:
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                conv_nd(conf.dims, input_ch, conf.out_channels, 3, padding=1),
            )

    def forward(self, x, t, y=None, **kwargs):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.conf.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        # hs = []
        hs = [[] for _ in range(len(self.conf.channel_mult))]
        emb = self.time_embed(timestep_embedding(t, self.time_emb_channels))

        if self.conf.num_classes is not None:
            raise NotImplementedError()
            # assert y.shape == (x.shape[0], )
            # emb = emb + self.label_emb(y)

        # new code supports input_num_blocks != output_num_blocks
        h = x.type(self.dtype)
        k = 0
        for i in range(len(self.input_num_blocks)):
            for j in range(self.input_num_blocks[i]):
                h = self.input_blocks[k](h, emb=emb)
                # print(i, j, h.shape)
                hs[i].append(h)
                k += 1
        assert k == len(self.input_blocks)

        h = self.middle_block(h, emb=emb)
        k = 0
        for i in range(len(self.output_num_blocks)):
            for j in range(self.output_num_blocks[i]):
                # take the lateral connection from the same layer (in reserve)
                # until there is no more, use None
                try:
                    lateral = hs[-i - 1].pop()
                    # print(i, j, lateral.shape)
                except IndexError:
                    lateral = None
                    # print(i, j, lateral)
                h = self.output_blocks[k](h, emb=emb, lateral=lateral)
                k += 1

        h = h.type(x.dtype)
        pred = self.out(h)
        return Return(pred=pred)


class BeatGANsDecoderModel(nn.Module):
    def __init__(self, conf: BeatGANsUNetConfig):
        super().__init__()
        self.conf = conf

        if conf.num_heads_upsample == -1:
            self.num_heads_upsample = conf.num_heads

        self.dtype = th.float32

        channel_mult=(1,1,2,2,3,4)#(1,2,3,4)#

        ch = input_ch = int(channel_mult[0] * conf.model_channels)

        kwargs = dict(
            use_condition=False,
        )


        self._feature_size = ch

        # input_block_chans = [ch]
        input_block_chans = [[] for _ in range(len(channel_mult))]
        input_block_chans[0].append(ch)

        # number of blocks at each resolution
        self.output_num_blocks = [0 for _ in range(len(channel_mult))]

        ds = 1
        resolution = conf.image_size

        ch=512
        ich = 384

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(conf.num_res_blocks + 1):
                if mult==3:
                    layers = [
                        ResBlockConfig(
                            # only direct channels when gated
                            channels=ch + ich,
                            emb_channels=conf.embed_channels,
                            dropout=conf.dropout,
                            out_channels=int(conf.model_channels * mult),
                            dims=conf.dims,
                            use_checkpoint=conf.use_checkpoint,
                            # lateral channels are described here when gated
                            has_lateral=True if ich>0 else False,
                            lateral_channels=None,
                            **kwargs,
                        ).make_model()
                    ]
                    ich=0
                else:
                    layers = [
                        ResBlockConfig(
                            # only direct channels when gated
                            channels=ch,
                            emb_channels=conf.embed_channels,
                            dropout=conf.dropout,
                            out_channels=int(conf.model_channels * mult),
                            dims=conf.dims,
                            use_checkpoint=conf.use_checkpoint,
                            # lateral channels are described here when gated
                            has_lateral=False,
                            lateral_channels=None,
                            **kwargs,
                        ).make_model()
                    ]
                ch = int(conf.model_channels * mult)
                if resolution in conf.attention_resolutions:
                    layers.append(
                        AttentionBlock2(
                            ch,
                            use_checkpoint=conf.use_checkpoint
                            or conf.attn_checkpoint,
                            num_heads=self.num_heads_upsample,
                            num_head_channels=conf.num_head_channels,
                            use_new_attention_order=conf.
                            use_new_attention_order,
                        ))
                if level and i == conf.num_res_blocks:
                    resolution *= 2
                    out_ch = ch
                    layers.append(
                        ResBlockConfig(
                            ch,
                            conf.embed_channels,
                            conf.dropout,
                            out_channels=out_ch,
                            dims=conf.dims,
                            use_checkpoint=conf.use_checkpoint,
                            up=True,
                            **kwargs,
                        ).make_model() if (
                            conf.resblock_updown
                        ) else Upsample(ch,
                                        conf.conv_resample,
                                        dims=conf.dims,
                                        out_channels=out_ch))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self.output_num_blocks[level] += 1
                self._feature_size += ch

        # print(input_block_chans)
        # print('inputs:', self.input_num_blocks)
        # print('outputs:', self.output_num_blocks)


        self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                conv_nd(conf.dims, input_ch, 19, 3, padding=1),
            )

    def forward(self, x, y=None, **kwargs):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.conf.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        # hs = []
        hs = [[] for _ in range(len(self.conf.channel_mult))]



        # new code supports input_num_blocks != output_num_blocks
        #x.pop()
        h = x[2]
        #x.pop()
        k = 0
        for i in range(len(self.output_num_blocks)):
            for j in range(self.output_num_blocks[i]):
                # take the lateral connection from the same layer (in reserve)
                # until there is no more, use None
                #lateral = None
                if i==1 and j==0:
                    h=self.output_blocks[k](h,lateral=x[0])
                    # print(i, j, lateral)
                else:
                    h = self.output_blocks[k](h)
                k += 1

        #h = h.type(x.dtype)
        pred = self.out(h)
        return pred


class Return(NamedTuple):
    pred: th.Tensor


@dataclass
class BeatGANsEncoderConfig(BaseConfig):
    image_size: int
    in_channels: int
    model_channels: int
    out_hid_channels: int
    out_channels: int
    num_res_blocks: int
    attention_resolutions: Tuple[int]
    dropout: float = 0
    channel_mult: Tuple[int] = (1, 2, 4, 8)
    use_time_condition: bool = True
    conv_resample: bool = True
    dims: int = 2
    use_checkpoint: bool = False
    num_heads: int = 1
    num_head_channels: int = -1
    resblock_updown: bool = False
    use_new_attention_order: bool = False
    pool: str = 'adaptivenonzero'

    def make_model(self):
        return BeatGANsEncoderModel(self)

        #self.resize1 = nn.AvgPool2d(4, padding=1, stride=1)
        #self.resize2 = Resize(64)

    #from torchvision.transforms import Resize
class BeatGANsEncoderModel(nn.Module):
    """
    The half UNet model with attention and timestep embedding.

    For usage, see UNet.
    """
    def __init__(self, conf: BeatGANsEncoderConfig):
        super().__init__()
        self.conf = conf
        self.dtype = th.float32

        if conf.use_time_condition:
            time_embed_dim = conf.model_channels * 4
            self.time_embed = nn.Sequential(
                linear(conf.model_channels, time_embed_dim),
                nn.SiLU(),
                linear(time_embed_dim, time_embed_dim),
            )
        else:
            time_embed_dim = None

        ch = int(conf.channel_mult[0] * conf.model_channels)
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(
                conv_nd(conf.dims, conf.in_channels, ch, 3, padding=1))
        ])
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        resolution = conf.image_size
        for level, mult in enumerate(conf.channel_mult):
            for _ in range(conf.num_res_blocks):
                layers = [
                    ResBlockConfig(
                        ch,
                        time_embed_dim,
                        conf.dropout,
                        out_channels=int(mult * conf.model_channels),
                        dims=conf.dims,
                        use_condition=conf.use_time_condition,
                        use_checkpoint=conf.use_checkpoint,
                    ).make_model()
                ]
                ch = int(mult * conf.model_channels)
                if resolution in conf.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=conf.use_checkpoint,
                            num_heads=conf.num_heads,
                            num_head_channels=conf.num_head_channels,
                            use_new_attention_order=conf.
                            use_new_attention_order,
                        ))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(conf.channel_mult) - 1:
                resolution //= 2
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlockConfig(
                            ch,
                            time_embed_dim,
                            conf.dropout,
                            out_channels=out_ch,
                            dims=conf.dims,
                            use_condition=conf.use_time_condition,
                            use_checkpoint=conf.use_checkpoint,
                            down=True,
                        ).make_model() if (
                            conf.resblock_updown
                        ) else Downsample(ch,
                                          conf.conv_resample,
                                          dims=conf.dims,
                                          out_channels=out_ch)))
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlockConfig(
                ch,
                time_embed_dim,
                conf.dropout,
                dims=conf.dims,
                use_condition=conf.use_time_condition,
                use_checkpoint=conf.use_checkpoint,
            ).make_model(),
            AttentionBlock(
                ch,
                use_checkpoint=conf.use_checkpoint,
                num_heads=conf.num_heads,
                num_head_channels=conf.num_head_channels,
                use_new_attention_order=conf.use_new_attention_order,
            ),
            ResBlockConfig(
                ch,
                time_embed_dim,
                conf.dropout,
                dims=conf.dims,
                use_condition=conf.use_time_condition,
                use_checkpoint=conf.use_checkpoint,
            ).make_model(),
        )
        self._feature_size += ch
        if conf.pool == "adaptivenonzero":
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                conv_nd(conf.dims, ch, conf.out_channels, 1),
                nn.Flatten(),
            )
        else:
            raise NotImplementedError(f"Unexpected {conf.pool} pooling")


    def forward(self, x, t=None, return_2d_feature=False, super=None, selected=None,latent=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x K] Tensor of outputs.
        """
        f = 4
        ff = 4
        if super != None:
            fusion, cond_s, tag = super

        if self.conf.use_time_condition:
            emb = self.time_embed(timestep_embedding(t, self.model_channels))
        else:
            emb = None  # yes

        results = []
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb=emb)
            if self.conf.pool.startswith("spatial"):
                results.append(h.type(x.dtype).mean(dim=(2, 3)))

            if super != None and h.size(3) == f:
                f = 0

                if h.size(3) == 4:
                    if latent==None:
                        cond_s = fusion.transform(cond_s,tag[0],tag[1])  # 20*4*512
                        cond_s,loss=(cond_s[0],cond_s[1]),cond_s[2]
                    h = fusion(cond_s, tag, h, selected)

            if h.size(3)==ff:
                ff=ff/2
                results.append(h)

        h = self.middle_block(h, emb=emb)
        if self.conf.pool.startswith("spatial"):
            results.append(h.type(x.dtype).mean(dim=(2, 3)))
            h = th.cat(results, axis=-1)
        else:
            h = h.type(x.dtype)

        h_2d = h  # 8*512*4*4
        results.append(h)
        h = self.out(h)  # 8*512
        results.append(h)

        if return_2d_feature:
            return h, results
        else:
            return h

    def forward_flatten(self, x):
        """
        transform the last 2d feature into a flatten vector
        """
        h = self.out(x)
        return h

class SuperNet(nn.Module):
    """
    The half UNet model with attention and timestep embedding.

    For usage, see UNet.
    """

    def __init__(self, tag_number):
        super().__init__()

        self.AdaINNets = nn.ModuleList()
        style_dim=512
        self.AdaINNets.append(STNet(style_dim,tag_number))
        self.transform = Extractors()
        self.image = ImageE()




    def forward(self, ref, tag, d2_feature,selected,h_t=None):

        s=self.AdaINNets[0](ref, tag, d2_feature,selected,h_t)
        return s

class KeepBlockIN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_dim, in_dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_dim, out_dim, 3, 1, 1)

        # use nn.InstanceNorm2d(in_dim, affine=True) if you want.
        self.in1 = InstanceNorm2d(in_dim)
        self.in2 = InstanceNorm2d(in_dim)

        self.activ = nn.LeakyReLU(0.2, inplace=True)

        self.sc = nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=False)
    def forward(self, x):
        residual = self.sc(x)
        out = self.conv2(self.activ(self.in2(self.conv1(self.activ(self.in1(x.clone()))))))
        out = residual + out
        return out / math.sqrt(2)

class DownBlockIN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_dim, in_dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_dim, out_dim, 3, 1, 1)

        # use nn.InstanceNorm2d(in_dim, affine=True) if you want.
        self.in1 = InstanceNorm2d(in_dim)
        self.in2 = InstanceNorm2d(in_dim)

        self.activ = nn.LeakyReLU(0.2, inplace=True)

        self.sc = nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=False)

    def forward(self, x):
        residual = F.avg_pool2d(self.sc(x), 2)
        out = self.conv2(self.activ(self.in2(F.avg_pool2d(self.conv1(self.activ(self.in1(x.clone()))), 2))))
        out = residual + out
        return out / math.sqrt(2)

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class MultiScaleMask(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.branch_dilated = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.weight = nn.Parameter(torch.ones(3))

    def forward(self, x):
        m1 = self.branch1(x)
        m2 = self.branch3(x)
        m3 = self.branch_dilated(x)
        w = F.softmax(self.weight, dim=0)
        return w[0] * m1 + w[1] * m2 + w[2] * m3


class ChannelGroupedMask(nn.Module):
    def __init__(self, channels, num_groups=4):
        super().__init__()
        group_size = channels // num_groups
        self.groups = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(group_size, group_size, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(group_size, group_size, 1),
                nn.Sigmoid()
            ) for _ in range(num_groups)
        ])
        self.num_groups = num_groups
        self.group_size = group_size

    def forward(self, x):
        split = torch.split(x, self.group_size, dim=1)
        out = [self.groups[i](split[i]) for i in range(self.num_groups)]
        return torch.cat(out, dim=1)

class Translator(nn.Module):
    def __init__(self,style_dim,tag_number):
        super().__init__()
        self.pre_cov=nn.Conv2d(style_dim, style_dim, 1, 1, 0)
        self.model = nn.ModuleList([
            MiddleBlock(style_dim, style_dim, tag_number) for _ in range(7)
        ])

        self.ST = SpatialTransformer(style_dim)
        self.style_to_params = nn.Linear(256,self.get_num_adain_params(self.model))
        self.masks = nn.ModuleList([MultiScaleMask(style_dim) for _ in range(2)])
        self.features = nn.Conv2d(style_dim, style_dim, 1, 1, 0)

    def forward(self, e, s,tag_j_trg):

        p = self.style_to_params(s[0])
        self.assign_adain_params(p, self.model)
        mid = self.pre_cov(e)

        for block in self.model:
            mid = block(mid, tag_j_trg)
        mid = self.ST(mid,s[1])
        f = self.features(mid)
        m = self.masks[tag_j_trg](mid)

        return f * m + e * (1 - m)

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ in ["AdaptiveInstanceNorm2d"]:
                m.bias = adain_params[:, :m.num_features].contiguous().view(-1, m.num_features, 1)
                m.weight = adain_params[:, m.num_features:2 * m.num_features].contiguous().view(-1, m.num_features, 1) + 1
                if adain_params.size(1) > 2 * m.num_features:
                    adain_params = adain_params[:, 2 * m.num_features:]

    def get_num_adain_params(self, model):
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ in ["AdaptiveInstanceNorm2d"]:
                num_adain_params += 2 * m.num_features
        return num_adain_params

class STNet(nn.Module):
    """
    The half UNet model with attention and timestep embedding.

    For usage, see UNet.
    """
    def __init__(self, style_dim,tag_number):
        super().__init__()


        self.translators = nn.ModuleList([Translator(512,len(tag_number.tags[i]['attributes']))#
                                      for i in range(len(tag_number.tags))]) #

        channels = [style_dim, style_dim]
        self.encoder = nn.Sequential(
            nn.Conv2d(style_dim, style_dim, 1, 1, 0),
            *[KeepBlockIN(channels[i], channels[i + 1]) for i in range(len(channels) - 1)]
        )
        channels = [style_dim, style_dim]
        self.decoder = nn.Sequential(
            *[KeepBlockIN(channels[i], channels[i + 1]) for i in range(len(channels) - 1)],
            nn.Conv2d(style_dim, style_dim, 1, 1, 0)
        )



    def ema(self,source, target, decay):
        source_dict = source.state_dict()
        target_dict = target.state_dict()
        for key in source_dict.keys():
            target_dict[key].data.copy_(target_dict[key].data * decay +
                                        source_dict[key].data * (1 - decay))




    def forward(self, ref,tag,d2_feature_,selected,h_t):
        if selected != None:
            tag, tag_j_trg, tag_j = tag

        else:

            tag, tag_j_trg = tag

        b,c,h,w=d2_feature_.shape
        d2_feature=self.encoder(d2_feature_)
        f_t = self.translators[tag](d2_feature, ref,tag_j_trg)
        s = self.decoder(f_t)

        return s



class SuperResModel(BeatGANsUNetModel):
    """
    A UNetModel that performs super-resolution.

    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """
    def __init__(self, image_size, in_channels, *args, **kwargs):
        super().__init__(image_size, in_channels * 2, *args, **kwargs)

    def forward(self, x, timesteps, low_res=None, **kwargs):
        _, _, new_height, new_width = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width),
                                  mode="bilinear")
        x = th.cat([x, upsampled], dim=1)
        return super().forward(x, timesteps, **kwargs)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // reduction, in_planes, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, in_planes, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out

class SwiGLU2d(nn.Module):
    def __init__(self, c_in, c_hidden):
        super().__init__()
        self.proj = nn.Conv2d(c_in, 2 * c_hidden, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.proj(x)
        a, b = torch.chunk(x, 2, dim=1)
        return a * F.silu(b)

class MiddleBlock(nn.Module):
    def __init__(self, in_dim, out_dim,tag_number):
        super().__init__()

        self.conv1 = nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1)

        self.adain1_ = AdaptiveInstanceNorm2d_(in_dim,512)
        self.adain2 = AdaptiveInstanceNorm2d(out_dim)

        self.activ = nn.LeakyReLU(0.2, inplace=True)

        self.sc = nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=False)

        self.token = nn.ParameterList([
            nn.Parameter(torch.randn(1, 512)) for _ in range(tag_number)
        ])


    def forward(self, x,tag_j_trg):
        B, C, H, W = x.shape
        residual = self.sc(x)
        e = self.adain1_(x.clone(), self.token[tag_j_trg].repeat(B, 1))
        out = self.conv2(self.activ(self.adain2(self.conv1(self.activ(e)))))
        out = residual + out

        return out/ math.sqrt(2)

class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

        # weight and bias are dynamically assigned
        self.bias = None
        self.weight = None
        #self.norm = normalization(num_features)

    def forward(self, x):
        assert self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        N, C, H, W = x.size()
        x = x.view(N, C, -1)
        bias_in = x.mean(-1, keepdim=True)
        weight_in = x.std(-1, keepdim=True)

        out = (x - bias_in) / (weight_in + self.eps) * (self.weight+1) + self.bias
        return out.view(N, C, H, W)

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'

class AdaptiveInstanceNorm2d_(nn.Module):
    def __init__(self, num_features, dim,eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

        # weight and bias are dynamically assigned
        self.fc = nn.Linear(dim, num_features*2)

    def forward(self, x,s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = th.chunk(h, chunks=2, dim=1)

        N, C, H, W = x.size()
        x = x.view(N, C, -1)
        bias_in = x.mean(-1, keepdim=True)
        weight_in = x.std(-1, keepdim=True)

        out = (x - bias_in) / (weight_in + self.eps) * (gamma+1) + beta
        return out.view(N, C, H, W)

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'

class AdainResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=512, w_hpf=0,
                 actv=nn.LeakyReLU(0.2), upsample=False):
        super().__init__()
        self.w_hpf = w_hpf  #1
        self.actv = actv
        self.upsample = upsample  #True
        self.learned_sc = dim_in != dim_out  #True
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in, dim_out, style_dim=64):
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.learned_sc:
            x = self.conv1x1(x)

        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        if self.w_hpf == 0:
            out = (out + self._shortcut(x)) / math.sqrt(2)
        return out

class AdaIN(nn.Module):
    '''def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)'''
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = normalization(num_features)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = th.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta

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
        return self.to_out(out)

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
        return out

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
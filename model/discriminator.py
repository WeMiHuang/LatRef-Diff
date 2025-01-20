from torch import nn
import torch.nn.functional as F
import numpy as np
import math
import torch
from model.model_irse import IDnet

class DiffusionVideoAutoencoder(nn.Module):  # 提取id和landmark特征
    def __init__(self, id_file_path, lnd_file_path, bg_file_path):
        super(DiffusionVideoAutoencoder, self).__init__()
        self.idnet = IDnet(id_file_path).requires_grad_(False)  # ID编码

    def forward(self, id):  # id：x_start
        id_feats = self.idnet(id)  # [B, 512]  #4*512
        return id_feats  # 返回id和landmark的融合特征

    def id_forward(self, id):
        return self.idnet(id)  # [B, 512]

    def forward_with_id(self, id_feats, lnd):
        lnd_feats = self.lndnet(lnd)  # [B, 102]
        feats = self.linear(torch.cat([id_feats, lnd_feats], dim=1))  # [B, 512]
        return feats

    def face_mask(self, bg, for_video=False):
        fg_mask = self.bgnet(bg, for_video=for_video)  # [B, 1, 256, 256]
        return fg_mask

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
    def __init__(self, n_in, n_out, norm_fn='none', acti_fn='none'):
        super(LinearBlock, self).__init__()
        layers = [nn.Linear(n_in, n_out, bias=(norm_fn == 'none'))]
        layers = add_normalization_1d(layers, norm_fn, n_out)
        layers = add_activation(layers, acti_fn)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super(Unsqueeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)


class Squeeze(nn.Module):
    def __init__(self, dim):
        super(Squeeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(self.dim)

def add_normalization_1d(layers, fn, n_out):
    if fn == 'none':
        pass
    elif fn == 'batchnorm':
        layers.append(nn.BatchNorm1d(n_out))
    elif fn == 'instancenorm':
        layers.append(Unsqueeze(-1))
        layers.append(nn.InstanceNorm1d(n_out, affine=True))
        layers.append(Squeeze(-1))
    elif fn == 'switchnorm':
        layers.append(SwitchNorm1d(n_out))
    else:
        raise Exception('Unsupported normalization: ' + str(fn))
    return layers

def add_activation(layers, fn):
    if fn == 'none':
        pass
    elif fn == 'relu':
        layers.append(nn.ReLU())
    elif fn == 'lrelu':
        layers.append(nn.LeakyReLU())
    elif fn == 'sigmoid':
        layers.append(nn.Sigmoid())
    elif fn == 'tanh':
        layers.append(nn.Tanh())
    else:
        raise Exception('Unsupported activation function: ' + str(fn))
    return layers

class Dis_att(nn.Module):  # 判别器
    def __init__(self, hyperparameters):
        super().__init__()
        self.tags = hyperparameters.tags  # 列表，字典
        channels = [32, 64, 128, 256, 512, 512, 512]

        self.conv = nn.Sequential(
            nn.Conv2d(3, channels[0], 1, 1, 0),
            *[DownBlock(channels[i], channels[i + 1]) for i in range(len(channels) - 1)],
            #nn.AdaptiveAvgPool2d(1),
        )  # shared

        '''self.fcs = nn.ModuleList([nn.Sequential(
            nn.Linear(channels[-1]*4*4 +
                      # ALI part which is not shown in the original submission but help disentangle the extracted style.
                      hyperparameters['style_dim'] +
                      # Tag-irrelevant part. Sec.3.4
                      self.tags[i]['tag_irrelevant_conditions_dim'],
                      # One for translated, one for cycle. Eq.4  #即使都是有刘海，变换后的和循环一致的用的不同分类分支
                      1024),
                      nn.ReLU(),
                      nn.Linear(1024,len(self.tags[i]['attributes'] * 2))
        ) for i in range(len(self.tags))])'''

        self.fcs = nn.ModuleList([nn.Sequential(
            nn.Linear(channels[-1] * 4 * 4 +
                      # ALI part which is not shown in the original submission but help disentangle the extracted style.
                      hyperparameters.net_beatgans_embed_channels +
                      # Tag-irrelevant part. Sec.3.4
                      2,
                      # One for translated, one for cycle. Eq.4  #即使都是有刘海，变换后的和循环一致的用的不同分类分支
                      1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, len(self.tags[i]['attributes'] * 2))
        ) for i in range(len(self.tags))])

        self.cls=nn.Sequential(nn.Linear(channels[-1]*4*4,1024),
                               nn.LeakyReLU(0.2, inplace=True),
                               nn.Linear(1024, 7))

    def forward(self, x, s, y, i,c=None):  # 8*3*256*256,8*256，8*2
        f = self.conv(x).squeeze(-1).squeeze(-1)  # 8*512*1*1  8*512*4*4


        if  c==None:
            fsy = torch.cat([f.view(f.size(0), -1), s, y], 1)  # 8*770*1*1
            t=self.cls(f.view(f.size(0),-1))
            return self.fcs[i](fsy).view(f.size(0), 2, -1),t

        elif c=='adv':
            fsy = torch.cat([f.view(f.size(0), -1), s, y], 1)  # 8*770*1*1
            return self.fcs[i](fsy).view(f.size(0), 2, -1)  # 8*2*2  有刘海有两个分类分支，没有刘海也有两个分类分支，最后那维是指示属性
        elif c=='cls':
            return self.cls(f.view(f.size(0),-1))
        # t=self.fcs[i](fsy)  #8*4*1*1


    def calc_dis_loss_real(self, x, s, y, i, j):  # 计算更新判别器的loss
        loss = 0
        x = x.requires_grad_()
        out,cla = self.forward(x, s, y, i)
        out=out[:, :, j]
        loss += F.relu(1 - out[:, 0]).mean()
        loss += F.relu(1 - out[:, 1]).mean()  # 真实样本分类为正，真实样本大于1时，loss为0
        loss += self.compute_grad2(out[:, 0], x)
        loss += self.compute_grad2(out[:, 1], x)
        loss += self.compute_grad2(cla, x)
        return loss,cla

    def calc_dis_loss_fake_trg(self, x, s, y, i, j):
        out = self.forward(x, s, y, i,'adv')[:, :, j]
        loss = F.relu(1 + out[:, 0]).mean()  # 生成样本分类为小于等于-1
        return loss

    def calc_dis_loss_fake_cyc(self, x, s, y, i, j):
        out = self.forward(x, s, y, i,'adv')[:, :, j]
        loss = F.relu(1 + out[:, 1]).mean()
        return loss

    def calc_gen_loss_real(self, x, s, y, i, j):  # 计算更新生成器的对抗loss
        loss = 0
        out = self.forward(x, s, y, i,'adv')[:, :, j]  # 8*2
        loss += out[:, 0].mean()
        loss += out[:, 1].mean()
        return loss  # 训练gen，真实样本分类为负

    def calc_gen_loss_fake_trg(self, x, s, y, i, j):
        out,cla = self.forward(x, s, y, i)
        out=out[:, :, j]
        loss = - out[:, 0].mean()
        return loss,cla  # 训练gen，生成样本分类为正

    def calc_gen_loss_fake_cyc(self, x, s, y, i, j):
        out = self.forward(x, s, y, i,'adv')[:, :, j]
        loss = - out[:, 1].mean()
        return loss  # 训练gen，生成样本分类为正

    def compute_grad2(self, d_out, x_in):
        batch_size = x_in.size(0)
        grad_dout = torch.autograd.grad(
            outputs=d_out.sum(), inputs=x_in,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        grad_dout2 = grad_dout.pow(2)
        assert (grad_dout2.size() == x_in.size())
        reg = grad_dout2.view(batch_size, -1).sum(1)
        return reg.mean()

    def cls_gen_loss(self, x, s, y, i, j):  # 计算更新判别器的loss
        cla = self.forward(x, s, y, i,'cls')
        return cla


class Dis(nn.Module):  # 判别器
    def __init__(self, tags):
        super().__init__()
        self.tags = tags  # 列表，字典
        channels =  [32, 64, 128, 256, 512, 512, 512]

        self.conv = nn.Sequential(
            nn.Conv2d(3, channels[0], 1, 1, 0),
            *[DownBlock(channels[i], channels[i + 1]) for i in range(len(channels) - 1)],
            nn.AdaptiveAvgPool2d(1),
        )  # shared

        self.fcs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(channels[-1],
                      # ALI part which is not shown in the original submission but help disentangle the extracted style.

                      # Tag-irrelevant part. Sec.3.4
                      # One for translated, one for cycle. Eq.4  #即使都是有刘海，变换后的和循环一致的用的不同分类分支
                      len(self.tags[i]['attributes'] * 2), 1, 1, 0),
        ) for i in range(len(self.tags))])

    def forward(self, x, s, i):  # 8*3*256*256,8*256，8*2
        f = self.conv(x)  # 8*512*1*1
        fsy = f#torch.cat([f, tile_like(s, f)], 1)  # 8*770*1*1
        # t=self.fcs[i](fsy)  #8*4*1*1
        return self.fcs[i](fsy).view(f.size(0), 2, -1)  # 8*2*2  有刘海有两个分类分支，没有刘海也有两个分类分支，最后那维是指示属性

    def calc_dis_loss_real(self, x, s, i, j):  # 计算更新判别器的loss
        loss = 0
        x = x.requires_grad_()
        out = self.forward(x, s, i)[:, :, j]
        loss += F.relu(1 - out[:, 0]).mean()
        loss += F.relu(1 - out[:, 1]).mean()  # 真实样本分类为正
        loss += self.compute_grad2(out[:, 0], x)
        loss += self.compute_grad2(out[:, 1], x)
        return loss

    def calc_dis_loss_fake_trg(self, x, s,i, j):
        out = self.forward(x, s, i)[:, :, j]
        loss = F.relu(1 + out[:, 0]).mean()  # 生成样本分类为小于等于-1
        return loss

    def calc_dis_loss_fake_cyc(self, x, s, y, i, j):
        out = self.forward(x, s, y, i)[:, :, j]
        loss = F.relu(1 + out[:, 1]).mean()
        return loss

    def calc_gen_loss_real(self, x, s, i, j):  # 计算更新生成器的对抗loss
        loss = 0
        out = self.forward(x, s, i)[:, :, j]  # 8*2
        loss += out[:, 0].mean()
        loss += out[:, 1].mean()
        return loss  # 训练gen，真实样本分类为负

    def calc_gen_loss_fake_trg(self, x, s, i, j):
        out = self.forward(x, s, i)[:, :, j]
        loss = - out[:, 0].mean()
        return loss  # 训练gen，生成样本分类为正

    def calc_gen_loss_fake_cyc(self, x, s, y, i, j):
        out = self.forward(x, s, y, i)[:, :, j]
        loss = - out[:, 1].mean()
        return loss  # 训练gen，生成样本分类为正

    def compute_grad2(self, d_out, x_in):
        batch_size = x_in.size(0)
        grad_dout = torch.autograd.grad(
            outputs=d_out.sum(), inputs=x_in,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        grad_dout2 = grad_dout.pow(2)
        assert (grad_dout2.size() == x_in.size())
        reg = grad_dout2.view(batch_size, -1).sum(1)
        return reg.mean()



class Cla(nn.Module):  # 分类
    def __init__(self, ):
        super().__init__()

        '''self.fc_cls = nn.Sequential(
            LinearBlock(512, 256, 'none', 'lrelu'),
            LinearBlock(256, 7, 'none', 'none')
        )'''
        self.fc_cls = nn.Sequential(nn.Linear(512, 1024),
                                 nn.LeakyReLU(0.2, inplace=True),
                                    nn.Linear(1024, 1024),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Linear(1024, 1024),
                                    nn.LeakyReLU(0.2, inplace=True),
                                 nn.Linear(1024, 8))


    def forward(self, x):  # 8*3*256*256,8*256，8*2
        cls=self.fc_cls(x)
        return cls




class Decoder_(nn.Module):
    def __init__(self):
        super(Decoder_, self).__init__()
        # 上采样层
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # 逐步减少通道数的卷积层
        self.conv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(32, 3, kernel_size=3, padding=1)  # 输出层，3个通道对应RGB

        # 归一化层
        self.norm1 = nn.BatchNorm2d(256)
        self.norm2 = nn.BatchNorm2d(128)
        self.norm3 = nn.BatchNorm2d(64)
        self.norm4 = nn.BatchNorm2d(32)

        # 激活函数
        self.activation = nn.ReLU()

    def forward(self, x):
        # 从 4x4 开始逐步上采样和卷积
        x = self.upsample(x)  # 8x8
        x = self.activation(self.norm1(self.conv1(x)))

        x = self.upsample(x)  # 16x16
        x = self.activation(self.norm2(self.conv2(x)))

        x = self.upsample(x)  # 32x32
        x = self.activation(self.norm3(self.conv3(x)))

        x = self.upsample(x)  # 64x64
        x = self.activation(self.norm4(self.conv4(x)))

        x = self.upsample(x)  # 128x128
        x = self.conv5(x)  # 不在最后一层使用激活函数

        # 将输出限制在 [0, 1] 范围内，因为我们假设图像像素值被归一化
        #x = torch.sigmoid(x)

        return x

class InstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

        # weight and bias are dynamically assigned
        self.weight = nn.Parameter(torch.ones(1, num_features, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1))

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, -1)
        bias_in = x.mean(-1, keepdim=True)
        weight_in = x.std(-1, keepdim=True)

        out = (x - bias_in) / (weight_in + self.eps) * self.weight + self.bias
        return out.view(N, C, H, W)

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

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # 上采样层

        self.channels=[384,256,128,64,32]

        self.decoder = nn.Sequential(
            *[UpBlockIN(self.channels[i], self.channels[i + 1]) for i in range(len(self.channels) - 1)],
            nn.Conv2d(self.channels[-1], 3, 1, 1, 0)
        )

    def forward(self, x):
        # 从 4x4 开始逐步上采样和卷积
        x=self.decoder(x)

        return x






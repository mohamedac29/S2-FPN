
import torch.nn as nn
import torch
from torchvision.models import resnet34, resnet50, resnet101, resnet152, resnet18
# from model.backbones.resnet import resnet34, resnet50, resnet101, resnet152,resnet18
from torchsummaryX import summary
import torch.nn.functional as F
from collections import OrderedDict
from timm.models.layers import DropPath


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=(1, 1), group=1, bn_act=False,
                 bias=False):
        super(conv_block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=group, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.PReLU(out_channels)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.act(self.bn(self.conv(x)))
        else:
            return self.conv(x)

def drop_path(x,drop_prob: float =0, training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1- drop_prob
    shape = (x.shape[0],)+ (1,)*(x.ndim-1)
    random_tensor = keep_prob + torch.rand(shape,dtype=x.dtype,device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor

    return output


class DropPath(nn.Module):
    def __init__(self,drop_path=None):
        super().__init__()
        self.drop_path = drop_path
    def forward(self,x):
        return drop_path(x,self.drop_path,self.training)

def get_shape(tensor):
    shape = tensor.shape
    if torch.onnx.is_in_onnx_export():
        shape = [i.cpu().numpy() for i in shape]
    return shape

class ScaleAwareStripAttention(nn.Module):
    def __init__(self, in_ch, out_ch, droprate=0.15):
        super(ScaleAwareStripAttention, self).__init__()
        self.conv_sh = nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0)
        self.bn_sh1 = nn.BatchNorm2d(in_ch)
        self.bn_sh2 = nn.BatchNorm2d(in_ch)
        self.augmment_conv = nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0)
        self.conv_v = nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0)
        self.conv_res = nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0)
        self.drop = droprate
        self.fuse = conv_block(in_ch, in_ch, kernel_size=1, stride=1, padding=0, bn_act=False)
        self.fuse_out = conv_block(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bn_act=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.size()

        mxpool = F.max_pool2d(x, [h, 1])  #
        mxpool = F.conv2d(mxpool, self.conv_sh.weight, padding=0, dilation=1)
        mxpool = self.bn_sh1(mxpool)
        mxpool_v= mxpool.view(b,c,-1).permute(0,2,1)

        #
        avgpool = F.conv2d(x, self.conv_sh.weight, padding=0, dilation=1)
        avgpool = self.bn_sh2(avgpool)
        avgpool_v = avgpool.view(b,c,-1)

        att = torch.bmm(mxpool_v, avgpool_v)
        att = torch.softmax(att, 1)

        v = F.avg_pool2d(x, [h, 1])  # .view(b,c,-1)
        v = self.conv_v(v)
        v = v.view(b,c,-1)
        att = torch.bmm(v,att)
        att = att.view(b,c,h,w)
        att = self.augmment_conv(att)

        attt1 = att[:, 0, :, :].unsqueeze(1)
        attt2 = att[:, 1, :, :].unsqueeze(1)
        fusion = attt1 * avgpool + attt2 * mxpool
        out = F.dropout(self.fuse(fusion), p=self.drop, training=self.training)
        out = F.relu(self.gamma * out + (1 - self.gamma) * x)
        out = self.fuse_out(out)

        return out


class AttentionBlock(nn.Module):

    def __init__(self, dim,
                 key_dim,
                 num_heads,
                 mlp_ratio=4.,
                 attn_ratio=2.,
                 drop=0.,
                 drop_path=0.):
        super(AttentionBlock,self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.attn = ScaleAwareStripAttention(dim,dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim,out_features=dim,drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(x))
        x = x + self.drop_path(self.mlp(x))
        return x

class ScaleAwareBlock(nn.Module):
    def __init__(self, dim, key_dim, num_heads, mlp_ratio,attn_ratio,num_layers):
        super(ScaleAwareBlock,self).__init__()
        self.tr = nn.Sequential(*(AttentionBlock(dim,key_dim, num_heads,mlp_ratio,attn_ratio) for _ in range(num_layers)))

    def forward(self, x):
        return self.tr(x)

class Mlp(nn.Module):
    def __init__(self, in_features,out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        self.dconv1 = conv_block(in_features, in_features//2, kernel_size=3, stride=1, padding=2, dilation=2, bn_act=True)
        self.dconv2 = conv_block(in_features, in_features//2, kernel_size=3, stride=1, padding=4, dilation=4, bn_act=True)
        self.fuse = conv_block(in_features, out_features, 1,1,0)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        d1 = self.dconv1(x)
        d2 = self.dconv2(x)
        dd = torch.cat([d1,d2],1)
        x = self.fuse(dd)
        x = torch.sigmoid(x)
        x = self.drop(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_pool1 = conv_block(channel, channel // reduction, 1, 1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=False)
        self.conv_pool2 = conv_block(channel // reduction, channel, 1, 1, padding=0, bias=False)
        self.gimoid = nn.Sigmoid()

    def forward(self, x):
        p1 = self.avg_pool1(x)
        p2 = self.avg_pool1(x)
        p = p1+p2
        y = self.conv_pool(p)
        y = x * y
        return y

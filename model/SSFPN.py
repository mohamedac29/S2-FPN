import torch.nn as nn
import torch
from torchvision.models import resnet34, resnet50, resnet101, resnet152, resnet18
from torchsummaryX import summary
import torch.nn.functional as F
from collections import OrderedDict
import functools
from functools import partial
import os, sys
from model.sync_batchnorm import SynchronizedBatchNorm2d

__all__ = ['SSFPN']


class SSFPN(nn.Module):
    def __init__(self, backbone,
                 pretrained=True,
                 ResNet34M= False,
                 classes=11):
        super(SSFPN, self).__init__()
        self.ResNet34M = ResNet34M
        self.backbone = backbone

        if backbone.lower() == "resnet18":
            encoder = resnet18(pretrained=pretrained)
        elif backbone.lower() == "resnet34":
            encoder = resnet34(pretrained=pretrained)
        elif backbone.lower() == "resnet50":
            encoder = resnet50(pretrained=pretrained)
        elif backbone.lower() == "resnet101":
            encoder = resnet101(pretrained=pretrained)
        elif backbone.lower() == "resnet152":
            encoder = resnet152(pretrained=pretrained)
        else:
            raise NotImplementedError("{} Backbone not implemented".format(backbone))

        self.out_channels = [32,64,128,256,512,1024,2048]
        # self.conv1 = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu, encoder.maxpool)
        self.conv1_x = encoder.conv1
        self.bn1 = encoder.bn1
        self.relu = encoder.relu
        self.maxpool = encoder.maxpool
        self.conv2_x = encoder.layer1  # 1/4
        self.conv3_x = encoder.layer2  # 1/8
        self.conv4_x = encoder.layer3  # 1/16
        self.conv5_x = encoder.layer4  # 1/32

        # if backbone in ['resnet50','resnet101','resnet152']:
        self.down2 = conv_block(self.out_channels[-4], self.out_channels[1], 3, 1, 1, 1, 1, bn_act=True)
        self.down3 = conv_block(self.out_channels[-3], self.out_channels[2], 3, 1, 1, 1, 1, bn_act=True)
        self.down4 = conv_block(self.out_channels[-2], self.out_channels[3], 3, 1, 1, 1, 1, bn_act=True)
        self.down5 = conv_block(self.out_channels[-1], self.out_channels[4], 3, 1, 1, 1, 1, bn_act=True)

        self.fab = nn.Sequential(
            conv_block(self.out_channels[4],
                       self.out_channels[4]//2,
                       kernel_size = 3,
                       stride= 1,
                       padding=1,
                       group=self.out_channels[4]//2,
                       dilation=1,
                       bn_act=True),
                       nn.Dropout(p=0.15))

        self.cfgb = nn.Sequential(
            conv_block(self.out_channels[4],
                       self.out_channels[4],
                       kernel_size =3,
                       stride= 2,
                       padding = 1,
                       group=self.out_channels[4],
                       dilation=1,
                       bn_act=True),
                       nn.Dropout(p=0.15))



        self.gfu4 = GlobalFeatureUpsample(self.out_channels[3], self.out_channels[3], self.out_channels[3])
        self.gfu3 = GlobalFeatureUpsample(self.out_channels[2], self.out_channels[3], self.out_channels[2])
        self.gfu2 = GlobalFeatureUpsample(self.out_channels[1], self.out_channels[2], self.out_channels[1])
        self.gfu1 = GlobalFeatureUpsample(self.out_channels[0], self.out_channels[1], self.out_channels[0])


        self.apf1 = PyrmidFusionNet(self.out_channels[4], self.out_channels[4], self.out_channels[3], classes=classes)
        self.apf2 = PyrmidFusionNet(self.out_channels[3], self.out_channels[3], self.out_channels[2], classes=classes)
        self.apf3 = PyrmidFusionNet(self.out_channels[2], self.out_channels[2], self.out_channels[1], classes=classes)
        self.apf4 = PyrmidFusionNet(self.out_channels[1], self.out_channels[1], self.out_channels[0], classes=classes)



        self.classifier = SegHead(self.out_channels[0], classes)

    def forward(self, x):
        B, C, H, W = x.size()
        x = self.conv1_x(x)
        x = self.bn1(x)
        x1 = self.relu(x)
        x = self.maxpool(x1)
        if self.ResNet34M:
            x2 = self.conv2_x(x1)
        else:
            x2 = self.conv2_x(x)
        x3 = self.conv3_x(x2)
        x4 = self.conv4_x(x3)
        x5 = self.conv5_x(x4)

        if self.backbone in ['resnet50', 'resnet101', 'resnet152']:
            x2 = self.down2(x2)
            x3 = self.down3(x3)
            x4 = self.down4(x4)
            x5 = self.down5(x5)

        CFGB = self.cfgb(x5)

        APF1, cls1 = self.apf1(CFGB, x5)
        APF2, cls2 = self.apf2(APF1, x4)
        APF3, cls3 = self.apf3(APF2, x3)
        APF4, cls4 = self.apf4(APF3, x2)

        FAB = self.fab(x5)

        dec5 = self.gfu4(APF1, FAB)  # [2, 512, 16, 16]
        dec4 = self.gfu3(APF2, dec5)
        dec3 = self.gfu2(APF3, dec4)
        dec2 = self.gfu1(APF4, dec3)

        classifier = self.classifier(dec2)

        sup1 = F.interpolate(cls1, size=(H, W), mode="bilinear", align_corners=True)
        sup2 = F.interpolate(cls2, size=(H, W), mode="bilinear", align_corners=True)
        sup3 = F.interpolate(cls3, size=(H, W), mode="bilinear", align_corners=True)
        sup4 = F.interpolate(cls4, size=(H, W), mode="bilinear", align_corners=True)
        predict = F.interpolate(classifier, size=(H, W), mode="bilinear", align_corners=True)

        if self.training:
            return predict, sup1, sup2, sup3, sup4
        else:
            return predict


class PyrmidFusionNet(nn.Module):
    def __init__(self, channels_high, channels_low, channel_out, classes=11):
        super(PyrmidFusionNet, self).__init__()

        self.lateral_low = conv_block(channels_low, channels_high, 1, 1, bn_act=True, padding=0)

        self.conv_low = conv_block(channels_high, channel_out, 3, 1, bn_act=True, padding=1)
        self.sa = SpatialAttention(channel_out, channel_out)

        self.conv_high = conv_block(channels_high, channel_out, 3, 1, bn_act=True, padding=1)
        self.ca = ChannelWise(channel_out)

        self.FRB = nn.Sequential(
            conv_block(2 * channels_high, channel_out, 1, 1, bn_act=True, padding=0),
            conv_block(channel_out, channel_out, 3, 1, bn_act=True, group=1, padding=1))

        self.classifier = nn.Sequential(
            conv_block(channel_out, channel_out, 3, 1, padding=1, group=1, bn_act=True),
            nn.Dropout(p=0.15),
            conv_block(channel_out, classes, 1, 1, padding=0, bn_act=False))
        self.apf = conv_block(channel_out, channel_out, 3, 1, padding=1, group=1, bn_act=True)

    def forward(self, x_high, x_low):
        _, _, h, w = x_low.size()

        lat_low = self.lateral_low(x_low)

        high_up1 = F.interpolate(x_high, size=lat_low.size()[2:], mode='bilinear', align_corners=False)

        concate = torch.cat([lat_low, high_up1], 1)
        concate = self.FRB(concate)

        conv_high = self.conv_high(high_up1)
        conv_low = self.conv_low(lat_low)

        sa = self.sa(concate)
        ca = self.ca(concate)

        mul1 = torch.mul(sa, conv_high)
        mul2 = torch.mul(ca, conv_low)

        att_out = mul1 + mul2

        sup = self.classifier(att_out)
        APF = self.apf(att_out)
        return APF,sup


class GlobalFeatureUpsample(nn.Module):
    def __init__(self, low_channels, in_channels, out_channels):
        super(GlobalFeatureUpsample, self).__init__()

        self.conv1 = conv_block(low_channels, out_channels, kernel_size=1, stride=1, padding=0, bn_act=True)
        self.conv2 = nn.Sequential(
            conv_block(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bn_act=False),
            nn.ReLU(inplace=True))
        self.conv3 = conv_block(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bn_act=True)

        # self._init_weight()

    def forward(self, x_gui, y_high):
        h, w = x_gui.size(2), x_gui.size(3)
        y_up = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(y_high)
        x_gui = self.conv1(x_gui)
        y_up = F.avg_pool2d(self.conv2(y_up), (1, 1))
        out = y_up + x_gui

        return self.conv3(out)

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=(1, 1), group=1, bn_act=False,
                 bias=False):
        super(conv_block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=group, bias=bias)
        self.bn = SynchronizedBatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=False)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.act(self.bn(self.conv(x)))
        else:
            return self.conv(x)


class SegHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SegHead, self).__init__()

        self.fc = conv_block(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        return self.fc(x)


class SpatialAttention(nn.Module):
    def __init__(self, in_ch, out_ch, droprate=0.15):
        super(SpatialAttention, self).__init__()
        self.conv_sh = nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0)
        self.bn_sh1 = nn.BatchNorm2d(in_ch)
        self.bn_sh2 = nn.BatchNorm2d(in_ch)
        self.conv_res = nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0)
        self.drop = droprate
        self.fuse = conv_block(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bn_act=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.size()

        mxpool = F.max_pool2d(x, [h, 1])  # .view(b,c,-1).permute(0,2,1)
        mxpool = F.conv2d(mxpool, self.conv_sh.weight, padding=0, dilation=1)
        mxpool = self.bn_sh1(mxpool)

        avgpool = F.avg_pool2d(x, [h, 1])  # .view(b,c,-1)
        avgpool = F.conv2d(avgpool, self.conv_sh.weight, padding=0, dilation=1)
        avgpool = self.bn_sh2(avgpool)

        att = torch.softmax(torch.mul(mxpool, avgpool), 1)
        attt1 = att[:, 0, :, :].unsqueeze(1)
        attt2 = att[:, 1, :, :].unsqueeze(1)

        fusion = attt1 * avgpool + attt2 * mxpool
        out = F.dropout(self.fuse(fusion), p=self.drop, training=self.training)

        # out = out.expand(residual.shape[0],residual.shape[1],residual.shape[2],residual.shape[3])
        out = F.relu(self.gamma * out + (1 - self.gamma) * x)
        return out


class ChannelWise(nn.Module):
    def __init__(self, channel, reduction=4):
        super(ChannelWise, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_pool = nn.Sequential(
            conv_block(channel, channel // reduction, 1, 1, padding=0, bias=False), nn.ReLU(inplace=False),
            conv_block(channel // reduction, channel, 1, 1, padding=0, bias=False), nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_pool(y)

        return x * y


if __name__ == "__main__":
    input1 = torch.rand(2, 3, 360, 480)
    model = SSFPN("resnet18",ResNet34M=False)
    summary(model, torch.rand((2, 3, 360, 480)))

    # python train_cityscapes.py --dataset camvid --model MSFFNet --batch_size 4 --max_epochs 300 --train_type trainval --lr 1e-3

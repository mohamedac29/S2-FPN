import torch.nn as nn
import torch
from torchvision.models import resnet34, resnet50, resnet101, resnet152, resnet18
# from model.backbones.resnet import resnet34, resnet50, resnet101, resnet152,resnet18
from torchsummaryX import summary
import torch.nn.functional as F
from collections import OrderedDict

# from conv_block import Conv
import functools
from functools import partial
import os, sys

# from inplace_abn import InPlaceABN, InPlaceABNSync
# from model.sync_batchnorm import SynchronizedBatchNorm2d

# BatchNorm2d = functools.partial(InPlaceABNSync, activation='identity')
# from torch.nn import SyncBatchNorm

__all__ = ['SPFNet']

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
class SPFNet(nn.Module):
    def __init__(self, backbone, pretrained=True, classes=11):
        super(SPFNet, self).__init__()

        if backbone.lower() == "resnet18":
            encoder = resnet18(pretrained=pretrained)
            out_channels = 512
        elif backbone.lower() == "resnet34":
            encoder = resnet34(pretrained=pretrained)
            out_channels = 512
        elif backbone.lower() == "resnet50":
            encoder = resnet50(pretrained=pretrained)
            out_channels = 2048
        elif backbone.lower() == "resnet101":
            encoder = resnet101(pretrained=pretrained)
            out_channels = 2048
        elif backbone.lower() == "resnet152":
            encoder = resnet152(pretrained=pretrained)
        else:
            raise NotImplementedError("{} Backbone not implemented".format(backbone))

        # self.conv1 = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu, encoder.maxpool)
        self.conv = encoder.conv1
        self.bn1 = encoder.bn1
        self.relu = encoder.relu
        self.maxpool = encoder.maxpool
        self.conv2_x = encoder.layer1  # 1/4
        self.conv3_x = encoder.layer2  # 1/8
        self.conv4_x = encoder.layer3  # 1/16
        self.conv5_x = encoder.layer4  # 1/32
        
        
        self.spfm = SPFM(512, 512,4)       
        
     
        channels = [64, 64, 128, 256, 512]
      
        self.egca1 = EGCA(channels[1])
        self.egca2 = EGCA(channels[2])
        self.egca3 = EGCA(channels[3])
        self.egca4 = EGCA(channels[4])
        
        self.adjust2 = Adjustment(64, 64)
        self.adjust3 = Adjustment(128, 64)
        self.adjust4 = Adjustment(256, 64)
        self.adjust5 = Adjustment(512, 64)

        # Multi-Scale feature fusion
        self.fuse = conv_block(320, 64, 3, 1, padding=1, bn_act=True)
        
        # Decoder-based subpixel convolution
        self.dsc5 = DSCModule(512, 256)
        self.dsc4 = DSCModule(256, 128)
        self.dsc3 = DSCModule(128, 64)
        self.dsc2 = DSCModule(64, 64)

        self.classifier = Classifier(64, classes)

    def forward(self, x):
        B, C, H, W = x.size()
        x = self.conv(x)
        x = self.bn1(x)
        x1 = self.relu(x)

        x = self.maxpool(x1)
        x2 = self.conv2_x(x)
        x3 = self.conv3_x(x2)
        x4 = self.conv4_x(x3)
        x5 = self.conv5_x(x4)

        Spfm = self.spfm(x5)

        dsc5 = self.dsc5(x5,Spfm)  
        dsc4 = self.dsc4(x4,dsc5)
        dsc3 = self.dsc3(x3,dsc4)
        dsc2 = self.dsc2(x2,dsc3)

        # Efficient global context aggregation
        gui1 = self.egca1(x2) 
        gui2 = self.egca2(x3) 
        gui3 = self.egca3(x4)  
        gui4 = self.egca4(x5) 

        adj2 = self.adjust2(gui1)
        adj3 = self.adjust3(gui2)
        adj4 = self.adjust4(gui3)
        adj5 = self.adjust5(gui4)

        adj2 = F.interpolate(adj2, size=x2.size()[2:], mode="bilinear")
        adj3 = F.interpolate(adj3, size=x2.size()[2:], mode="bilinear")
        adj4 = F.interpolate(adj4, size=x2.size()[2:], mode="bilinear")
        adj5 = F.interpolate(adj5, size=x2.size()[2:], mode="bilinear")
        dsc2 = F.interpolate(dsc2, size=x2.size()[2:], mode="bilinear")
      

        msfuse = torch.cat((adj2,dsc2,adj3,adj4,adj5), dim=1)
        msfuse = self.fuse(msfuse)

        classifier = self.classifier(msfuse)
        classifier = F.interpolate(classifier, size=(H, W), mode="bilinear", align_corners=True)

        return classifier


class RPPModule(nn.Module):
    def __init__(self, in_channels: int, groups=2) -> None:
        super(RPPModule, self).__init__()
        self.groups = groups
        self.conv_dws1 = nn.Sequential(
            conv_block(in_channels, 4*in_channels, kernel_size=3, stride=1, padding=4,
                                    group=1, dilation=4, bn_act=True),
            nn.PixelShuffle(upscale_factor=2))
        self.conv_dws2 = nn.Sequential(
            conv_block(in_channels, 4*in_channels, kernel_size=3, stride=1, padding=8,
                                    group=1, dilation=8, bn_act=True),
            nn.PixelShuffle(upscale_factor=2))

        self.fusion = nn.Sequential(
            conv_block(2 * in_channels, in_channels, kernel_size=1, stride=1, padding=0, group=1, bn_act=True),
            conv_block(in_channels, in_channels, kernel_size=3, stride=1, padding=1, group=1, bn_act=True),
            conv_block(in_channels, in_channels, kernel_size=1, stride=1, padding=0, group=1, bn_act=True))

        self.conv_dws3 = conv_block(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bn_act=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        br1 = self.conv_dws1(x)
        b2 = self.conv_dws1(x)

        out = torch.cat((br1, b2), dim=1)
        out = self.fusion(out)

        br3 = self.conv_dws3(F.adaptive_avg_pool2d(x, (1, 1)))
        output = br3 + out

        return output


class SPFM(nn.Module):
    def __init__(self, in_channels, out_channels, num_splits):
        super(SPFM,self).__init__()

        assert in_channels % num_splits == 0

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_splits = num_splits
        self.subspaces = nn.ModuleList(
            [RPPModule(int(self.in_channels / self.num_splits)) for i in range(self.num_splits)])

        self.out = conv_block(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bn_act=True)

    def forward(self, x):
        group_size = int(self.in_channels / self.num_splits)
        sub_Feat = torch.chunk(x, self.num_splits, dim=1)
        out = []
        for id, l in enumerate(self.subspaces):
            out.append(self.subspaces[id](sub_Feat[id]))
        out = torch.cat(out, dim=1)
        out = self.out(out)
        return out


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)

    return x


class EGCA(nn.Module):
    def __init__(self, in_channels: int, groups=2) -> None:
        super(EGCA, self).__init__()
        self.groups = groups
        self.conv_dws1 = conv_block(in_channels // 2, in_channels, kernel_size=1, stride=1, padding=0,
                                    group=in_channels // 2, bn_act=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv_pw1 = conv_block(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bn_act=False)
        self.softmax = nn.Softmax(dim=1)

        self.conv_dws2 = conv_block(in_channels // 2, in_channels, kernel_size=1, stride=1, padding=0,
                                    group=in_channels // 2,
                                    bn_act=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv_pw2 = conv_block(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bn_act=False)

        self.branch3 = nn.Sequential(
            conv_block(in_channels, in_channels, kernel_size=3, stride=1, padding=1, group=in_channels, bn_act=True),
            conv_block(in_channels, in_channels, kernel_size=1, stride=1, padding=0, group=1, bn_act=True)) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0, x1 = x.chunk(2, dim=1)
        out1 = self.conv_dws1(x0)
        out1 = self.maxpool1(out1)
        out1 = self.conv_pw1(out1)

        out2 = self.conv_dws1(x1)
        out2 = self.maxpool1(out2)
        out2 = self.conv_pw1(out2)

        out = torch.add(out1, out2)

        b, c, h, w = out.size()
        out = self.softmax(out.view(b, c, -1))
        out = out.view(b, c, h, w)

        out = out.expand(x.shape[0], x.shape[1], x.shape[2], x.shape[3])

        out = torch.mul(out, x)
        out = torch.add(out, x)
        out = channel_shuffle(out, groups=self.groups)

        br3 = self.branch3(x)

        output = br3 + out

        return output



class DSCModule(nn.Module):
    def __init__(self, in_channels, out_channels, red=1):
        super(DSCModule, self).__init__()
     
        self.conv1 = conv_block(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bn_act=True)
        self.conv2 = conv_block(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bn_act=True)
        self.conv3 = nn.Sequential(
            conv_block(2 * in_channels, 4 * out_channels, kernel_size=3, stride=1, padding=1, bn_act=True),
            nn.PixelShuffle(upscale_factor=2))

    def forward(self, x_gui, y_high):
        h, w = x_gui.size(2), x_gui.size(3)

        y_high = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(y_high)
        x_gui = self.conv1(x_gui)
        y_high = self.conv2(y_high)
       
        out = torch.cat([y_high, x_gui], 1)

        out = self.conv3(out)

        return out


class Classifier(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Classifier, self).__init__()
        self.fc = conv_block(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return self.fc(x)


class Adjustment(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Adjustment, self).__init__()
        self.conv = conv_block(in_channels, out_channels, 1, 1, padding=0, bn_act=True)

    def forward(self, x):
        return self.conv(x)




if __name__ == "__main__":
    input_tensor = torch.rand(2, 3, 512, 1024)
    model = SPFNet("resnet18")
    summary(model, input_tensor)





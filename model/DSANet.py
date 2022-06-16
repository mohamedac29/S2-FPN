import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

__all__ = ["DSANet"]


def split(x):
    c = int(x.size()[1])
    c1 = round(c * 0.5)
    x1 = x[:, :c1, :, :].contiguous()
    x2 = x[:, c1:, :, :].contiguous()

    return x1, x2


def channel_shuffle(x, groups):
    batchSize, channels, height, width = x.data.size()
    channel_per_group = channels // groups
    x = x.view(batchSize, groups, channel_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchSize, -1, height, width)

    return x


class ConvBNPreLU(nn.Module):
    def __init__(self, in_channels, out_channels, kSize, stride, padding, dilation=(1, 1),groups=1,bn_act=False, bias=False):
        super().__init__()
        self.bn_act = bn_act
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kSize,stride=stride,
                              padding=padding,dilation=dilation, groups=groups, bias=bias)
        if self.bn_act:
            self.bn_prelu = BNPReLU(out_channels)

    def forward(self, inputs):
        output = self.conv(inputs)
        if self.bn_act:
            output = self.bn_prelu(output)

        return output


class BNPReLU(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels, eps=1e-3)
        self.acti = nn.PReLU(in_channels)

    def forward(self, inputs):
        output = self.bn(inputs)
        output = self.acti(output)

        return output


class FDSSModule(nn.Module):
    def __init__(self, in_channels, d=1, kSize=1, dkSize=3):
        super(FDSSModule,self).__init__()

        self.bn_relu_1 = BNPReLU(in_channels)
        self.conv3x3 = ConvBNPreLU(in_channels, in_channels, kSize, 1, padding=0, bn_act=True)
        self.conv3x31 =ConvBNPreLU(in_channels, in_channels, 3, 1, padding=1, bn_act=True)

        self.dconv3x1 = ConvBNPreLU(in_channels // 2, in_channels // 2, (dkSize, 1), 1,
                             padding=(1, 0), groups=in_channels // 2, bn_act=True)
        self.dconv1x3 = ConvBNPreLU(in_channels // 2, in_channels // 2, (1, dkSize), 1,
                             padding=(0, 1), groups=in_channels // 2, bn_act=True)
        self.ddconv3x1 = ConvBNPreLU(in_channels // 2, in_channels // 2, (dkSize, 1), 1,
                              padding=(1 * d, 0), dilation=(d, 1), groups=in_channels // 2, bn_act=True)
        self.ddconv1x3 = ConvBNPreLU(in_channels // 2, in_channels // 2, (1, dkSize), 1,
                              padding=(0, 1 * d), dilation=(1, d), groups=in_channels // 2, bn_act=True)

    @staticmethod
    def _concat(x, out):
        return torch.cat((x, out), 1)

    def forward(self, input):
        output = self.bn_relu_1(input)
        output = self.conv3x3(output)
        output = self.conv3x31(output)

        x1, x2 = split(output)

        br1 = self.dconv3x1(x1)
        br1 = self.dconv1x3(br1)

        br2 = self.ddconv3x1(x2)
        br2 = self.ddconv1x3(br2)

        output = self._concat(br1, br2)

        out = output + input
        out = channel_shuffle(out, 2)

        return out


class DownSamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if self.in_channels < self.out_channels:
            nConv = out_channels - in_channels
        else:
            nConv = out_channels

        self.conv3x3 = ConvBNPreLU(in_channels, nConv, kSize=3, stride=2, padding=1)
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.bn_prelu = BNPReLU(out_channels)

    def forward(self, input):
        output = self.conv3x3(input)

        if self.in_channels < self.out_channels:
            max_pool = self.max_pool(input)
            output = torch.cat([output, max_pool], 1)

        output = self.bn_prelu(output)

        return output


class Spatial_path(nn.Module):
    def __init__(self, kSize=3, dkSize=3, d=3):
        super().__init__()

        # branch 1
        self.conv1 = ConvBNPreLU(3, 32, kSize=3, stride=2, padding=1, bn_act=True)
        self.conv11 = ConvBNPreLU(32, 32, kSize=3, stride=1, padding=1, bn_act=True)
        # branch 2
        self.conv2 = ConvBNPreLU(32, 64, kSize=3, stride=2, padding=1, bn_act=True)
        self.conv21 = ConvBNPreLU(64, 64, kSize=3, stride=1, padding=1, bn_act=True)
        # branch 3
        self.conv3 = ConvBNPreLU(64, 128, kSize=3, stride=2, padding=1, bn_act=True)
        self.conv31 = ConvBNPreLU(128, 128, kSize=3, stride=1, padding=1, bn_act=True)

    def forward(self, inputs):
        output1 = self.conv1(inputs)
        output1 = self.conv11(output1)
        output2 = self.conv2(output1)
        output2 = self.conv21(output2)
        output3 = self.conv3(output2)
        output3 = self.conv31(output3)

        return output3


class DSAModule(nn.Module):
    def __init__(self,in_channels, kSize=1, dkSize=1):
        super(DSAModule,self).__init__()

        self.A1 = nn.Sequential(
            BNPReLU(in_channels),
            ConvBNPreLU(in_channels, in_channels, kSize, 1, padding=0, bn_act=True),
        )
        self.dconv2 = ConvBNPreLU(in_channels, in_channels, 3, 1, padding=3, dilation=3, bn_act=False)
        self.dconv3 = ConvBNPreLU(in_channels, in_channels, 3, 1, padding=3, dilation=3, bn_act=False)
        self.dconv4 = ConvBNPreLU(in_channels, in_channels, 1, 1, padding=0, bn_act=True)
        self.dconv5 = ConvBNPreLU(in_channels, in_channels, 1, 1, padding=0, bn_act=True)

        self.conv = ConvBNPreLU(in_channels, in_channels, 1, 1, padding=0, bn_act=True)
        self.sigmoid = nn.Softmax(dim=1)

    def forward(self, input_):
        output = self.A1(input_)

        br1 = self.dconv2(output)
        br2 = self.dconv3(output)

        br3 = self.dconv4(output)
        br4 = self.dconv5(output)

        out1 = torch.mul(br1, br2)
        out1 = self.sigmoid(out1)
        out2 = torch.mul(out1, br3)
        # out2 = self.sigmoid(out2)
        out3 = torch.add(output, out2)
        x1 = self.conv(out3)
        # x1 = self.sigmoid(x1)
        out = torch.add(input_, x1)
        return out


class CAttention(nn.Module):

    def __init__(self, in_channels, reduction=4):
        super(CAttention, self).__init__()

        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.fc2 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_):
        b, c, h, w = input_.size()

        feat = F.adaptive_avg_pool2d(input_, (1, 1)).view(b, c)
        # feat1 = F.relu(self.fc1(feat))
        # feat2 = torch.relu(self.fc2(feat))
        # out = torch.add(feat1, feat2)
        # out = torch.sigmoid(out)

        # Activity regularizer
        # ca_act_reg = torch.mean(feat)
        feat = feat.view(b, c, 1, 1)
        feat = feat.expand_as(input_).clone()

        return feat


class DualAttention(nn.Module):

    def __init__(self, in_channels,reduction=2):
        super(DualAttention, self).__init__()
        self.spatial = DSAModule(in_channels)
        self.convs = ConvBNPreLU(in_channels, in_channels, 1, 1, padding=0, bn_act=True)
        self.chan_wise = CAttention(in_channels, reduction)
        self.convc = ConvBNPreLU(in_channels, in_channels, 1, 1, padding=0, bn_act=True)

    def forward(self, inputs):
        spatial = self.spatial(inputs)
        spatial = self.convs(spatial)
        channW = self.chan_wise(inputs)
        channW = self.convc(channW)
        out = torch.add(spatial, channW)
        return out


class DSANet(nn.Module):
    def __init__(self, classes=11, block_1=3, block_2=4, block_3=4):
        super(DSANet,self).__init__()

        # Spatial Encoding Network
        self.spatial = nn.Sequential()
        self.spatial.add_module("Spatial_Encoding_Network", Spatial_path())
        self.bn_prelu_sp = BNPReLU(128)

        self.init_block = nn.Sequential(
            ConvBNPreLU(3, 32, 3, 2, padding=1, bn_act=True),
            ConvBNPreLU(32, 32, 3, 1, padding=1, bn_act=True),
            ConvBNPreLU(32, 32, 3, 1, padding=1, bn_act=True))

        self.bn_prelu_1 = BNPReLU(32)

        # Semantic Path
        # FDSSModule Block 1
        self.downsample_1 = DownSamplingBlock(32, 64)
        self.FDSS_Block_1 = nn.Sequential()
        for i in range(0, block_1):
            self.FDSS_Block_1.add_module("FDSS_Module_1_" + str(i),FDSSModule(64, d=1))
        self.bn_prelu_2 = BNPReLU(64)
        self.attentio1 = nn.Sequential()
        self.attentio1.add_module("DualAttention_1_", DualAttention(64))
        self.refine1 = nn.Sequential(
            ConvBNPreLU(64, 64, 3, 1, padding=1, bn_act=True),
            ConvBNPreLU(64, 64, 3, 1, padding=1, bn_act=True),
            ConvBNPreLU(64, classes, 1, 1, padding=0))

        # FDSSModule Block 2
        dilation_block2 = [1, 3, 6, 12]
        self.downsample_2 = DownSamplingBlock(64, 128)
        self.FDSS_Block_2 = nn.Sequential()
        for i in range(0, block_2):
            self.FDSS_Block_2.add_module("FDSS_Module_2_" + str(i),
                                       FDSSModule(128, d=dilation_block2[i]))
        self.bn_prelu_3 = BNPReLU(128)
        self.attentio2 = nn.Sequential()
        self.attentio2.add_module("DualAttention_2_", DualAttention(128))
        self.refine2 = nn.Sequential(
            ConvBNPreLU(128, 64, 3, 1, padding=1, bn_act=True),
            ConvBNPreLU(64, 128, 3, 1, padding=1, bn_act=True),
            ConvBNPreLU(128, classes, 1, 1, padding=0))

        # FDSSModule Block 3
        dilation_block3 = [3, 6, 12, 24]
        self.FDSS_Block_3 = nn.Sequential()
        for i in range(0, block_3):
            self.FDSS_Block_3.add_module("FDSS_Module_3_" + str(i),
                                       FDSSModule(128, d=dilation_block3[i]))
        self.bn_prelu_4 = BNPReLU(128)

        self.attentio3 = nn.Sequential()
        self.attentio3.add_module("DualAttention_3_", DualAttention(128))
        self.refine3 = nn.Sequential(
            ConvBNPreLU(128, 64, 3, 1, padding=1, bn_act=True),
            ConvBNPreLU(64, 128, 3, 1, padding=1, bn_act=True),
            ConvBNPreLU(128, classes, 1, 1, padding=0))

        # self.conv3 = nn.Sequential(
        #     Conv(256, 128, 3, stride=1, padding=1,bn_act=True),
        #     Conv(128, 128, 1, 1, padding=0,bn_act=True))
        # # self.conv = nn.Sequential(Conv(128,128,1,1,padding=0))
        # # self.bn_prelu_5 = BNPReLU(128)

        self.classifier = ConvBNPreLU(256,classes, 1, 1, padding=0)

    def forward(self, input):
        spatial_ = self.spatial(input)
        spatial_ = self.bn_prelu_sp(spatial_)
        # print("Spatial Size: ",spatial_.size())

        output = self.init_block(input)
        output = self.bn_prelu_1(output)
        # Block 1
        output1 = self.downsample_1(output)
        output1 = self.FDSS_Block_1(output1)
        output1 = self.bn_prelu_2(output1)
        predict1 = self.attentio1(output1)
        predict1 = self.refine1(predict1)
        predict1 = F.interpolate(predict1, input.size()[2:], mode='bilinear', align_corners=False)
        # Block 2
        output2 = self.downsample_2(output1)
        output2 = self.FDSS_Block_2(output2)
        output2 = self.bn_prelu_3(output2)
        predict2 = self.attentio2(output2)
        predict2 = self.refine2(predict2)
        predict2 = F.interpolate(predict2, input.size()[2:], mode='bilinear', align_corners=False)
        # Block 3
        output3 = self.FDSS_Block_3(output2)
        output3 = self.bn_prelu_4(output3)
        # output3 = self.conv(output3)

        predict3 = self.attentio3(output3)
        predict3 = self.refine3(predict3)
        predict3 = F.interpolate(predict3, input.size()[2:], mode='bilinear', align_corners=False)
        # print("output3 size",output3.size())

        ## Feature Fusion
        FFM = torch.cat((spatial_, output3), 1)
        # FFM = self.conv3(FFM)
        # FFM = self.conv(FFM)
        # FFM = self.bn_prelu_5(FFM)
        # print("FFM Size: ",FFM.size())

        # Classifier
        out = self.classifier(FFM)
        # print("Classifier Size: ",out.size())
        out = F.interpolate(out, input.size()[2:], mode='bilinear', align_corners=False)

        return out, predict1, predict2, predict3


if __name__ == "__main__":
    from pthflops import count_ops
    input = torch.Tensor(1, 3,360,480).cuda()
    model = DSANet(classes=11).cuda()
    model.eval()
    print(model)
    output = model(input)
    summary(model, (3, 360, 480))




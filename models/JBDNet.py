import torch
import torch.nn as nn
import torch.nn.functional as F

from models.res2net import res2net50_v1b_26w_4s


# 基础卷积层
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# decoder层
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DecoderBlock, self).__init__()

        self.conv1 = ConvBlock(in_channels, in_channels // 4, kernel_size=kernel_size, stride=stride, padding=padding)

        self.conv2 = ConvBlock(in_channels // 4, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.pre = nn.Conv2d(out_channels, 1, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.upsample(x)

        # PR 
        pre = self.pre(x)
        # 边界信息
        return x, pre


# 桥阶层
class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )


class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(1, eps=1e-5, momentum=0.01, affine=True)
        )

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale


class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.hw = AttentionGate()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = 1 / 3 * (x_out + x_out11 + x_out21)
        else:
            x_out = 1 / 2 * (x_out11 + x_out21)
        return x_out


# 层层之间使用多尺度的方法进行融合，输入的图像尺寸维度应该相同
class BIII(nn.Module):
    def __init__(self, in_channel1, in_channel2, out_channel, N=2):
        super(BIII, self).__init__()

        act_fn = nn.ReLU(inplace=False)

        ## ---------------------------------------- ##
        self.layer0 = ConvBlock(in_channel1, out_channel // 2, 1, stride=1, padding=0)
        self.layer1 = ConvBlock(in_channel2, out_channel // 2, 1, 1, 0)

        # 改为 ModuleList 以便手动控制输入
        self.layer3 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channel, out_channel // 2, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channel // 2), act_fn
            ) for _ in range(N)
        ])

        self.layer5 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channel, out_channel // 2, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(out_channel // 2), act_fn
            ) for _ in range(N)
        ])

        self.layer_out1 = nn.Conv2d(out_channel // 2, in_channel1, kernel_size=1, stride=1, padding=0)
        self.layer_out2 = nn.Conv2d(out_channel // 2, in_channel2, kernel_size=1, stride=1, padding=0)

    def forward(self, x0, x1, original_size):
        ## ------------------------------------------------------------------ ##
        x0_1 = self.layer0(x0)  # 初始变换
        x1_1 = self.layer1(x1)  # 初始变换

        # 初始化第一层输入
        prev3 = torch.cat((x0_1, x1_1), dim=1)
        prev5 = torch.cat((x1_1, x0_1), dim=1)

        # 逐层处理
        for i in range(len(self.layer3)):
            out3 = self.layer3[i](prev3)  # self.layer3[i] 的输入是 prev3
            out5 = self.layer5[i](prev5)  # self.layer5[i] 的输入是 prev5

            # 交叉输入
            prev3 = torch.cat((out3, out5), dim=1)
            prev5 = torch.cat((out5, out3), dim=1)

        # 计算最终输出
        out1 = self.layer_out1(x0_1 + torch.mul(out3, out5))
        out2 = x1_1 + torch.mul(out3, out5)
        out2 = self.layer_out2(F.interpolate(out2, size=original_size, mode='bilinear', align_corners=True))

        return out1, out2

    # 边界信息的回归


# CABM 模块
# 通道注意力模块
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 自适应最大池化

        # 两个卷积层用于从池化后的特征中学习注意力权重
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)  # 第一个卷积层，降维
        self.relu1 = nn.ReLU()  # ReLU激活函数
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)  # 第二个卷积层，升维
        self.sigmoid = nn.Sigmoid()  # Sigmoid函数生成最终的注意力权重

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))  # 对平均池化的特征进行处理
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))  # 对最大池化的特征进行处理
        out = avg_out + max_out  # 将两种池化的特征加权和作为输出
        return self.sigmoid(out)  # 使用sigmoid激活函数计算注意力权重

class BR(nn.Module):
    def __init__(self, in_channel, kernel_size=7):
        super().__init__()

        self.att = ChannelAttention(in_channel)

    def forward(self, x, pre):
        x_att = x * self.att(x)
        pre_ra = 1 - torch.sigmoid(pre)

        return x_att * pre_ra



# 与上采样的结合模块
class DWF(nn.Module):
    def __init__(self, in_channel, kernel_size=7):
        super().__init__()
        self.conv1x1 = ConvBlock(in_channels=2 * in_channel, out_channels=in_channel, stride=1, kernel_size=1,
                                 padding=0)
        self.conv3x3_1 = ConvBlock(in_channels=in_channel, out_channels=in_channel // 2, stride=1, kernel_size=3,
                                   padding=1)

        self.conv3x3_2 = ConvBlock(in_channels=in_channel // 2, out_channels=2, stride=1, kernel_size=3, padding=1)

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x1, x2):
        feat = torch.cat([x1, x2], dim=1)
        feat = self.relu(self.conv1x1(feat))
        feat = self.relu(self.conv3x3_1(feat))
        att = self.conv3x3_2(feat)

        att_1 = torch.sigmoid(att[:, 0, :, :].unsqueeze(1))
        att_2 = torch.sigmoid(att[:, 1, :, :].unsqueeze(1))

        fusion_1_2 = att_1 * x1 + att_2 * x2

        return fusion_1_2


class JBDNet(nn.Module):
    def __init__(self):
        super().__init__()
        res2net = res2net50_v1b_26w_4s(pretrained=True)

        # Encoder
        self.encoder1_conv = res2net.conv1
        self.encoder1_bn = res2net.bn1
        self.encoder1_relu = res2net.relu
        self.maxpool = res2net.maxpool
        self.encoder2 = res2net.layer1
        self.encoder3 = res2net.layer2
        self.encoder4 = res2net.layer3
        self.encoder5 = res2net.layer4

        self.reduce2 = nn.Conv2d(256, 64, 1)
        self.reduce3 = nn.Conv2d(512, 128, 1)
        self.reduce4 = nn.Conv2d(1024, 256, 1)
        self.reduce5 = nn.Conv2d(2048, 512, 1)

        # 三重注意力模块
        self.bottom = TripletAttention()

        # CFF 模块
        self.layer12 = BIII(64, 64, 64)
        self.layer23 = BIII(64, 128, 64)
        self.layer34 = BIII(128, 256, 128)
        self.layer45 = BIII(256, 512, 256)

        # backInfoBounary 模块
        self.bi1 = BR(64)
        self.bi2 = BR(64)
        self.bi3 = BR(128)
        self.bi4 = BR(256)

        # scaleAware 模块
        self.sa1 = DWF(64)
        self.sa2 = DWF(128)
        self.sa3 = DWF(256)

        # decoder 模块
        self.de1 = DecoderBlock(64, 3)
        self.de2 = DecoderBlock(64, 64)
        self.de3 = DecoderBlock(128, 64)
        self.de4 = DecoderBlock(256, 128)
        self.de5 = DecoderBlock(512, 256)

        self.cat1 = nn.Conv2d(64 + 64, 64, 1)
        self.cat2 = nn.Conv2d(64 + 64, 64, 1)
        self.cat3 = nn.Conv2d(128 + 128, 128, 1)
        self.cat4 = nn.Conv2d(256 + 256, 256, 1)
        self.cat5 = nn.Conv2d(512 + 512, 512, 1)
        #     结果
        self.decode_head = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1),
        )

    def forward(self, x):
        e1_ = self.encoder1_conv(x)  # H/2*W/2*64
        e1_ = self.encoder1_bn(e1_)
        e1_ = self.encoder1_relu(e1_)
        e1_pool_ = self.maxpool(e1_)  # H/4*W/4*64
        e2_ = self.encoder2(e1_pool_)  # H/4*W/4*64
        e3_ = self.encoder3(e2_)  # H/8*W/8*128
        e4_ = self.encoder4(e3_)  # H/16*W/16*256
        e5_ = self.encoder5(e4_)  # H/32*W/32*512

        e1 = e1_
        e2 = self.reduce2(e2_)
        e3 = self.reduce3(e3_)
        e4 = self.reduce4(e4_)
        e5 = self.reduce5(e5_)

        bottom = self.bottom(e5)

        skip4_2, skip5 = self.layer45(e4, F.interpolate(e5, scale_factor=(2, 2), mode='bilinear', align_corners=True),
                                      original_size=e5.shape[2:])
        cat5 = self.cat5(torch.cat((skip5, bottom), dim=1))
        d5, pre5 = self.de5(cat5)

        skip3_2, skip4_1 = self.layer34(e3, F.interpolate(e4, scale_factor=(2, 2), mode='bilinear', align_corners=True),
                                        original_size=e4.shape[2:])
        sa3 = self.sa3(skip4_1, skip4_2)
        bi4 = self.bi4(sa3, pre5)
        cat4 = self.cat4(torch.cat((bi4, d5), dim=1))
        d4, pre4 = self.de4(cat4)

        skip2_2, skip3_1 = self.layer23(e2, F.interpolate(e3, scale_factor=(2, 2), mode='bilinear', align_corners=True),
                                        original_size=e3.shape[2:])
        sa2 = self.sa2(skip3_1, skip3_2)
        bi3 = self.bi3(sa2, pre4)
        cat3 = self.cat3(torch.cat((bi3, d4), dim=1))
        d3, pre3 = self.de3(cat3)

        skip1, skip2_1 = self.layer12(e1, F.interpolate(e2, scale_factor=(2, 2), mode='bilinear', align_corners=True),
                                      original_size=e2.shape[2:])
        sa1 = self.sa1(skip2_1, skip2_2)
        bi2 = self.bi2(sa1, pre3)
        cat2 = self.cat2(torch.cat((bi2, d3), dim=1))
        d2, pre2 = self.de2(cat2)

        bi1 = self.bi1(skip1, pre2)
        cat1 = self.cat1(torch.cat((bi1, d2), dim=1))
        d1, pre1 = self.de1(cat1)

        out = self.decode_head(d1)

        pre5 = F.interpolate(pre5, scale_factor=16, mode='bilinear', align_corners=True)
        pre4 = F.interpolate(pre4, scale_factor=8, mode='bilinear', align_corners=True)
        pre3 = F.interpolate(pre3, scale_factor=4, mode='bilinear', align_corners=True)
        pre2 = F.interpolate(pre2, scale_factor=2, mode='bilinear', align_corners=True)
        pre1 = F.interpolate(pre1, scale_factor=1, mode='bilinear', align_corners=True)

        # for i in (pre1,pre2,pre3,pre4,pre5):
        #     print(i.shape)

        return torch.sigmoid(out), torch.sigmoid(pre1), torch.sigmoid(pre2), torch.sigmoid(pre3), torch.sigmoid(
            pre4), torch.sigmoid(pre5)
        # return out, pre1, pre2, pre3, pre4, pre5



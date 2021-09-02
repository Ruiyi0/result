import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
#空间卷积金字塔池化
class ASPPModule(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,padding,dilation,BatchNorm):
        super().__init__()
        self.atrous_conv=nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,
                                   stride=1,padding=padding,dilation=dilation,bias=False)
        self.bn=BatchNorm(out_channels)
        self.relu=nn.ReLU()
        self._init_weight()

    def forward(self,x):
        x=self.atrous_conv(x)
        x=self.bn(x)
        x=self.relu(x)
        return  x
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m,SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m,nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self,backbone,output_stride,BatchNorm):
        super().__init__()
        #根据backbone确定输入通道数
        if backbone == 'drn':
            in_channels=512
        elif backbone == 'mobilenet':
            in_channels = 320
        else:
            in_channels=2048
        if output_stride == 16:
            dilations = [1,6,12,18]
        elif output_stride == 8:
            dilations = [1,12,24,36]
        else:
            raise NotImplementedError

        self.aspp1 = ASPPModule(in_channels=in_channels,out_channels=256,kernel_size=1,padding=0,
                                dilation=dilations[0],BatchNorm=BatchNorm)
        self.aspp2 = ASPPModule(in_channels=in_channels,out_channels=256,kernel_size=3,padding=dilations[1],
                                dilation=dilations[1],BatchNorm=BatchNorm)
        self.aspp3 = ASPPModule(in_channels=in_channels,out_channels=256,kernel_size=3,padding=dilations[2],
                                dilation=dilations[2],BatchNorm=BatchNorm)
        self.aspp4 = ASPPModule(in_channels=in_channels,out_channels=256,kernel_size=3,padding=dilations[3],
                                dilation=dilations[3],BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),#outputsize=(1,1)
            nn.Conv2d(in_channels,256,kernel_size=1,stride=1,bias=False),
            BatchNorm(256),
            nn.ReLU()
        )

        #输入阶段1x1的卷积
        self.conv1 = nn.Conv2d(in_channels=1280,out_channels=256,kernel_size=1,bias=False)
        self.bn1 = BatchNorm(256)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self._init_weight()

    def forward(self,x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        #x5阶段插值 size为x4的h和w
        x5 = F.interpolate(x5,size=x4.size()[2:],mode='bilinear',align_corners=True)
        x = torch.cat((x1,x3,x4,x4,x5),dim=1) #在通道维上合并

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m,SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m,nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def build_aspp(backbone,output_stride,BatchNorm):
    return ASPP(backbone,output_stride,BatchNorm)



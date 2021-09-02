import torch
import torch.nn as nn
import torch.nn.functional as F
from sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from ASPP import build_aspp
from Decoder import build_decoder
from backbone import build_backbone
from dataset_ import BasicDataset

class DeepLab(nn.Module):
    def __init__(self,backbone='resnet',output_stride=16,num_classes=1,
                 sync_bn=True,freeze_bn=False):
        super().__init__()
        #num_classes为最后所分类别数
        if backbone == 'drn':
            output_stride = 8
        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone=backbone,output_stride=output_stride,BatchNorm=BatchNorm)
        self.aspp = build_aspp(backbone=backbone,output_stride=output_stride,BatchNorm=BatchNorm)
        self.decoder = build_decoder(num_classes=num_classes,backbone=backbone,BatchNorm=BatchNorm)


        self.freeze_bn = freeze_bn
    def forward(self,input):
        x,low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x,low_level_feat)
        #上采样到与原图同大小
        x = F.interpolate(x,size=input.size()[2:],mode='bilinear',align_corners=True)
        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m,SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m,nn.BatchNorm2d):
                m.eval()
    #获取conv1及lr参数
    def get_1x_lr_params(self):
        modules=[self.backbone]
        #len(modules)为所采用的backbone数
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1],nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1],nn.Conv2d) or isinstance(m[1],SynchronizedBatchNorm2d)\
                        or isinstance(m[1],nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = [self.assp,self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1],nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1],nn.Conv2d) or isinstance(m[1].SynchronizedBatchNorm2d)\
                        or isinstance(m[1],nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p



if __name__ == '__main__':

    model = DeepLab(backbone='mobilenet',output_stride=16)
    model.eval()
    output = model(torch.rand(1,3,513,513))
    print(output.shape)

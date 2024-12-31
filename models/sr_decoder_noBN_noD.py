import torch
import torch.nn as nn
import torch.nn.functional as F
from models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
#from models.common import AttentionModel
################lzy NAM#########################
#NAM
class NAM(nn.Module):
    def __init__(self, channels, t=16):
        super(NAM, self).__init__()
        self.channels = channels
        self.bn2 = nn.BatchNorm2d(self.channels, affine=True)
 
    def forward(self, x):
        residual = x
        x = self.bn2(x)
        weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = torch.sigmoid(x) * residual  #
        return x
class Decoder(nn.Module):
    def __init__(self, c1,c2):
        super(Decoder, self).__init__()
        
        self.conv1 = nn.Conv2d(c1, c1//2, 1, bias=False)
        self.conv2 = nn.Conv2d(c2, c2//2, 1, bias=False)
        #self.bn1 = BatchNorm(48)
        self.relu = nn.ReLU()
        #self.pixel_shuffle = nn.PixelShuffle(4)
        #self.attention = AttentionModel(48+c2)
        self.last_conv = nn.Sequential(nn.Conv2d((c1+c2)//2, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       #BatchNorm(256),
                                       nn.ReLU(),
                                       #nn.Dropout(0.5),
                                       nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
                                       #BatchNorm(128),
                                       nn.ReLU(),
                                       #nn.Dropout(0.1),
                                       nn.Conv2d(128, 64, kernel_size=1, stride=1))
        self._init_weight()
        self.NAM_low=NAM(c1)
        self.NAM_high=NAM(c2)
    def forward(self, x, low_level_feat,factor):
        #x, low_level_feat = input[-2],input[-1]
        low_level_feat=self.NAM_low(low_level_feat)
        low_level_feat = self.conv1(low_level_feat)
        #low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat) 

        #x = F.interpolate(x, size=[i*2 for i in low_level_feat.size()[2:]], mode='bilinear', align_corners=True)
        #low_level_feat = F.interpolate(low_level_feat, size=[i*2 for i in low_level_feat.size()[2:]], mode='bilinear', align_corners=True)
        #x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x=self.NAM_high(x)
        x = self.conv2(x)
        x = self.relu(x) 
        x = F.interpolate(x, size=[i*(factor//2) for i in low_level_feat.size()[2:]], mode='bilinear', align_corners=True)
        if factor>1:
            low_level_feat = F.interpolate(low_level_feat, size=[i*(factor//2) for i in low_level_feat.size()[2:]], mode='bilinear', align_corners=True)
        #x = self.pixel_shuffle(x)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)

        return x
    # def forward(self, x, low_level_feat):
    #     low_level_feat = self.conv1(low_level_feat)
    #     low_level_feat = self.bn1(low_level_feat)
    #     low_level_feat = self.relu(low_level_feat) 

    #     #x = F.interpolate(x, size=[i*2 for i in low_level_feat.size()[2:]], mode='bilinear', align_corners=True)
    #     #low_level_feat = F.interpolate(low_level_feat, size=[i*2 for i in low_level_feat.size()[2:]], mode='bilinear', align_corners=True)
    #     x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
    #     #x = self.pixel_shuffle(x)
    #     x = torch.cat((x, low_level_feat), dim=1)
    #     attention_map,x_sr_attention = self.attention(x)
    #     x = self.last_conv(x_sr_attention)

    #     return x,attention_map

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

# def build_sr_decoder(BatchNorm):
#     return Decoder(BatchNorm)
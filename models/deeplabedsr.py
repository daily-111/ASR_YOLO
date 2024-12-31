import torch
import torch.nn as nn
import torch.nn.functional as F

from models.sr_decoder_noBN_noD import Decoder
from models.edsr import EDSR

from models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
# class AttentionModel(nn.module):
#     def __init__(self,feature_in):
#         self.conv = nn.conv2d(feature_in,1,kernel_size=3, padding=1)
#         self.output_act = nn.Sigmoid()
#     def foward(self,x):
#         x = self.conv(x)
#         attention_map = self.output_act(x)
#         output = x + x * torch.exp(attention_map)
#         return attention_map,output

class EDSRConv(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super(EDSRConv, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_ch, out_ch, 3, padding=1),
            )

        self.residual_upsampler = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            )

    def forward(self, input):
        return self.conv(input)+self.residual_upsampler(input)


class DeepLab(nn.Module):
    def __init__(self, ch, c1=128, c2=512,factor=2, sync_bn=True, freeze_bn=False):
        super(DeepLab, self).__init__()

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        #self.attention = AttentionModel(128)
        self.sr_decoder = Decoder(c1,c2)
        self.edsr = EDSR(num_channels=ch,input_channel=64, factor=8)
        # self.up_sr_1 = nn.ConvTranspose2d(64, 64, 2, stride=2) 
        # self.up_edsr_1 = EDSRConv(64,64)
        # self.up_sr_2 = nn.ConvTranspose2d(64, 32, 2, stride=2) 
        # self.up_edsr_2 = EDSRConv(32,32)
        # self.up_sr_3 = nn.ConvTranspose2d(32, 16, 2, stride=2) 
        # self.up_edsr_3 = EDSRConv(16,16)
        # self.up_conv_last = nn.Conv2d(16,ch,1)
        self.factor = factor


        # self.freeze_bn = freeze_bn

    def forward(self, low_level_feat,x):
        x_sr= self.sr_decoder(x, low_level_feat,self.factor)

        
        x_sr_up = self.edsr(x_sr)

      
        #attention_map,x_sr = self.attention(x_sr)
        # x_sr_up = self.up_sr_1(x_sr)
        # x_sr_up=self.up_edsr_1(x_sr_up)

        # x_sr_up = self.up_sr_2(x_sr_up)
        # x_sr_up=self.up_edsr_2(x_sr_up)

        # x_sr_up = self.up_sr_3(x_sr_up)
        # x_sr_up=self.up_edsr_3(x_sr_up)
        # x_sr_up=self.up_conv_last(x_sr_up)

        return x_sr_up

    # def freeze_bn(self):
    #     for m in self.modules():
    #         if isinstance(m, SynchronizedBatchNorm2d):
    #             m.eval()
    #         elif isinstance(m, nn.BatchNorm2d):
    #             m.eval()
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm

@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
def pixel_unshuffle(x, scale):
    """ Pixel unshuffle.

    Args:
        x (Tensor): Input feature with shape (b, c, hh, hw).
        scale (int): Downsample ratio.

    Returns:
        Tensor: the pixel unshuffled feature.
    """
    b, c, hh, hw = x.size()
    out_channel = c * (scale**2)
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)
def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)

class ResidualDenseBlock(nn.Module):
    """Residual Dense Block.

    Used in RRDB block in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Empirically, we use 0.2 to scale the residual for better performance
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block.

    Used in RRDB-Net in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Empirically, we use 0.2 to scale the residual for better performance
        return out * 0.2 + x



class RRDBNet(nn.Module):
    """Networks consisting of Residual in Residual Dense Block, which is used
    in ESRGAN.

    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.

    We extend ESRGAN for scale x2 and scale x1.
    Note: This is one option for scale 1, scale 2 in RRDBNet.
    We first employ the pixel-unshuffle (an inverse operation of pixelshuffle to reduce the spatial size
    and enlarge the channel size before feeding inputs into the main ESRGAN architecture.

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64
        num_block (int): Block number in the trunk network. Defaults: 23
        num_grow_ch (int): Channels for each growth. Default: 32.
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, scale=2, num_feat=64, num_block=14, num_grow_ch=32):
        super(RRDBNet, self).__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        # print("feat:",feat.shape)
        body_feat = self.conv_body(self.body(feat))
        # print("body_feat:",body_feat.shape)
        feat = feat + body_feat
        # print(" feat + body_feat:",feat.shape)
        # upsample
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        # print(" feat:",feat.shape)
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        # print(" feat:",feat.shape)
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        # print(" feat:",feat.shape)
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        # print(" out:",out.shape)
        return out

class DeepLab_lzy(nn.Module):
    def __init__(self, ch, c1=128, c2=512,factor=2, sync_bn=True, freeze_bn=False):
        super(DeepLab_lzy, self).__init__()

        # if sync_bn == True:
        #     BatchNorm = SynchronizedBatchNorm2d
        # else:
        #     BatchNorm = nn.BatchNorm2d

        #self.attention = AttentionModel(128)
        self.sr_decoder = Decoder(c1,c2)
        self.edsr = RRDBNet(num_in_ch=64,num_out_ch=ch,scale=8)
        # self.up_sr_1 = nn.ConvTranspose2d(64, 64, 2, stride=2) 
        # self.up_edsr_1 = EDSRConv(64,64)
        # self.up_sr_2 = nn.ConvTranspose2d(64, 32, 2, stride=2) 
        # self.up_edsr_2 = EDSRConv(32,32)
        # self.up_sr_3 = nn.ConvTranspose2d(32, 16, 2, stride=2) 
        # self.up_edsr_3 = EDSRConv(16,16)
        # self.up_conv_last = nn.Conv2d(16,ch,1)
        self.factor = factor


        # self.freeze_bn = freeze_bn

    def forward(self, low_level_feat,x):
        # print('x:',x.shape)

        x_sr= self.sr_decoder(x, low_level_feat,self.factor)
        # print('x_sr:',x_sr.shape)
        x_sr_up = self.edsr(x_sr)
        # print('x_sr_up:',x_sr_up.shape)
        #attention_map,x_sr = self.attention(x_sr)
        # x_sr_up = self.up_sr_1(x_sr)
        # x_sr_up=self.up_edsr_1(x_sr_up)

        # x_sr_up = self.up_sr_2(x_sr_up)
        # x_sr_up=self.up_edsr_2(x_sr_up)

        # x_sr_up = self.up_sr_3(x_sr_up)
        # x_sr_up=self.up_edsr_3(x_sr_up)
        # x_sr_up=self.up_conv_last(x_sr_up)

        return x_sr_up


#################CARRDRB###############################
import torch
from torch import nn as nn
from torch.nn import functional as F





class OurResidualDenseBlock(nn.Module):
    """Residual Dense Block.

    Used in RRDB block in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    # def __init__(self, num_feat=64, num_grow_ch=32):
    #     super(ResidualDenseBlock, self).__init__()
    #     self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
    #     self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
    #     self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
    #     self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
    #     self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

    #     self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    #     # initialization
    #     default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    # def forward(self, x):
    #     x1 = self.lrelu(self.conv1(x))
    #     x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
    #     x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
    #     x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
    #     x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
    #     # Empirically, we use 0.2 to scale the residual for better performance
    #     return x5 * 0.2 + x
    def __init__(self, num_feat=64, num_grow_ch=32,gaussian_noise=True):
        super(OurResidualDenseBlock, self).__init__()
        self.noise = GaussianNoise() if gaussian_noise else None
        self.conv1x1 = conv1x1(num_feat,num_feat)
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # self.trans1 = CotLayer(num_feat, kernel_size=3)
        # self.conv1 = nn.Conv2d(num_feat, num_grow_ch, kernel_size=1, stride=1, bias=True)
        # self.trans2 = CotLayer(num_feat + num_grow_ch, kernel_size=3)
        # self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, kernel_size=1, stride=1, bias=True)
        # self.trans3 = CotLayer(num_feat + 2 * num_grow_ch, kernel_size=3)
        # self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, kernel_size=1, stride=1, bias=True)
        self.conv3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # self.trans4 = CotLayer(num_feat + 3 * num_grow_ch, kernel_size=3)
        # self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, kernel_size=1, stride=1, bias=True)
        self.conv5 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # self.trans5 = CotLayer(num_feat + 4 * num_grow_ch, kernel_size=3)
        # self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_grow_ch, kernel_size=1, stride=1, bias=True)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.CA = ChannelAttention(num_feat,squeeze_factor=16)
        
        # initialization
        default_init_weights([self.CA,self.noise,self.conv1x1,self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.CA(x)
        x1 = self.lrelu(self.conv1(x1))
        # x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x2 = self.lrelu(self.conv2(x1))
        x2 = x2 + self.conv1x1(x)
        # x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x3 = self.lrelu(self.conv3(x2))
        # x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x4 = self.lrelu(self.conv4(x3))
        
        x5 = self.conv5(x4)
        
        
        # Empirically, we use 0.2 to scale the residual for better performance
        return self.noise(x5 * 0.2 + x )

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out

class OurRRDB(nn.Module):
    """Residual in Residual Dense Block.

    Used in RRDB-Net in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat, num_grow_ch=32):
        super(OurRRDB, self).__init__()
        # self.attention = ChannelAttention(num_feat,squeeze_factor=16)
        
        self.rdb1 = OurResidualDenseBlock(num_feat, num_grow_ch)

        self.rdb2 = OurResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = OurResidualDenseBlock(num_feat, num_grow_ch)
        # self.attention = ChannelAttention(num_feat,squeeze_factor=16)
    def forward(self, x):
        # out = self.attention(x)
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # out = self.attention(out)
        # Empirically, we use 0.2 to scale the residual for better performance
        return out * 0.2 + x



class OurRRDBNet(nn.Module):
    """Networks consisting of Residual in Residual Dense Block, which is used
    in ESRGAN.

    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.

    We extend ESRGAN for scale x2 and scale x1.
    Note: This is one option for scale 1, scale 2 in RRDBNet.
    We first employ the pixel-unshuffle (an inverse operation of pixelshuffle to reduce the spatial size
    and enlarge the channel size before feeding inputs into the main ESRGAN architecture.

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64
        num_block (int): Block number in the trunk network. Defaults: 23
        num_grow_ch (int): Channels for each growth. Default: 32.
    """

    def __init__(self, num_in_ch, num_out_ch, scale=8, num_feat=64, num_block=16, num_grow_ch=32):
        super(OurRRDBNet, self).__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(OurRRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        # upsample
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out



###################################
######con1x1模块####################
##################################
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



#############GaussianNoise模块###############
#############################################
#############################################
class GaussianNoise(nn.Module):
    def __init__(self, sigma=0.1, is_relative_detach=False):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0, dtype=torch.float).to(torch.device('cuda'))

    def forward(self, x):
        device= x.device
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale.to(torch.device('cuda'))
            x = x.to(torch.device('cuda')) + sampled_noise
            x = x.to(device)
        return x 





#################CAB_attention#########################
######################################################

class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
       
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class CAB(nn.Module):

    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30):
        super(CAB, self).__init__()

        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
            )

    def forward(self, x):
        return self.cab(x)

class DeepLab_lzy_CARRDRB(nn.Module):
    def __init__(self, ch, c1=128, c2=512,factor=2, sync_bn=True, freeze_bn=False):
        super(DeepLab_lzy_CARRDRB, self).__init__()

    
        self.sr_decoder = Decoder(c1,c2)
        self.edsr = OurRRDBNet(num_in_ch=64,num_out_ch=ch,scale=8)
        self.factor = factor




    def forward(self, low_level_feat,x):
   
        x_sr= self.sr_decoder(x, low_level_feat,self.factor)
      
        x_sr_up = self.edsr(x_sr)
       

        return x_sr_up
if __name__ == "__main__":
    # model = DeepLab(backbone='mobilenet', output_stride=16)
    model=RRDBNet()
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())



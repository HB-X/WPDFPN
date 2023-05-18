#import torch
#import torch.nn as nn
#import torch.nn.functional as F
#from mmcv.cnn import ConvModule, xavier_init
#
#from mmcv.runner import auto_fp16
#
#
#class SELayer(nn.Module):
#    def __init__(self, channel, reduction=16):
#        super(SELayer, self).__init__()
#        self.avg_pool = nn.AdaptiveAvgPool2d(1)
#        self.fc = nn.Sequential(
#            nn.Linear(channel, channel // reduction, bias=False),
#            nn.ReLU(inplace=True),
#            nn.Linear(channel // reduction, channel, bias=False),
#            nn.Sigmoid()
#        )
#
#    def forward(self, x):
#        b, c, _, _ = x.size()
#        y = self.avg_pool(x).view(b, c)
#        y = self.fc(y).view(b, c, 1, 1)
#        return x * y.expand_as(x)
#
#
#class MGF_1(nn.Module):
#    def __init__(self, in_channels=256, compressed_channels=256, out_channels=256,
#                 kernel_size=3, upsample_cfg=dict(mode='nearest')):
#        super(MGF_1, self).__init__()
#        self.in_channels = in_channels
#        self.out_channels = out_channels
#        self.kernel_size = kernel_size
#        self.compressed_channels = compressed_channels
#        self.upsample_cfg = upsample_cfg.copy()
#        self.SE_layer = SELayer(channel=in_channels)
#        self.encoder = nn.Sequential(
#            nn.Conv2d(in_channels=self.in_channels, out_channels=self.compressed_channels,
#                      kernel_size=self.kernel_size, padding=self.kernel_size // 2),
#            nn.ReLU(inplace=True),
#            nn.Conv2d(in_channels=self.compressed_channels, out_channels=self.compressed_channels,
#                      kernel_size=self.kernel_size, padding=self.kernel_size // 2),
#            nn.ReLU(inplace=True),
#            nn.Conv2d(in_channels=self.compressed_channels, out_channels=self.compressed_channels,
#                      kernel_size=self.kernel_size, padding=self.kernel_size // 2),
#            nn.ReLU(inplace=True),
#            nn.Conv2d(in_channels=self.compressed_channels, out_channels=self.compressed_channels,
#                      kernel_size=self.kernel_size, padding=self.kernel_size // 2),
#            nn.ReLU(inplace=True)
#        )
#
#        self.mask_yx = nn.Sequential(
#            nn.Conv2d(in_channels=self.compressed_channels, out_channels=self.in_channels,
#                      kernel_size=self.kernel_size, padding=self.kernel_size // 2),
#            nn.BatchNorm2d(self.in_channels),
#            nn.Sigmoid())
#
#        self.mask_xy = nn.Sequential(
#            nn.Conv2d(in_channels=self.compressed_channels, out_channels=self.in_channels,
#                      kernel_size=self.kernel_size, padding=self.kernel_size // 2),
#            nn.BatchNorm2d(self.in_channels),
#            nn.Sigmoid())
#
#    def forward(self, x, y):
#        if 'scale_factor' in self.upsample_cfg:
#            upsampled_x = F.interpolate(x, **self.upsample_cfg)
#        else:
#            upsampled_x = F.interpolate(x, size=y.shape[2:], **self.upsample_cfg)
#        encoded = self.encoder(upsampled_x + y + self.SE_layer(upsampled_x + y))
##        encoded = self.encoder(upsampled_x + y)
#        enhanced_x = x + F.adaptive_max_pool2d(self.mask_yx(encoded) * y,
#                                               output_size=x.shape[2:])
#        enhanced_y = y + self.mask_xy(encoded) * upsampled_x
#
#        return enhanced_x, enhanced_y
#
#
#class MGF_2(nn.Module):
#    def __init__(self, in_channels=256, compressed_channels=256, out_channels=256,
#                 kernel_size=3, upsample_cfg=dict(mode='nearest')):
#        super(MGF_2, self).__init__()
#        self.in_channels = in_channels
#        self.out_channels = out_channels
#        self.compressed_channels = compressed_channels
#        self.kernel_size = kernel_size
#        self.upsample_cfg = upsample_cfg.copy()
#        self.SE_layer = SELayer(channel=in_channels)
#        self.encoder = nn.Sequential(
#            nn.Conv2d(in_channels=self.in_channels, out_channels=self.compressed_channels,
#                      kernel_size=self.kernel_size, padding=self.kernel_size // 2),
#            nn.ReLU(inplace=True),
#            nn.Conv2d(in_channels=self.compressed_channels, out_channels=self.compressed_channels,
#                      kernel_size=self.kernel_size, padding=self.kernel_size // 2),
#            nn.ReLU(inplace=True),
#            nn.Conv2d(in_channels=self.compressed_channels, out_channels=self.compressed_channels,
#                      kernel_size=self.kernel_size, padding=self.kernel_size // 2),
#            nn.ReLU(inplace=True),
#            nn.Conv2d(in_channels=self.compressed_channels, out_channels=self.compressed_channels,
#                      kernel_size=self.kernel_size, padding=self.kernel_size // 2),
#            nn.ReLU(inplace=True)
#        )
#
#        self.mask_yx = nn.Sequential(
#            nn.Conv2d(in_channels=self.compressed_channels, out_channels=self.in_channels,
#                      kernel_size=self.kernel_size, padding=self.kernel_size // 2),
#            nn.BatchNorm2d(self.in_channels),
#            nn.Sigmoid())
#
#        self.mask_xy = nn.Sequential(
#            nn.Conv2d(in_channels=self.compressed_channels, out_channels=self.in_channels,
#                      kernel_size=self.kernel_size, padding=self.kernel_size // 2),
#            nn.BatchNorm2d(self.in_channels),
#            nn.Sigmoid())
#
#    def forward(self, x, y):
#        if 'scale_factor' in self.upsample_cfg:
#            upsampled_x = F.interpolate(x, **self.upsample_cfg)
#        else:
#            upsampled_x = F.interpolate(x, size=y.shape[2:], **self.upsample_cfg)
#        encoded = self.encoder(upsampled_x + y + self.SE_layer(upsampled_x + y))
##        encoded = self.encoder(upsampled_x + y)
#        enhanced_x = x + F.adaptive_max_pool2d(self.mask_yx(encoded) * y,
#                                               output_size=x.shape[2:])
#        enhanced_y = y + self.mask_xy(encoded) * upsampled_x
#
#        return enhanced_x, enhanced_y
#
#
#class MGF_3(nn.Module):
#    def __init__(self, in_channels=256, compressed_channels=256, out_channels=256,
#                 kernel_size=3, upsample_cfg=dict(mode='nearest')):
#        super(MGF_3, self).__init__()
#        self.in_channels = in_channels
#        self.out_channels = out_channels
#        self.compressed_channels = compressed_channels
#        self.kernel_size = kernel_size
#        self.upsample_cfg = upsample_cfg.copy()
#        self.SE_layer = SELayer(channel=in_channels)
#        self.encoder = nn.Sequential(
#            nn.Conv2d(in_channels=self.in_channels, out_channels=self.compressed_channels,
#                      kernel_size=self.kernel_size, padding=self.kernel_size // 2),
#            nn.ReLU(inplace=True),
#            nn.Conv2d(in_channels=self.compressed_channels, out_channels=self.compressed_channels,
#                      kernel_size=self.kernel_size, padding=self.kernel_size // 2),
#            nn.ReLU(inplace=True),
#            nn.Conv2d(in_channels=self.compressed_channels, out_channels=self.compressed_channels,
#                      kernel_size=self.kernel_size, padding=self.kernel_size // 2),
#            nn.ReLU(inplace=True),
#            nn.Conv2d(in_channels=self.compressed_channels, out_channels=self.compressed_channels,
#                      kernel_size=self.kernel_size, padding=self.kernel_size // 2),
#            nn.ReLU(inplace=True)
#        )
#
#        self.mask_yx = nn.Sequential(
#            nn.Conv2d(in_channels=self.compressed_channels, out_channels=self.in_channels,
#                      kernel_size=self.kernel_size, padding=self.kernel_size // 2),
#            nn.BatchNorm2d(self.in_channels),
#            nn.Sigmoid())
#
#        self.mask_xy = nn.Sequential(
#            nn.Conv2d(in_channels=self.compressed_channels, out_channels=self.in_channels,
#                      kernel_size=self.kernel_size, padding=self.kernel_size // 2),
#            nn.BatchNorm2d(self.in_channels),
#            nn.Sigmoid())
#
#    def forward(self, x, y):
#        if 'scale_factor' in self.upsample_cfg:
#            upsampled_x = F.interpolate(x, **self.upsample_cfg)
#        else:
#            upsampled_x = F.interpolate(x, size=y.shape[2:], **self.upsample_cfg)
#        encoded = self.encoder(upsampled_x + y + self.SE_layer(upsampled_x + y))
##        encoded = self.encoder(upsampled_x + y)
#        enhanced_x = x + F.adaptive_max_pool2d(self.mask_yx(encoded) * y,
#                                               output_size=x.shape[2:])
#        enhanced_y = y + self.mask_xy(encoded) * upsampled_x
#
#        return enhanced_x, enhanced_y
#
#
#from ..builder import NECKS
##
##
#@NECKS.register_module()
#class MGF_FPN(nn.Module):
#    def __init__(self,
#                 in_channels,
#                 out_channels,
#                 num_outs,
#                 start_level=0,
#                 end_level=-1,
#                 no_norm_on_lateral=False,
#                 conv_cfg=None,
#                 norm_cfg=None,
#                 act_cfg=None,
#                 upsample_cfg=dict(mode='nearest')):
#        super(MGF_FPN, self).__init__()
#        assert isinstance(in_channels, list)
#        self.in_channels = in_channels
#        self.out_channels = out_channels
#        self.num_ins = len(in_channels)
#        self.num_outs = num_outs
#        self.no_norm_on_lateral = no_norm_on_lateral
#        self.fp16_enabled = False
#        self.upsample_cfg = upsample_cfg.copy()
#        self.mutually_guided_filtering = nn.ModuleList()
#        self.mutually_guided_filtering = self.mutually_guided_filtering.append(
#            MGF_1().cuda()).append(MGF_2().cuda()).append(MGF_3().cuda())
#
#        if end_level == -1:
#            self.backbone_end_level = self.num_ins
#            assert num_outs >= self.num_ins - start_level
#        else:
#            # if end_level < inputs, no extra level is allowed
#            self.backbone_end_level = end_level
#            assert end_level <= len(in_channels)
#            assert num_outs == end_level - start_level
#        self.start_level = start_level
##        self.SE_Layers = nn.ModuleList()
#        self.lateral_convs = nn.ModuleList()
#
#
#        kernel_size = [1, 3, 5, 7]
#        dilation = [1, 1, 1, 1]
#        padding = [0, 1, 2, 3]
#
#        for i in range(self.start_level, self.backbone_end_level):
#            l_conv = ConvModule(
#                in_channels[i],
#                out_channels,
#                kernel_size[i],
#                padding=padding[i],
#                dilation=dilation[i],
#                conv_cfg=conv_cfg,
#                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
#                act_cfg=act_cfg,
#                inplace=False)
##            se_layer = SELayer(channel=in_channels[i])
#
#            self.lateral_convs.append(l_conv)
##            self.SE_Layers.append(se_layer)
#
#    # default init_weights for conv(msra) and norm in ConvModule
#    def init_weights(self):
#        """Initialize the weights of FPN module."""
#        for m in self.modules():
#            if isinstance(m, nn.Conv2d):
#                xavier_init(m, distribution='uniform')
#
#    @auto_fp16()
#    def forward(self, inputs):
#        """Forward function."""
#        assert len(inputs) == len(self.in_channels)
#
##        # build se_layer
##        inputs = [inputs[i + self.start_level] + Se_layer(inputs[i + self.start_level])
##                  for i, Se_layer in enumerate(self.SE_Layers)]
#
#        # build laterals
#        laterals = [
#            lateral_conv(inputs[i + self.start_level])
#            for i, lateral_conv in enumerate(self.lateral_convs)
#        ]
#
#        # build top-down path
#        used_backbone_levels = len(laterals)
#        for i in range(used_backbone_levels - 1, 0, -1):
#            laterals[i], laterals[i - 1] = self.mutually_guided_filtering[i - 1](laterals[i], laterals[i - 1])
#
#        laterals.append(F.max_pool2d(laterals[-1], 1, stride=2))
#
#        return tuple(laterals)
#


#import torch
#import torch.nn as nn
#import torch.nn.functional as F
#from mmcv.cnn import ConvModule, xavier_init
#
#from mmcv.runner import auto_fp16
#
#
#class SELayer(nn.Module):
#    def __init__(self, channel, reduction=16):
#        super(SELayer, self).__init__()
#        self.avg_pool = nn.AdaptiveAvgPool2d(1)
#        self.fc = nn.Sequential(
#            nn.Linear(channel, channel // reduction, bias=False),
#            nn.ReLU(inplace=True),
#            nn.Linear(channel // reduction, channel, bias=False),
#            nn.Sigmoid()
#        )
#
#    def forward(self, x):
#        b, c, _, _ = x.size()
#        y = self.avg_pool(x).view(b, c)
#        y = self.fc(y).view(b, c, 1, 1)
#        return x * y.expand_as(x)
#
#
#class MGF_1(nn.Module):
#    def __init__(self, in_channels=256, compressed_channels=256, out_channels=256,
#                 kernel_size=3, upsample_cfg=dict(mode='nearest')):
#        super(MGF_1, self).__init__()
#        self.in_channels = in_channels
#        self.out_channels = out_channels
#        self.kernel_size = kernel_size
#        self.compressed_channels = compressed_channels
#        self.upsample_cfg = upsample_cfg.copy()
#        self.SE_layer = SELayer(channel=in_channels)
#        self.encoder = nn.Sequential(
#            nn.Conv2d(in_channels=self.in_channels, out_channels=self.compressed_channels,
#                      kernel_size=self.kernel_size, padding=self.kernel_size // 2),
#            nn.ReLU(inplace=True),
#            nn.Conv2d(in_channels=self.compressed_channels, out_channels=self.compressed_channels,
#                      kernel_size=self.kernel_size, padding=self.kernel_size // 2),
#            nn.ReLU(inplace=True),
#            nn.Conv2d(in_channels=self.compressed_channels, out_channels=self.compressed_channels,
#                      kernel_size=self.kernel_size, padding=self.kernel_size // 2),
#            nn.ReLU(inplace=True),
#            nn.Conv2d(in_channels=self.compressed_channels, out_channels=self.compressed_channels,
#                      kernel_size=self.kernel_size, padding=self.kernel_size // 2),
#            nn.ReLU(inplace=True)
#        )

#        self.mask_yx = nn.Sequential(
#            nn.Conv2d(in_channels=self.compressed_channels, out_channels=self.in_channels,
#                      kernel_size=self.kernel_size, padding=self.kernel_size // 2),
#            nn.BatchNorm2d(self.in_channels),
#            nn.Sigmoid())
#
#        self.mask_xy = nn.Sequential(
#            nn.Conv2d(in_channels=self.compressed_channels, out_channels=self.in_channels,
#                      kernel_size=self.kernel_size, padding=self.kernel_size // 2),
#            nn.BatchNorm2d(self.in_channels),
#            nn.Sigmoid())
#
#    def forward(self, x, y):
#        if 'scale_factor' in self.upsample_cfg:
#            upsampled_x = F.interpolate(x, **self.upsample_cfg)
#        else:
#            upsampled_x = F.interpolate(x, size=y.shape[2:], **self.upsample_cfg)
#        encoded = self.encoder(upsampled_x + y + self.SE_layer(upsampled_x + y))
#        enhanced_x = x + F.adaptive_max_pool2d(self.mask_yx(encoded) * y,
#                                               output_size=x.shape[2:])
#        enhanced_y = y + self.mask_xy(encoded) * upsampled_x
#
#        return enhanced_x, enhanced_y
#
#
#class MGF_2(nn.Module):
#    def __init__(self, in_channels=256, compressed_channels=256, out_channels=256,
#                 kernel_size=3, upsample_cfg=dict(mode='nearest')):
#        super(MGF_2, self).__init__()
#        self.in_channels = in_channels
#        self.out_channels = out_channels
#        self.compressed_channels = compressed_channels
#        self.kernel_size = kernel_size
#        self.upsample_cfg = upsample_cfg.copy()
#        self.SE_layer = SELayer(channel=in_channels)
#        self.encoder = nn.Sequential(
#            nn.Conv2d(in_channels=self.in_channels, out_channels=self.compressed_channels,
#                      kernel_size=self.kernel_size, padding=self.kernel_size // 2),
#            nn.ReLU(inplace=True),
#            nn.Conv2d(in_channels=self.compressed_channels, out_channels=self.compressed_channels,
#                      kernel_size=self.kernel_size, padding=self.kernel_size // 2),
#            nn.ReLU(inplace=True),
#            nn.Conv2d(in_channels=self.compressed_channels, out_channels=self.compressed_channels,
#                      kernel_size=self.kernel_size, padding=self.kernel_size // 2),
#            nn.ReLU(inplace=True),
#            nn.Conv2d(in_channels=self.compressed_channels, out_channels=self.compressed_channels,
#                      kernel_size=self.kernel_size, padding=self.kernel_size // 2),
#            nn.ReLU(inplace=True)
#        )
#
#        self.mask_yx = nn.Sequential(
#            nn.Conv2d(in_channels=self.compressed_channels, out_channels=self.in_channels,
#                      kernel_size=self.kernel_size, padding=self.kernel_size // 2),
#            nn.BatchNorm2d(self.in_channels),
#            nn.Sigmoid())
#
#        self.mask_xy = nn.Sequential(
#            nn.Conv2d(in_channels=self.compressed_channels, out_channels=self.in_channels,
#                      kernel_size=self.kernel_size, padding=self.kernel_size // 2),
#            nn.BatchNorm2d(self.in_channels),
#            nn.Sigmoid())
#
#    def forward(self, x, y):
#        if 'scale_factor' in self.upsample_cfg:
#            upsampled_x = F.interpolate(x, **self.upsample_cfg)
#        else:
#            upsampled_x = F.interpolate(x, size=y.shape[2:], **self.upsample_cfg)
#        encoded = self.encoder(upsampled_x + y + self.SE_layer(upsampled_x + y))
#        enhanced_x = x + F.adaptive_max_pool2d(self.mask_yx(encoded) * y,
#                                               output_size=x.shape[2:])
#        enhanced_y = y + self.mask_xy(encoded) * upsampled_x
#
#        return enhanced_x, enhanced_y
#
#
#class MGF_3(nn.Module):
#    def __init__(self, in_channels=256, compressed_channels=256, out_channels=256,
#                 kernel_size=3, upsample_cfg=dict(mode='nearest')):
#        super(MGF_3, self).__init__()
#        self.in_channels = in_channels
#        self.out_channels = out_channels
#        self.compressed_channels = compressed_channels
#        self.kernel_size = kernel_size
#        self.upsample_cfg = upsample_cfg.copy()
#        self.SE_layer = SELayer(channel=in_channels)
#        self.encoder = nn.Sequential(
#            nn.Conv2d(in_channels=self.in_channels, out_channels=self.compressed_channels,
#                      kernel_size=self.kernel_size, padding=self.kernel_size // 2),
#            nn.ReLU(inplace=True),
#            nn.Conv2d(in_channels=self.compressed_channels, out_channels=self.compressed_channels,
#                      kernel_size=self.kernel_size, padding=self.kernel_size // 2),
#            nn.ReLU(inplace=True),
#            nn.Conv2d(in_channels=self.compressed_channels, out_channels=self.compressed_channels,
#                      kernel_size=self.kernel_size, padding=self.kernel_size // 2),
#            nn.ReLU(inplace=True),
#            nn.Conv2d(in_channels=self.compressed_channels, out_channels=self.compressed_channels,
#                      kernel_size=self.kernel_size, padding=self.kernel_size // 2),
#            nn.ReLU(inplace=True)
#        )
#
#        self.mask_yx = nn.Sequential(
#            nn.Conv2d(in_channels=self.compressed_channels, out_channels=self.in_channels,
#                      kernel_size=self.kernel_size, padding=self.kernel_size // 2),
#            nn.BatchNorm2d(self.in_channels),
#            nn.Sigmoid())
#
#        self.mask_xy = nn.Sequential(
#            nn.Conv2d(in_channels=self.compressed_channels, out_channels=self.in_channels,
#                      kernel_size=self.kernel_size, padding=self.kernel_size // 2),
#            nn.BatchNorm2d(self.in_channels),
#            nn.Sigmoid())
#
#    def forward(self, x, y):
#        if 'scale_factor' in self.upsample_cfg:
#            upsampled_x = F.interpolate(x, **self.upsample_cfg)
#        else:
#            upsampled_x = F.interpolate(x, size=y.shape[2:], **self.upsample_cfg)
#        encoded = self.encoder(upsampled_x + y + self.SE_layer(upsampled_x + y))
#        enhanced_x = x + F.adaptive_max_pool2d(self.mask_yx(encoded) * y,
#                                               output_size=x.shape[2:])
#        enhanced_y = y + self.mask_xy(encoded) * upsampled_x
#
#        return enhanced_x, enhanced_y
#
#
#from ..builder import NECKS
#
#
#@NECKS.register_module()
#class MGF_FPN(nn.Module):
#    def __init__(self,
#                 in_channels,
#                 out_channels,
#                 num_outs,
#                 start_level=0,
#                 end_level=-1,
#                 no_norm_on_lateral=False,
#                 conv_cfg=None,
#                 norm_cfg=None,
#                 act_cfg=None,
#                 upsample_cfg=dict(mode='nearest')):
#        super(MGF_FPN, self).__init__()
#        assert isinstance(in_channels, list)
#        self.in_channels = in_channels
#        self.out_channels = out_channels
#        self.num_ins = len(in_channels)
#        self.num_outs = num_outs
#        self.no_norm_on_lateral = no_norm_on_lateral
#        self.fp16_enabled = False
#        self.upsample_cfg = upsample_cfg.copy()
#        self.mutually_guided_filtering = nn.ModuleList()
#        self.mutually_guided_filtering = self.mutually_guided_filtering.append(
#            MGF_1().cuda()).append(MGF_2().cuda()).append(MGF_3().cuda())
#
#        if end_level == -1:
#            self.backbone_end_level = self.num_ins
#            assert num_outs >= self.num_ins - start_level
#        else:
#            # if end_level < inputs, no extra level is allowed
#            self.backbone_end_level = end_level
#            assert end_level <= len(in_channels)
#            assert num_outs == end_level - start_level
#        self.start_level = start_level
#        self.SE_Layers = nn.ModuleList()
#        self.lateral_convs = nn.ModuleList()
#
#
#        kernel_size = [1, 3, 5, 7]
#        dilation = [1, 1, 1, 1]
#        padding = [0, 1, 2, 3]
#
#        for i in range(self.start_level, self.backbone_end_level):
#            l_conv = ConvModule(
#                in_channels[i],
#                out_channels,
#                kernel_size[i],
#                padding=padding[i],
#                dilation=dilation[i],
#                conv_cfg=conv_cfg,
#                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
#                act_cfg=act_cfg,
#                inplace=False)
#
#            self.lateral_convs.append(l_conv)
#    # default init_weights for conv(msra) and norm in ConvModule
#    def init_weights(self):
#        """Initialize the weights of FPN module."""
#        for m in self.modules():
#            if isinstance(m, nn.Conv2d):
#                xavier_init(m, distribution='uniform')
#
#    @auto_fp16()
#    def forward(self, inputs):
#        """Forward function."""
#        assert len(inputs) == len(self.in_channels)
#
#
#        # build laterals
#        laterals = [
#            lateral_conv(inputs[i + self.start_level])
#            for i, lateral_conv in enumerate(self.lateral_convs)
#        ]
#
#        # build top-down path
#        used_backbone_levels = len(laterals)
#        for i in range(used_backbone_levels - 1, 0, -1):
#            laterals[i], laterals[i - 1] = self.mutually_guided_filtering[i - 1](laterals[i], laterals[i - 1])
#
#        laterals.append(F.max_pool2d(laterals[-1], 1, stride=2))
#
#        return tuple(laterals)



import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init

from mmcv.runner import auto_fp16


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class MGF_1(nn.Module):
    def __init__(self, in_channels=256, compressed_channels=256, out_channels=256,
                 kernel_size=3, dilation=[1, 2, 3, 4], upsample_cfg=dict(mode='nearest')):
        super(MGF_1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.compressed_channels = compressed_channels
        self.dilation = dilation
        self.upsample_cfg = upsample_cfg.copy()
        self.SE_layer = SELayer(channel=in_channels)

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.compressed_channels,
                      kernel_size=self.kernel_size, padding=(self.kernel_size - 1) * self.dilation[0] // 2,
                      dilation=self.dilation[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.compressed_channels, out_channels=self.compressed_channels,
                      kernel_size=self.kernel_size, padding=(self.kernel_size - 1) * self.dilation[1] // 2,
                      dilation=self.dilation[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.compressed_channels, out_channels=self.compressed_channels,
                      kernel_size=self.kernel_size, padding=(self.kernel_size - 1) * self.dilation[2] // 2,
                      dilation=self.dilation[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.compressed_channels, out_channels=self.compressed_channels,
                      kernel_size=self.kernel_size, padding=(self.kernel_size - 1) * self.dilation[3] // 2,
                      dilation=self.dilation[3]),
            nn.ReLU(inplace=True)
        )

        self.mask_yx = nn.Sequential(
            nn.Conv2d(in_channels=self.compressed_channels, out_channels=self.in_channels,
                      kernel_size=self.kernel_size, padding=(self.kernel_size - 1) * 2 // 2,
                      dilation=2),
            nn.BatchNorm2d(self.in_channels),
            nn.Sigmoid())

        self.mask_xy = nn.Sequential(
            nn.Conv2d(in_channels=self.compressed_channels, out_channels=self.in_channels,
                      kernel_size=self.kernel_size, padding=(self.kernel_size - 1) * 2 // 2,
                      dilation=2),
            nn.BatchNorm2d(self.in_channels),
            nn.Sigmoid())

    def forward(self, x, y):
        if 'scale_factor' in self.upsample_cfg:
            upsampled_x = F.interpolate(x, **self.upsample_cfg)
        else:
            upsampled_x = F.interpolate(x, size=y.shape[2:], **self.upsample_cfg)
        encoded = self.encoder(upsampled_x + y + self.SE_layer(upsampled_x + y))
        enhanced_x = x + F.adaptive_max_pool2d(self.mask_yx(encoded) * y,
                                               output_size=x.shape[2:])
        enhanced_y = y + self.mask_xy(encoded) * upsampled_x

        return enhanced_x, enhanced_y


class MGF_2(nn.Module):
    def __init__(self, in_channels=256, compressed_channels=256, out_channels=256,
                 kernel_size=3, dilation=[1, 2, 3, 4], upsample_cfg=dict(mode='nearest')):
        super(MGF_2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.compressed_channels = compressed_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.upsample_cfg = upsample_cfg.copy()
        self.SE_layer = SELayer(channel=in_channels)

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.compressed_channels,
                      kernel_size=self.kernel_size, padding=(self.kernel_size - 1) * self.dilation[0] // 2,
                      dilation=self.dilation[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.compressed_channels, out_channels=self.compressed_channels,
                      kernel_size=self.kernel_size, padding=(self.kernel_size - 1) * self.dilation[1] // 2,
                      dilation=self.dilation[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.compressed_channels, out_channels=self.compressed_channels,
                      kernel_size=self.kernel_size, padding=(self.kernel_size - 1) * self.dilation[2] // 2,
                      dilation=self.dilation[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.compressed_channels, out_channels=self.compressed_channels,
                      kernel_size=self.kernel_size, padding=(self.kernel_size - 1) * self.dilation[3] // 2,
                      dilation=self.dilation[3]),
            nn.ReLU(inplace=True)
        )

        self.mask_yx = nn.Sequential(
            nn.Conv2d(in_channels=self.compressed_channels, out_channels=self.in_channels,
                      kernel_size=self.kernel_size, padding=(self.kernel_size - 1) * 2 // 2,
                      dilation=2),
            nn.BatchNorm2d(self.in_channels),
            nn.Sigmoid())

        self.mask_xy = nn.Sequential(
            nn.Conv2d(in_channels=self.compressed_channels, out_channels=self.in_channels,
                      kernel_size=self.kernel_size, padding=(self.kernel_size - 1) * 2 // 2,
                      dilation=2),
            nn.BatchNorm2d(self.in_channels),
            nn.Sigmoid())

    def forward(self, x, y):
        if 'scale_factor' in self.upsample_cfg:
            upsampled_x = F.interpolate(x, **self.upsample_cfg)
        else:
            upsampled_x = F.interpolate(x, size=y.shape[2:], **self.upsample_cfg)
        encoded = self.encoder(upsampled_x + y + self.SE_layer(upsampled_x + y))
        enhanced_x = x + F.adaptive_max_pool2d(self.mask_yx(encoded) * y,
                                               output_size=x.shape[2:])
        enhanced_y = y + self.mask_xy(encoded) * upsampled_x

        return enhanced_x, enhanced_y


class MGF_3(nn.Module):
    def __init__(self, in_channels=256, compressed_channels=256, out_channels=256,
                 kernel_size=3, dilation=[1, 2, 3, 4], upsample_cfg=dict(mode='nearest')):
        super(MGF_3, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.compressed_channels = compressed_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.upsample_cfg = upsample_cfg.copy()
        self.SE_layer = SELayer(channel=in_channels)

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.compressed_channels,
                      kernel_size=self.kernel_size, padding=(self.kernel_size - 1) * self.dilation[0] // 2,
                      dilation=self.dilation[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.compressed_channels, out_channels=self.compressed_channels,
                      kernel_size=self.kernel_size, padding=(self.kernel_size - 1) * self.dilation[1] // 2,
                      dilation=self.dilation[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.compressed_channels, out_channels=self.compressed_channels,
                      kernel_size=self.kernel_size, padding=(self.kernel_size - 1) * self.dilation[2] // 2,
                      dilation=self.dilation[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=self.compressed_channels, out_channels=self.compressed_channels,
                      kernel_size=self.kernel_size, padding=(self.kernel_size - 1) * self.dilation[3] // 2,
                      dilation=self.dilation[3]),
            nn.ReLU(inplace=True)
        )

        self.mask_yx = nn.Sequential(
            nn.Conv2d(in_channels=self.compressed_channels, out_channels=self.in_channels,
                      kernel_size=self.kernel_size, padding=(self.kernel_size - 1) * 2 // 2,
                      dilation=2),
            nn.BatchNorm2d(self.in_channels),
            nn.Sigmoid())

        self.mask_xy = nn.Sequential(
            nn.Conv2d(in_channels=self.compressed_channels, out_channels=self.in_channels,
                      kernel_size=self.kernel_size, padding=(self.kernel_size - 1) * 2 // 2,
                      dilation=2),
            nn.BatchNorm2d(self.in_channels),
            nn.Sigmoid())

    def forward(self, x, y):
        if 'scale_factor' in self.upsample_cfg:
            upsampled_x = F.interpolate(x, **self.upsample_cfg)
        else:
            upsampled_x = F.interpolate(x, size=y.shape[2:], **self.upsample_cfg)
        encoded = self.encoder(upsampled_x + y + self.SE_layer(upsampled_x + y))
        enhanced_x = x + F.adaptive_max_pool2d(self.mask_yx(encoded) * y,
                                               output_size=x.shape[2:])
        enhanced_y = y + self.mask_xy(encoded) * upsampled_x

        return enhanced_x, enhanced_y


from ..builder import NECKS


@NECKS.register_module()
class AMGF_FPN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest')):
        super(AMGF_FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()
        self.mutually_guided_filtering = nn.ModuleList()
        self.mutually_guided_filtering = self.mutually_guided_filtering.append(
            MGF_1().cuda()).append(MGF_2().cuda()).append(MGF_3().cuda())

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.SE_Layers = nn.ModuleList()
        self.lateral_convs = nn.ModuleList()


        kernel_size = [1, 3, 5, 7]
        dilation = [1, 1, 1, 1]
        padding = [0, 1, 2, 3]

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                kernel_size[i],
                padding=padding[i],
                dilation=dilation[i],
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        """Initialize the weights of FPN module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i], laterals[i - 1] = self.mutually_guided_filtering[i - 1](laterals[i], laterals[i - 1])

        laterals.append(F.max_pool2d(laterals[-1], 1, stride=2))

        return tuple(laterals)

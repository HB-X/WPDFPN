import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init
from mmcv.runner import auto_fp16


from ..builder import NECKS


@NECKS.register_module()
class BiFPN(nn.Module):
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
        super(BiFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level

        self.td_ws = []
        self.bu_ws = []
        self.td_relus = nn.ModuleList()
        self.bu_relus = nn.ModuleList()
        self.bu_convs = nn.ModuleList()
        self.d_convs = nn.ModuleList()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            if i != self.backbone_end_level - 1:
                td_w = nn.Parameter(torch.ones(2, dtype=torch.float32),
                                    requires_grad=True)
                bu_w = nn.Parameter(torch.ones(3, dtype=torch.float32),
                                    requires_grad=True)
                td_relu = nn.ReLU()
                bu_relu = nn.ReLU()
                bu_conv = ConvModule(
                    out_channels,
                    out_channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                d_conv = ConvModule(
                    out_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.td_ws.append(td_w)
                self.bu_ws.append(bu_w)
                self.td_relus.append(td_relu)
                self.bu_relus.append(bu_relu)
                self.bu_convs.append(bu_conv)
                self.d_convs.append(d_conv)
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

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

        td_ws = []
        bu_ws = []

        for i in range(self.backbone_end_level - 1):
            td_w = self.td_relus[i](self.td_ws[i])
            bu_w = self.bu_relus[i](self.bu_ws[i])
            td_w = td_w / (torch.sum(td_w, dim=0) + 0.0001)
            bu_w = bu_w / (torch.sum(bu_w, dim=0) + 0.0001)
            td_ws.append(td_w)
            bu_ws.append(bu_w)

        td_laterals = laterals

        for i in range(self.backbone_end_level - 1, 0, -1):
            pre_shape = td_laterals[i - 1].shape[2:]
            td_laterals[i - 1] = td_ws[i - 1][0] * F.interpolate(
                td_laterals[i], size=pre_shape, **self.upsample_cfg) + \
                                 td_ws[i - 1][1] * td_laterals[i - 1]

        td_outputs = [
            self.fpn_convs[i](td_laterals[i]) for i in range(self.backbone_end_level)
        ]

        bu_outputs = td_outputs
        for i in range(self.start_level + 1, self.backbone_end_level):
            bu_outputs[i] = bu_ws[i - 1][0] * laterals[i] + \
                            bu_ws[i - 1][1] * bu_outputs[i] + \
                            bu_ws[i - 1][2] * self.d_convs[i - 1](bu_outputs[i - 1])

        for i in range(1, self.backbone_end_level):
            bu_outputs[i] = self.bu_convs[i - 1](bu_outputs[i])

        bu_outputs.append(F.max_pool2d(bu_outputs[-1], 1, stride=2))

        return tuple(bu_outputs)







#import torch
#import torch.nn as nn
#import torch.nn.functional as F
#from mmcv.cnn import ConvModule, xavier_init
#from mmcv.runner import auto_fp16
#
#
#from ..builder import NECKS
#
#
#@NECKS.register_module()
#class BiFPN(nn.Module):
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
#        super(BiFPN, self).__init__()
#        assert isinstance(in_channels, list)
#        self.in_channels = in_channels
#        self.out_channels = out_channels
#        self.num_ins = len(in_channels)
#        self.num_outs = num_outs
#        self.no_norm_on_lateral = no_norm_on_lateral
#        self.fp16_enabled = False
#        self.upsample_cfg = upsample_cfg.copy()
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
#        self.end_level = end_level
#
#        self.tdw_54 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
#        self.tdw_43 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
#        self.tdw_32 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
#
#        self.buw_233 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
#        self.buw_334 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
#        self.buw_45 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
#
#        self.lateral_convs = nn.ModuleList()
#        self.td_convs = nn.ModuleList()
#        self.bu_convs = nn.ModuleList()
#
#        for i in range(self.start_level, self.backbone_end_level):
#            if i != self.backbone_end_level - 1:
#                td_conv = ConvModule(
#                    out_channels,
#                    out_channels,
#                    3,
#                    padding=1,
#                    conv_cfg=conv_cfg,
#                    norm_cfg=norm_cfg,
#                    act_cfg=act_cfg,
#                    inplace=False)
#                bu_conv = ConvModule(
#                    out_channels,
#                    out_channels,
#                    3,
#                    padding=1,
#                    conv_cfg=conv_cfg,
#                    norm_cfg=norm_cfg,
#                    act_cfg=act_cfg,
#                    inplace=False)
#                self.td_convs.append(td_conv)
#                self.bu_convs.append(bu_conv)
#            l_conv = ConvModule(
#                in_channels[i],
#                out_channels,
#                1,
#                conv_cfg=conv_cfg,
#                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
#                act_cfg=act_cfg,
#                inplace=False)
#            self.lateral_convs.append(l_conv)
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
#        # build laterals
#        laterals = [
#            lateral_conv(inputs[i + self.start_level])
#            for i, lateral_conv in enumerate(self.lateral_convs)
#        ]
#
#        td_laterals = laterals
#        tdw_54 = F.relu(self.tdw_54)
#        tdw_43 = F.relu(self.tdw_43)
#        tdw_32 = F.relu(self.tdw_32)
#
#        tdw_54 = tdw_54 / (torch.sum(tdw_54, dim=0) + 0.0001)
#        tdw_43 = tdw_43 / (torch.sum(tdw_43, dim=0) + 0.0001)
#        tdw_32 = tdw_32 / (torch.sum(tdw_32, dim=0) + 0.0001)
#
#        buw_233 = F.relu(self.buw_233)
#        buw_334 = F.relu(self.buw_334)
#        buw_45 = F.relu(self.buw_45)
#
#        buw_233 = buw_233 / (torch.sum(buw_233, dim=0) + 0.0001)
#        buw_334 = buw_334 / (torch.sum(buw_334, dim=0) + 0.0001)
#        buw_45 = buw_45 / (torch.sum(buw_45, dim=0) + 0.0001)
#
#        td_laterals[2] = tdw_54[0] * F.interpolate(
#            td_laterals[3], size=td_laterals[2].shape[2:],
#            **self.upsample_cfg) + tdw_54[1] * td_laterals[2]
#        td_laterals[2] = self.td_convs[2](td_laterals[2])
#        
#        td_laterals[1] = tdw_43[0] * F.interpolate(
#            td_laterals[2], size=td_laterals[1].shape[2:],
#            **self.upsample_cfg) + tdw_43[1] * td_laterals[1]
#        td_laterals[1] = self.td_convs[1](td_laterals[1])
#        
#        td_laterals[0] = tdw_32[0] * F.interpolate(
#            td_laterals[1], size=td_laterals[0].shape[2:],
#            **self.upsample_cfg) + tdw_32[1] * td_laterals[0]
#        td_laterals[0] = self.td_convs[0](td_laterals[0])
#
#        bu_outputs = td_laterals
#
#        bu_outputs[1] = buw_233[0] * laterals[1] + buw_233[1] * bu_outputs[1] + \
#                        buw_233[2] * F.adaptive_max_pool2d(
#            bu_outputs[0], output_size=bu_outputs[1].shape[2:])
#        bu_outputs[1] = self.bu_convs[0](bu_outputs[1])
#        
#        bu_outputs[2] = buw_334[0] * laterals[2] + buw_334[1] * bu_outputs[2] + \
#                        buw_334[2] * F.adaptive_max_pool2d(
#            bu_outputs[1], output_size=bu_outputs[2].shape[2:])
#        bu_outputs[2] = self.bu_convs[1](bu_outputs[2])
#
#        bu_outputs[3] = buw_45[1] * laterals[3] + \
#                        buw_45[1] * F.adaptive_max_pool2d(
#            bu_outputs[2], output_size=bu_outputs[3].shape[2:])
#        bu_outputs[3] = self.bu_convs[2](bu_outputs[3])
#
#        bu_outputs.append(F.max_pool2d(bu_outputs[-1], 1, stride=2))
#
#        return tuple(bu_outputs)



#import warnings
#import torch
#import torch.nn as nn
#import torch.nn.functional as F
#from mmcv.cnn import ConvModule, xavier_init
#from mmcv.runner import auto_fp16
#
#
#class DepthwiseSeparableConv(nn.Module):
#    def __init__(self, in_channels, out_channels, kernel_size=3,
#                 stride=1, padding=1, dilation=1):
#        super(DepthwiseSeparableConv, self).__init__()
#        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride,
#                                   padding=padding, dilation=dilation, groups=in_channels, bias=False)
#        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1,
#                                   stride=1, padding=0, dilation=1, groups=1, bias=False)
#
#        self.bn = nn.BatchNorm2d(out_channels, momentum=0.01, eps=1e-3)
#        self.act = nn.ReLU()
#
#    def forward(self, x):
#        x = self.depthwise_conv(x)
#        x = self.pointwise_conv(x)
#        x = self.bn(x)
#        return self.act(x)
#
#from ..builder import NECKS
#
#
#@NECKS.register_module()
#class BiFPN(nn.Module):
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
#        super(BiFPN, self).__init__()
#        assert isinstance(in_channels, list)
#        self.in_channels = in_channels
#        self.out_channels = out_channels
#        self.num_ins = len(in_channels)
#        self.num_outs = num_outs
#        self.no_norm_on_lateral = no_norm_on_lateral
#        self.fp16_enabled = False
#        self.upsample_cfg = upsample_cfg.copy()
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
#        self.end_level = end_level
#
#        self.lateral_convs = nn.ModuleList()
#        self.td_convs = nn.ModuleList()
#        self.bu_convs = nn.ModuleList()
#        self.td_weights = []
#        self.bu_weights = []
#        self.td_relus = nn.ModuleList()
#        self.bu_relus = nn.ModuleList()
#
#        for i in range(self.start_level, self.backbone_end_level):
#            if i != self.backbone_end_level - 1:
#                td_relu = nn.ReLU()
#                td_weight = nn.Parameter(torch.ones(2, dtype=torch.float32),
#                                         requires_grad=True)
#                td_conv = DepthwiseSeparableConv(
#                    in_channels=out_channels,
#                    out_channels=out_channels)
#                self.td_relus.append(td_relu)
#                self.td_weights.append(td_weight)
#                self.td_convs.append(td_conv)
#
#                bu_conv = DepthwiseSeparableConv(
#                    in_channels=out_channels,
#                    out_channels=out_channels)
#                self.bu_convs.append(bu_conv)
#                if i != self.backbone_end_level - 2:
#                    bu_relu = nn.ReLU()
#                    bu_weight = nn.Parameter(torch.ones(3, dtype=torch.float32),
#                                             requires_grad=True)
#                    self.bu_relus.append(bu_relu)
#                    self.bu_weights.append(bu_weight)
#
#            l_conv = ConvModule(
#                in_channels[i],
#                out_channels,
#                1,
#                conv_cfg=conv_cfg,
#                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
#                act_cfg=act_cfg,
#                inplace=False)
#            self.lateral_convs.append(l_conv)
#
#        self.top_bu_relu = nn.ReLU()
#        self.top_bu_weight = nn.Parameter(torch.ones(2, dtype=torch.float32),
#                                          requires_grad=True)
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
#        # build laterals
#        laterals = [
#            lateral_conv(inputs[i + self.start_level])
#            for i, lateral_conv in enumerate(self.lateral_convs)
#        ]
#
#        td_laterals = laterals
#
#        td_w = [
#            self.td_relus[i](self.td_weights[i]) for i in range(0, 3)
#        ]
#        bu_w = [
#            self.bu_relus[i](self.bu_weights[i]) for i in range(0, 2)
#        ]
#        top_bu_w = self.top_bu_relu(self.top_bu_weight)
#
#        for i in range(0, len(td_w)):
#            td_w[i] = td_w[i] / (torch.sum(td_w[i], dim=0) + 0.0001)
#
#        for i in range(0, len(bu_w)):
#            bu_w[i] = bu_w[i] / (torch.sum(bu_w[i], dim=0) + 0.0001)
#
#        top_bu_w = top_bu_w / (torch.sum(top_bu_w, dim=0) + 0.0001)
#
#        # build top-down path
#        used_backbone_levels = len(laterals)
#
#        for i in range(used_backbone_levels - 1, 0, -1):
#            prev_shape = td_laterals[i - 1].shape[2:]
#            td_laterals[i - 1] = td_w[i - 1][0] * td_laterals[i - 1] + \
#                                 td_w[i - 1][1] * F.interpolate(
#                td_laterals[i], size=prev_shape, **self.upsample_cfg)
#
#        for i in range(0, self.backbone_end_level - 1):
#            td_laterals[i] = self.td_convs[i](td_laterals[i])
#
#        td_outputs = td_laterals
#
#        for i in range(1, used_backbone_levels):
#            current_shape = td_outputs[i].shape[2:]
#            if i != used_backbone_levels - 1:
#                td_outputs[i] = bu_w[i - 1][0] * laterals[i] + bu_w[i - 1][1] * td_outputs[i] + \
#                                bu_w[i - 1][2] * F.interpolate(
#                    td_outputs[i - 1], size=current_shape, **self.upsample_cfg)
#            else:
#                td_outputs[i] = top_bu_w[0] * td_outputs[i] + top_bu_w[1] * F.interpolate(
#                    td_outputs[i - 1], size=current_shape, **self.upsample_cfg)
#
#        for i in range(1, used_backbone_levels):
#            td_outputs[i] = self.bu_convs[i - 1](td_outputs[i])
#
#        td_outputs.append(F.max_pool2d(td_outputs[-1], 1, stride=2))
#
#        return tuple(td_outputs)

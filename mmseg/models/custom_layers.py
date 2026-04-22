import string
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from timm.layers import variance_scaling_
from mmcv.ops import CARAFEPack


class Upsample2x(nn.Module):
    def __init__(self, in_ch=None, out_ch=None, use_conv=True, transpose_conv=False):
        super().__init__()
        out_ch = out_ch if out_ch else in_ch
        self.use_conv = use_conv
        self.transpose_conv = transpose_conv
        self.out_ch = out_ch
        if use_conv:
            if transpose_conv:
                self.conv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
            else:
                self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, dilation=1)

    def forward(self, x):
        B, C, H, W = x.shape
        if self.use_conv:
            if self.transpose_conv:
                x = self.conv(x)
            else:
                x = F.interpolate(x, (H * 2, W * 2), mode="nearest")
                x = self.conv(x)
        else:
            x = F.interpolate(x, (H * 2, W * 2), mode="bilinear")
        return x


def build_norm_layer(norm_type, channels, num_groups=32, eps=1e-6):
    if norm_type is None:
        return nn.Identity()
    elif norm_type == "SyncBN":
        return nn.SyncBatchNorm(channels)
    elif norm_type == "GN":
        return nn.GroupNorm(
            num_groups=min(channels // 4, num_groups), num_channels=channels, eps=eps
        )
    else:
        return nn.BatchNorm2d(channels)


class ConvUp(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=2,
        stride=2,
        padding=0,
        act=nn.GELU,
        norm_type="SyncBN",
    ):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        self.norm = build_norm_layer(norm_type, out_channels)
        self.act = act()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class CA(nn.Module):
    def __init__(self, input_channels, squeeze_channels, init=False):
        super().__init__()
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc1 = torch.nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = torch.nn.Conv2d(squeeze_channels, input_channels, 1)
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        if init:
            variance_scaling_(
                self.fc2.weight, scale=0.0, mode="fan_avg", distribution="uniform"
            )
            torch.nn.init.constant_(self.fc2.bias, 4)

    def _scale(self, input):
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.sigmoid(scale)

    def forward(self, input):
        scale = self._scale(input)
        return scale * input


class ECA(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, k_size=3, init=False):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=True
        )
        self.sigmoid = nn.Sigmoid()

        if init:
            variance_scaling_(
                self.conv.weight, scale=0.0, mode="fan_avg", distribution="uniform"
            )
            torch.nn.init.constant_(self.conv.bias, 4)

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class ConvModule(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        transpose=False,
        act=nn.GELU,
        norm_type="SyncBN",
        init_scale=1.0,
        groups=1,
        dropout_ratio=0,
        bias=False,
    ):
        super().__init__()
        if transpose:
            self.conv = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
            )
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        if init_scale:
            variance_scaling_(
                self.conv.weight,
                scale=init_scale,
                mode="fan_avg",
                distribution="uniform",
            )
            if bias:
                nn.init.zeros_(self.conv.bias)
        self.norm = build_norm_layer(norm_type, out_channels)
        self.act = act()
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


def conv3x3(
    in_planes,
    out_planes,
    stride=1,
    bias=True,
    dilation=1,
    padding=1,
    init_scale=1.0,
    groups=1,
):
    conv = nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
        groups=groups,
    )
    variance_scaling_(
        conv.weight, scale=init_scale, mode="fan_avg", distribution="uniform"
    )
    nn.init.zeros_(conv.bias)
    return conv


def conv1x1(in_planes, out_planes, stride=1, bias=True, padding=0, init_scale=1.0):
    conv = nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, padding=padding, bias=bias
    )
    variance_scaling_(
        conv.weight, scale=init_scale, mode="fan_avg", distribution="uniform"
    )
    nn.init.zeros_(conv.bias)
    return conv


class GuideModule(nn.Module):
    def __init__(self, guide_type="cnn", guide_dim=64, with_eca=True):
        super().__init__()
        self.guide_type = guide_type
        self.guide_dim = guide_dim
        if guide_type == "cnn":
            self.guide_modules = nn.Sequential(
                *[
                    ConvModule(3, guide_dim, kernel_size=3, stride=2, padding=1),
                    ECA(),
                    ConvModule(
                        guide_dim, guide_dim, kernel_size=3, stride=1, padding=1
                    ),
                    ECA(),
                    ConvModule(
                        guide_dim, guide_dim, kernel_size=3, stride=1, padding=1
                    ),
                    ECA(),
                ]
            )
        elif guide_type == "sifmlp":
            self.guide_modules = nn.Sequential(
                *[
                    SimpleImplicitFeaturizer(),
                    ConvModule(83, guide_dim, kernel_size=1),
                    ECA(),
                    ConvModule(guide_dim, guide_dim, kernel_size=1),
                    ECA(),
                ]
            )
        else:
            self.guide_modules = None
            self.guide_dim = 3
        # self.eca = ECA() if with_eca else None

    def forward(self, x):
        if self.guide_modules:
            g = self.guide_modules(x)
            # if self.eca:
            #     g = self.eca(g)
        else:
            g = x
        return g


class SAPAUpsampler(torch.nn.Module):
    def __init__(self, dim_x, scale, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.up = SAPAModule(dim_x=dim_x, dim_y=3, up_factor=scale)

    def adapt_guidance(self, source, guidance):
        _, _, h, w = source.shape
        small_guidance = F.adaptive_avg_pool2d(guidance, (h * 2, w * 2))
        return small_guidance

    def forward(self, source, guidance):
        source = self.up(self.adapt_guidance(source, guidance), source)
        return source


class GuidedUpsampler(nn.Module):
    def __init__(
        self,
        scale,
        in_dim,
        out_dim,
        g_dim,
        upsampler_type="tconv",
        progressive=False,
        residual_out=False,
    ):
        super().__init__()
        self.progressive = progressive
        self.residual_out = residual_out
        self.upsampler_type = upsampler_type
        if upsampler_type == "tconv":
            self.up = ConvModule(
                transpose=True,
                in_channels=in_dim,
                out_channels=out_dim,
                kernel_size=scale,
                stride=scale,
            )
            self.fuse = ConvModule(
                in_channels=out_dim + g_dim, out_channels=out_dim, kernel_size=1
            )
        elif upsampler_type == "conv":
            self.up = nn.Sequential(
                *[
                    nn.Upsample(
                        scale_factor=scale, mode="bilinear", align_corners=False
                    ),
                    ConvModule(
                        in_channels=in_dim,
                        out_channels=out_dim,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                ]
            )
            self.fuse = ConvModule(
                in_channels=out_dim + g_dim, out_channels=out_dim, kernel_size=1
            )
        elif upsampler_type == "CARAFEPack":
            self.up = CARAFEPack(
                channels=in_dim, up_kernel=3, up_group=1, scale_factor=scale
            )
            self.fuse = ConvModule(
                in_channels=in_dim + g_dim, out_channels=out_dim, kernel_size=1
            )
        elif upsampler_type == "sapa":
            self.up = SAPAUpsampler(in_dim, scale)
            self.fuse = ConvModule(
                in_channels=in_dim, out_channels=out_dim, kernel_size=1
            )

    def apply_conv(self, feature, guidance, up, fuse):
        if self.upsampler_type == "sapa":
            output = fuse(self.up(feature, guidance))
            return output
        big_source = up(feature)
        _, _, h, w = big_source.shape
        small_guidance = F.interpolate(guidance, (h, w), mode="bilinear")
        output = fuse(torch.cat([big_source, small_guidance], dim=1))
        return big_source + output if self.residual_out else output

    def forward(self, feature, guidance):
        if not self.progressive:
            feature = self.apply_conv(feature, guidance, self.up, self.fuse)
        return feature


class RCAB(nn.Module):
    def __init__(self, n_feat, bias=True, bn=False, act=nn.GELU(), res_scale=1):
        super().__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv3x3(n_feat, n_feat, bias=bias))
            if bn:
                modules_body.append(nn.SyncBatchNorm(n_feat))
            if i == 0:
                modules_body.append(act)
        modules_body.append(ECA())
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        # res = self.body(x).mul(self.res_scale)
        res += x
        return res


class ResidualGroup(nn.Module):
    def __init__(self, n_feat, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = [RCAB(n_feat, bias=True, bn=False) for _ in range(n_resblocks)]
        modules_body.append(conv3x3(n_feat, n_feat))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class ResnetBlock(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch=None,
        norm_type="SyncBN",
        act=nn.GELU,
        skip_rescale=True,
        init_scale=0.0,
        use_dw=False,
    ):
        super().__init__()
        out_ch = out_ch if out_ch else in_ch
        self.in_ch = in_ch
        self.out_ch = out_ch

        groups = out_ch // 2 if use_dw else 1
        self.Conv_0 = ConvModule(
            in_ch,
            out_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            norm_type=norm_type,
            init_scale=init_scale,
            groups=groups,
        )
        self.Conv_1 = ConvModule(
            in_ch,
            out_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            norm_type=norm_type,
            init_scale=init_scale,
            groups=groups,
        )

        self.skip_rescale = skip_rescale
        self.out_norm = nn.SyncBatchNorm(out_ch)
        self.out_act = act()

    def forward(self, x):
        h = self.Conv_0(x)
        h = self.Conv_1(h)
        if not self.skip_rescale:
            out = x + h
        else:
            out = (x + h) / np.sqrt(2.0)
        out = self.out_act(self.out_norm(out))
        return out


# class ResnetBlock(nn.Module):
#     # ResnetBlockBigGANpp from https://github.com/yang-song/score_sde_pytorch
#     def __init__(self, in_ch, out_ch=None, up2=False, down2=False, any_resize=None, norm_type='SyncBN', act=nn.GELU,
#                  dropout=0.1, skip_rescale=True, init_scale=0.):
#         super().__init__()
#         out_ch = out_ch if out_ch else in_ch
#         self.in_ch = in_ch
#         self.out_ch = out_ch
#         self.Conv_0 = conv1x1(in_ch, out_ch)
#         self.norm_0 = build_norm_layer(norm_type, out_ch)
#         self.Dropout_0 = nn.Dropout(dropout)
#         self.Conv_1 = conv3x3(out_ch, out_ch)
#         self.norm_1 = build_norm_layer(norm_type, out_ch)
#         self.up = up2
#         self.down = down2
#         self.any_resize = any_resize
#         if self.in_ch != self.out_ch or self.up:
#             self.Conv_2 = conv1x1(in_ch, out_ch, init_scale=init_scale)
#         self.skip_rescale = skip_rescale
#         self.act = act()
#
#     def forward(self, x):
#         if self.any_resize is not None:
#             x = F.interpolate(x, size=self.any_resize, mode='bilinear')
#         if self.up:
#             x = F.interpolate(x, scale_factor=2, mode='bilinear')
#         if self.down:
#             x = F.interpolate(x, scale_factor=0.5, mode='bilinear')
#         h = self.Conv_0(x)
#         h = self.act(self.norm_0(h))
#         h = self.Dropout_0(h)
#         h = self.Conv_1(h)
#
#         if self.in_ch != self.out_ch or self.up:
#             x = self.Conv_2(x)
#
#         if not self.skip_rescale:
#             out = x + h
#         else:
#             out = (x + h) / np.sqrt(2.)
#         out = self.act(self.norm_1(out))
#         return out


def _einsum(a, b, c, x, y):
    einsum_str = "{},{}->{}".format("".join(a), "".join(b), "".join(c))
    return torch.einsum(einsum_str, x, y)


def contract_inner(x, y):
    """tensordot(x, y, 1)."""
    x_chars = list(string.ascii_lowercase[: len(x.shape)])
    y_chars = list(string.ascii_lowercase[len(x.shape) : len(y.shape) + len(x.shape)])
    y_chars[0] = x_chars[-1]  # first axis of y and last of x get summed
    out_chars = x_chars[:-1] + y_chars[1:]
    return _einsum(x_chars, y_chars, out_chars, x, y)


class NIN(nn.Module):
    def __init__(self, in_dim, num_units, init_scale=0.1):
        super().__init__()
        self.W = nn.Parameter(torch.empty([in_dim, num_units]), requires_grad=True)
        variance_scaling_(self.W.data, init_scale)
        self.b = nn.Parameter(torch.zeros(num_units), requires_grad=True)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        y = contract_inner(x, self.W) + self.b
        return y.permute(0, 3, 1, 2)


class AttnBlock(nn.Module):
    """Channel-wise self-attention block."""

    def __init__(self, channels, skip_rescale=False, init_scale=0.0):
        super().__init__()
        self.NIN_0 = NIN(channels, channels)
        self.NIN_1 = NIN(channels, channels)
        self.NIN_2 = NIN(channels, channels)
        self.NIN_3 = NIN(channels, channels, init_scale=init_scale)
        self.skip_rescale = skip_rescale

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.NIN_0(x)
        k = self.NIN_1(x)
        v = self.NIN_2(x)

        w = torch.einsum("bchw,bcij->bhwij", q, k) * (int(C) ** (-0.5))
        w = torch.reshape(w, (B, H, W, H * W))
        w = F.softmax(w, dim=-1)
        w = torch.reshape(w, (B, H, W, H, W))
        h = torch.einsum("bhwij,bcij->bchw", w, v)
        h = self.NIN_3(h)
        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / np.sqrt(2.0)

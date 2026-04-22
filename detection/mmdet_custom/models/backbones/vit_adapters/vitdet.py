import logging
import math

import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_
from mmdet.registry import MODELS

from mmseg.models.backbones.vit_hooks.timm_vit import TIMMVisionTransformer
from mmseg.models.backbones.vit_hooks.timm_vit import ResBottleneckBlock

_logger = logging.getLogger(__name__)


@MODELS.register_module()
class ViTDet(TIMMVisionTransformer):
    def __init__(self, pretrain_size=224, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.cls_token = None
        self.pretrain_size = (pretrain_size, pretrain_size)
        embed_dim = self.embed_dim

        # ViTDet usually uses final feature only
        self.norm = self.norm_layer(embed_dim)

        # simple feature pyramid
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            nn.GroupNorm(32, embed_dim),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
        )
        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
        )
        self.fpn3 = nn.Conv2d(
            embed_dim, embed_dim, kernel_size=3, stride=1, padding=1
        )  # nn.Identity()
        self.fpn4 = nn.Conv2d(
            embed_dim, embed_dim, kernel_size=3, stride=2, padding=1
        )  # nn.MaxPool2d(kernel_size=2, stride=2)

        self.fpn1.apply(self._init_weights)
        self.fpn2.apply(self._init_weights)
        self.fpn3.apply(self._init_weights)
        self.fpn4.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, ResBottleneckBlock):
            m.norm3.weight.data.zero_()
            m.norm3.bias.data.zero_()

    def _get_pos_embed(self, pos_embed, H, W):
        pos_embed = pos_embed.reshape(
            1,
            self.pretrain_size[0] // self.patch_embed.patch_size[0],
            self.pretrain_size[1] // self.patch_embed.patch_size[1],
            -1,
        ).permute(0, 3, 1, 2)
        pos_embed = (
            F.interpolate(pos_embed, size=(H, W), mode="bicubic", align_corners=False)
            .reshape(1, -1, H * W)
            .permute(0, 2, 1)
        )
        return pos_embed

    def forward_features(self, x):
        x, H, W = self.patch_embed(x)
        pos_embed = self._get_pos_embed(self.pos_embed[:, 1:], H, W)
        x = self.pos_drop(x + pos_embed)
        for blk in self.blocks:
            x = blk(x, H, W)

        return x, H, W

    def forward(self, x):
        x, H, W = self.forward_features(x)

        bs, n, dim = x.shape
        x = self.norm(x).transpose(1, 2).reshape(bs, dim, H, W)

        f1 = self.fpn1(x).contiguous()
        f2 = self.fpn2(x).contiguous()
        f3 = self.fpn3(x).contiguous()
        f4 = self.fpn4(x).contiguous()

        return [f1, f2, f3, f4]

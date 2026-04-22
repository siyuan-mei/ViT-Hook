import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS
from mmseg.models.backbones.vit_hooks.ops.modules import MSDeformAttn
from .hook_modules import SpatialPriorModule, multiscale_deform_inputs, HookModule
from timm.models.layers import trunc_normal_
from .beit2 import BEiT2


@MODELS.register_module()
class BEiTHook(BEiT2):
    def __init__(
        self,
        pretrain_size=224,
        crop_size=None,
        num_hook=4,
        with_self_attn=True,
        use_self_flash_attn=True,
        use_cross_flash_attn=False,
        conv_inplane=64,
        n_points=4,
        num_hook_heads=12,
        self_attn_drop=True,
        with_cross_attn=True,
        msmlp_ratio=0.25,
        deform_ratio=0.5,
        with_msmlp=True,
        add_vit_feature=True,
        use_abs_pos=True,
        with_cp=False,
        use_extra_crossattn=False,
        *args,
        **kwargs
    ):
        super().__init__(with_cp=with_cp, *args, **kwargs)
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        # self.cls_token = None
        self.num_block = len(self.blocks)
        self.pretrain_size = (pretrain_size, pretrain_size)
        self.add_vit_feature = add_vit_feature
        self.level_embed = nn.Parameter(
            torch.zeros(3, self.embed_dim), requires_grad=True
        )
        self.use_abs_pos = use_abs_pos
        if use_abs_pos and crop_size is not None:
            self.crop_size = crop_size
            self.num_patches = (crop_size[0] // self.patch_size) * (
                crop_size[1] // self.patch_size
            )
            self.level_pos_embed = nn.Parameter(
                torch.zeros(1, self.num_patches, self.embed_dim), requires_grad=True
            )
        #     rotary_emb_dim = 0
        # else:
        #     rotary_emb_dim = torch.tensor(self.embed_dim // deform_num_heads // 2).to(torch.int32)

        self.spm = SpatialPriorModule(
            inplanes=conv_inplane, embed_dim=self.embed_dim, with_cp=False
        )

        self.hooks = nn.Sequential(
            *[
                HookModule(
                    dim=self.embed_dim,
                    num_heads=num_hook_heads,
                    n_points=n_points,
                    with_self_attn=with_self_attn,
                    use_self_flash_attn=use_self_flash_attn,
                    use_cross_flash_attn=use_cross_flash_attn,
                    with_cross_attn=with_cross_attn,
                    norm_layer=norm_layer,
                    self_attn_drop=self_attn_drop,
                    deform_ratio=deform_ratio,
                    with_msmlp=with_msmlp,
                    msmlp_ratio=msmlp_ratio,
                    drop_path=self.drop_path_rate,
                    extra_crossattn=(
                        (True if i == num_hook - 1 else False) and use_extra_crossattn
                    ),
                )
                for i in range(num_hook)
            ]
        )

        if self.add_vit_feature:
            self.up_4x = nn.Upsample(
                scale_factor=4, mode="bilinear", align_corners=False
            )
            self.up_2x = nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=False
            )
            self.down_2x = nn.Upsample(
                scale_factor=0.5, mode="bilinear", align_corners=False
            )

        self.outlayer1 = nn.SyncBatchNorm(self.embed_dim)
        self.outlayer2 = nn.SyncBatchNorm(self.embed_dim)
        self.outlayer3 = nn.SyncBatchNorm(self.embed_dim)
        self.outlayer4 = nn.SyncBatchNorm(self.embed_dim)

        self.up = nn.ConvTranspose2d(self.embed_dim, self.embed_dim, 2, 2)
        self.hooks.apply(self._init_weights)
        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.apply(self._init_deform_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def _add_pos_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        if self.use_abs_pos:
            c2 = c2 + self._reshape_pos_embed(
                self.level_pos_embed, self.crop_size, scale_factor=2
            )
            c3 = c3 + self.level_pos_embed
            c4 = c4 + self._reshape_pos_embed(
                self.level_pos_embed, self.crop_size, scale_factor=0.5
            )
        return c2, c3, c4

    def _reshape_pos_embed(
        self, pos_embed, original_size=None, size=None, scale_factor=None
    ):
        pos_embed = pos_embed.reshape(
            1,
            original_size[0] // self.patch_size,
            original_size[1] // self.patch_size,
            self.embed_dim,
        ).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(
            pos_embed,
            size=size,
            scale_factor=scale_factor,
            mode="bicubic",
            antialias=True,
        )
        new_h, new_w = pos_embed.shape[2], pos_embed.shape[3]
        pos_embed = pos_embed.reshape(1, self.embed_dim, new_h * new_w).permute(0, 2, 1)
        return pos_embed

    def _get_pos_embed(self, pos_embed, H, W):
        pos_embed = pos_embed.reshape(
            1, self.pretrain_size[0] // 16, self.pretrain_size[1] // 16, -1
        ).permute(0, 3, 1, 2)
        pos_embed = (
            F.interpolate(
                pos_embed,
                size=(H, W),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
            .reshape(1, -1, H * W)
            .permute(0, 2, 1)
        )
        return pos_embed

    @torch.autocast(device_type="cuda", dtype=torch.float16)
    def forward(self, image):
        deform_inputs1, deform_inputs2 = multiscale_deform_inputs(image)

        # SPM forward
        c1, c2, c3, c4 = self.spm(image)
        c2, c3, c4 = self._add_pos_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        # Patch Embedding forward
        vit_feature, H, W = self.patch_embed(image)
        bs, n, dim = vit_feature.shape
        cls = self.cls_token.expand(bs, -1, -1)
        vit_feature = torch.cat((cls, vit_feature), dim=1)

        if self.pos_embed is not None:
            pos_embed = self._get_pos_embed(self.pos_embed, H, W)
            vit_feature = vit_feature + pos_embed
        # vit_feature = self.pos_drop(vit_feature)

        # rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None

        for blk in self.blocks:
            vit_feature = blk(vit_feature, H, W)

        vit_feature = vit_feature[
            :,
            1:,
        ]

        for i, layer in enumerate(self.hooks):
            c = layer(
                query=c,
                deform_inputs1=deform_inputs1,
                deform_inputs2=deform_inputs2,
                feat=vit_feature,
                H=H,
                W=W,
            )

        c2 = c[:, 0 : c2.size(1), :]
        c3 = c[:, c2.size(1) : c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1) :, :]

        c1 = c1.transpose(1, 2).view(bs, dim, H * 4, W * 4).contiguous()
        c2 = c2.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H, W).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        c1 = c1 + self.up(c2)

        if self.add_vit_feature:
            x = vit_feature.transpose(1, 2).view(bs, dim, H, W).contiguous()
            c1 += self.up_4x(x)
            c2 += self.up_2x(x)
            c3 += x
            c4 += self.down_2x(x)

        # Final Norm
        f1 = self.outlayer1(c1)
        f2 = self.outlayer2(c2)
        f3 = self.outlayer3(c3)
        f4 = self.outlayer4(c4)
        return [f1, f2, f3, f4]

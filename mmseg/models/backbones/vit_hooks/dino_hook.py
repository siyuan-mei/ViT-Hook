import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS
from mmseg.models.backbones.vit_hooks.ops.modules import MSDeformAttn
from mmseg.models.backbones.vit_hooks.hook_modules import (
    SpatialPriorModule,
    multiscale_deform_inputs,
    HookModule,
)
from timm.models.layers import trunc_normal_
from mmseg.models.backbones.vit_hooks.model_loader import DINOWrapper


@MODELS.register_module()
class DINOHook(DINOWrapper):
    def __init__(
        self,
        model_version: str = "dinov2_vitb14_reg",
        is_frozen=True,
        active_mlp_indices=None,
        active_block_indices=None,
        convert_14to16=False,
        img_size=None,
        num_hook=4,
        with_self_attn=True,
        with_msmlp=True,
        use_self_flash_attn=True,
        use_cross_flash_attn=False,
        conv_inplane=64,
        n_points=4,
        num_heads=16,
        drop_path_rate=0.2,
        msmlp_ratio=0.25,
        deform_ratio=0.5,
        add_vit_feature=True,
        use_abs_pos=True,
        use_extra_crossattn=False,
        self_attn_drop=True,
    ):
        super().__init__(model_version, is_frozen)
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.is_frozen = is_frozen
        self.add_vit_feature = add_vit_feature
        self.drop_path_rate = drop_path_rate

        if convert_14to16:
            self.convert_14to16()
        if active_mlp_indices is not None:
            self.activate_mlps(active_mlp_indices)
        if active_block_indices is not None and active_mlp_indices is None:
            self.activate_blocks(active_block_indices)

        self.level_embed = nn.Parameter(
            torch.zeros(3, self.embed_dim), requires_grad=True
        )
        self.use_abs_pos = use_abs_pos
        if use_abs_pos and img_size is not None:
            self.img_size = img_size
            self.num_patches = (img_size[0] // 16) * (img_size[1] // 16)
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.num_patches, self.embed_dim), requires_grad=True
            )
            # rotary_emb_dim = 0
        # else:
        #     rotary_emb_dim = torch.tensor(self.embed_dim // num_heads // 2).to(torch.int32)

        self.spm = SpatialPriorModule(
            inplanes=conv_inplane, embed_dim=self.embed_dim, with_cp=False
        )

        self.hooks = nn.Sequential(
            *[
                HookModule(
                    dim=self.embed_dim,
                    num_heads=num_heads,
                    n_points=n_points,
                    with_self_attn=with_self_attn,
                    use_self_flash_attn=use_self_flash_attn,
                    use_cross_flash_attn=use_cross_flash_attn,
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
            c2 = c2 + self._reshape_pos_embed(self.pos_embed, 2)
            c3 = c3 + self.pos_embed
            c4 = c4 + self._reshape_pos_embed(self.pos_embed, 0.5)
        return c2, c3, c4

    def _reshape_pos_embed(self, pos_embed, scale_factor=1.0):
        size = (self.img_size[0] // 16, self.img_size[1] // 16)
        pos_embed = pos_embed.reshape(1, size[0], size[1], self.embed_dim).permute(
            0, 3, 1, 2
        )
        pos_embed = F.interpolate(
            pos_embed, scale_factor=scale_factor, mode="bicubic", antialias=True
        )
        # Reshape back to 3D (1, L, C)
        new_h, new_w = pos_embed.shape[2], pos_embed.shape[3]
        pos_embed = pos_embed.reshape(1, self.embed_dim, new_h * new_w).permute(0, 2, 1)
        return pos_embed

    @torch.autocast(device_type="cuda", dtype=torch.float16)
    def forward(self, input):
        h_x, w_x = input.shape[-2:]

        # SPM forward
        c1, c2, c3, c4 = self.spm(input)
        c2, c3, c4 = self._add_pos_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        # Patch Embedding forward
        h_v, w_v = (h_x // self.patch_size, w_x // self.patch_size)

        vit_feature = self.vit.patch_embed(input)
        for blk in self.vit.blocks:
            vit_feature = blk(vit_feature)
        # vit_feature = self.vit.norm(vit_feature)
        vit_feature = vit_feature[:, -int(h_v * w_v) :, :]
        bs, n, dim = vit_feature.shape

        h_sp, w_sp = (h_x // 16, w_x // 16)
        deform_inputs1, deform_inputs2 = multiscale_deform_inputs(
            input, self.patch_size
        )

        if self.hooks is not None:
            for i, layer in enumerate(self.hooks):
                c = layer(
                    query=c,
                    deform_inputs1=deform_inputs1,
                    deform_inputs2=deform_inputs2,
                    feat=vit_feature,
                    H=h_sp,
                    W=w_sp,
                )

        c2 = c[:, 0 : c2.size(1), :]
        c3 = c[:, c2.size(1) : c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1) :, :]

        c1 = c1.transpose(1, 2).view(bs, dim, h_sp * 4, w_sp * 4).contiguous()
        c2 = c2.transpose(1, 2).view(bs, dim, h_sp * 2, w_sp * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, h_sp, w_sp).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, h_sp // 2, w_sp // 2).contiguous()
        c1 = c1 + self.up(c2)

        if self.add_vit_feature:
            x = vit_feature.transpose(1, 2).view(bs, dim, h_v, w_v).contiguous()
            c1 += nn.functional.interpolate(
                x, size=(c1.shape[2], c1.shape[3]), mode="bilinear", align_corners=False
            )
            c2 += nn.functional.interpolate(
                x, size=(c2.shape[2], c2.shape[3]), mode="bilinear", align_corners=False
            )
            if self.patch_size == 16:
                c3 += x
            else:
                c3 += nn.functional.interpolate(
                    x,
                    size=(c3.shape[2], c3.shape[3]),
                    mode="bilinear",
                    align_corners=False,
                )
            c4 += nn.functional.interpolate(
                x, size=(c4.shape[2], c4.shape[3]), mode="bilinear", align_corners=False
            )

        # Final Norm
        f1 = self.outlayer1(c1)
        f2 = self.outlayer2(c2)
        f3 = self.outlayer3(c3)
        f4 = self.outlayer4(c4)
        return [f1, f2, f3, f4]

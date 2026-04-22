from functools import partial
import torch
import torch.nn as nn
from mmseg.models.backbones.vit_hooks.ops.modules import MSDeformAttn
from timm.models.layers import DropPath
import torch.utils.checkpoint as cp
from flash_attn.modules.mha import MHA


def get_reference_points(spatial_shapes, device):
    reference_points_list = []
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
            torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
        )
        ref_y = ref_y.reshape(-1)[None] / H_
        ref_x = ref_x.reshape(-1)[None] / W_
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    reference_points = reference_points[:, :, None]
    return reference_points


def multiscale_deform_inputs(x, patch_size=16):
    bs, c, h, w = x.shape
    spatial_shapes = torch.as_tensor(
        [(h // 8, w // 8), (h // 16, w // 16), (h // 32, w // 32)],
        dtype=torch.long,
        device=x.device,
    )
    level_start_index = torch.cat(
        (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
    )
    reference_points = get_reference_points(
        [(h // 8, w // 8), (h // 16, w // 16), (h // 32, w // 32)], x.device
    )
    deform_inputs1 = [reference_points, spatial_shapes, level_start_index]

    spatial_shapes = torch.as_tensor(
        [(h // patch_size, w // patch_size)], dtype=torch.long, device=x.device
    )
    level_start_index = torch.cat(
        (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
    )
    reference_points = get_reference_points(
        [(h // 8, w // 8), (h // 16, w // 16), (h // 32, w // 32)], x.device
    )
    deform_inputs2 = [reference_points, spatial_shapes, level_start_index]
    return deform_inputs1, deform_inputs2


class MultiScaleDWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        n = N // 21
        x1 = x[:, 0 : 16 * n, :].transpose(1, 2).view(B, C, H * 2, W * 2).contiguous()
        x2 = x[:, 16 * n : 20 * n, :].transpose(1, 2).view(B, C, H, W).contiguous()
        x3 = x[:, 20 * n :, :].transpose(1, 2).view(B, C, H // 2, W // 2).contiguous()
        x1 = self.dwconv(x1).flatten(2).transpose(1, 2)
        x2 = self.dwconv(x2).flatten(2).transpose(1, 2)
        x3 = self.dwconv(x3).flatten(2).transpose(1, 2)
        x = torch.cat([x1, x2, x3], dim=1)
        return x


class MultiScaleMLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = MultiScaleDWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SelfAttn(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=6,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        drop_path=0.0,
        use_flash_attn=True,
        n_points=4,
        deform_ratio=1.0,
        rotary_emb_dim=0,
        self_attn_drop=True,
    ):
        super().__init__()
        self.use_flash_attn = use_flash_attn
        if use_flash_attn:
            self.attn = MHA(
                dim,
                num_heads=num_heads,
                rotary_emb_dim=rotary_emb_dim,
                cross_attn=False,
                use_flash_attn=True,
                use_alibi=False,
            )
        else:
            self.attn = MSDeformAttn(
                d_model=dim,
                n_levels=3,
                n_heads=num_heads,
                n_points=n_points,
                ratio=deform_ratio,
            )
        self.norm = norm_layer(dim)
        self.self_attn_drop = self_attn_drop
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, query, reference_points, spatial_shapes, level_start_index):
        if self.use_flash_attn:
            attn = self.attn(self.norm(query))
        else:
            norm_c = self.norm(query)
            attn = self.attn(
                norm_c,
                reference_points,
                norm_c,
                spatial_shapes,
                level_start_index,
                None,
            )
        q_sp = query + self.drop_path(attn) if self.self_attn_drop else query + attn
        return q_sp


class CrossAttn(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=6,
        n_points=4,
        deform_ratio=1.0,
        with_msmlp=True,
        msmlp_ratio=0.25,
        drop=0.0,
        drop_path=0.0,
        use_flash_attn=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.use_flash_attn = use_flash_attn
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        if use_flash_attn:
            self.attn = MHA(
                dim,
                num_heads=num_heads,
                cross_attn=True,
                use_flash_attn=True,
                use_alibi=False,
            )
        else:
            self.attn = MSDeformAttn(
                d_model=dim,
                n_levels=1,
                n_heads=num_heads,
                n_points=n_points,
                ratio=deform_ratio,
            )
        self.with_msmlp = with_msmlp
        if with_msmlp:
            self.msmlp = MultiScaleMLP(
                in_features=dim, hidden_features=int(dim * msmlp_ratio), drop=drop
            )
            self.msmlp_norm = norm_layer(dim)

    def forward(
        self, query, reference_points, feat, spatial_shapes, level_start_index, H, W
    ):
        if self.use_flash_attn:
            cross_attn = self.attn(self.query_norm(query), self.feat_norm(feat))
        else:
            cross_attn = self.attn(
                self.query_norm(query),
                reference_points,
                self.feat_norm(feat),
                spatial_shapes,
                level_start_index,
                None,
            )
        query = query + cross_attn
        if self.with_msmlp:
            query = query + self.drop_path(self.msmlp(self.msmlp_norm(query), H, W))
        return query


class HookModule(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=6,
        n_points=4,
        deform_ratio=1.0,
        with_msmlp=True,
        with_self_attn=True,
        use_self_flash_attn=True,
        rotary_emb_dim=0,
        msmlp_ratio=0.25,
        drop=0.0,
        drop_path=0.0,
        use_cross_flash_attn=False,
        with_cross_attn=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        extra_crossattn=False,
        self_attn_drop=True,
    ):
        super().__init__()
        self.with_self_attn = with_self_attn
        self.with_cross_attn = with_cross_attn
        if self.with_self_attn:
            self.self_attn = SelfAttn(
                dim=dim,
                num_heads=num_heads,
                use_flash_attn=use_self_flash_attn,
                n_points=n_points,
                deform_ratio=deform_ratio,
                norm_layer=norm_layer,
                drop_path=drop_path,
                rotary_emb_dim=rotary_emb_dim,
                self_attn_drop=self_attn_drop,
            )
        if self.with_cross_attn:
            self.cross_attn = CrossAttn(
                dim=dim,
                num_heads=num_heads,
                use_flash_attn=use_cross_flash_attn,
                n_points=n_points,
                deform_ratio=deform_ratio,
                with_msmlp=with_msmlp,
                msmlp_ratio=msmlp_ratio,
                drop=drop,
                drop_path=drop_path,
                norm_layer=norm_layer,
            )

        if extra_crossattn:
            self.extra_cross_attn = nn.Sequential(
                *[
                    CrossAttn(
                        dim=dim,
                        num_heads=num_heads,
                        use_flash_attn=use_cross_flash_attn,
                        n_points=n_points,
                        deform_ratio=deform_ratio,
                        with_msmlp=with_msmlp,
                        msmlp_ratio=msmlp_ratio,
                        drop=drop,
                        drop_path=drop_path,
                        norm_layer=norm_layer,
                    )
                    for _ in range(2)
                ]
            )
        else:
            self.extra_cross_attn = None

    def forward(self, query, feat, deform_inputs1, deform_inputs2, H, W):
        if self.with_self_attn:
            query = self.self_attn(
                query, deform_inputs1[0], deform_inputs1[1], deform_inputs1[2]
            )
        if self.with_cross_attn:
            query = self.cross_attn(
                query,
                deform_inputs2[0],
                feat,
                deform_inputs2[1],
                deform_inputs2[2],
                H,
                W,
            )
        if self.extra_cross_attn is not None:
            for cross_attn in self.extra_cross_attn:
                query = cross_attn(
                    query,
                    deform_inputs2[0],
                    feat,
                    deform_inputs2[1],
                    deform_inputs2[2],
                    H,
                    W,
                )
        return query


class SpatialPriorModule(nn.Module):
    def __init__(self, inplanes=64, embed_dim=384, with_cp=False):
        super().__init__()
        self.with_cp = with_cp

        self.stem = nn.Sequential(
            *[
                nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
                nn.SyncBatchNorm(inplanes),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False
                ),
                nn.SyncBatchNorm(inplanes),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False
                ),
                nn.SyncBatchNorm(inplanes),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ]
        )
        self.conv2 = nn.Sequential(
            *[
                nn.Conv2d(
                    inplanes,
                    2 * inplanes,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.SyncBatchNorm(2 * inplanes),
                nn.ReLU(inplace=True),
            ]
        )
        self.conv3 = nn.Sequential(
            *[
                nn.Conv2d(
                    2 * inplanes,
                    4 * inplanes,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.SyncBatchNorm(4 * inplanes),
                nn.ReLU(inplace=True),
            ]
        )
        self.conv4 = nn.Sequential(
            *[
                nn.Conv2d(
                    4 * inplanes,
                    4 * inplanes,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.SyncBatchNorm(4 * inplanes),
                nn.ReLU(inplace=True),
            ]
        )
        self.fc1 = nn.Conv2d(
            inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.fc2 = nn.Conv2d(
            2 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.fc3 = nn.Conv2d(
            4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.fc4 = nn.Conv2d(
            4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True
        )

    def forward(self, x):

        def _inner_forward(x):
            c1 = self.stem(x)
            c2 = self.conv2(c1)
            c3 = self.conv3(c2)
            c4 = self.conv4(c3)
            c1 = self.fc1(c1)
            c2 = self.fc2(c2)
            c3 = self.fc3(c3)
            c4 = self.fc4(c4)
            bs, dim, _, _ = c2.shape
            c1 = c1.view(bs, dim, -1).transpose(1, 2)
            c2 = c2.view(bs, dim, -1).transpose(1, 2)  # 8s
            c3 = c3.view(bs, dim, -1).transpose(1, 2)  # 16s
            c4 = c4.view(bs, dim, -1).transpose(1, 2)  # 32s

            return c1, c2, c3, c4

        if self.with_cp and x.requires_grad:
            outs = cp.checkpoint(_inner_forward, x)
        else:
            outs = _inner_forward(x)
        return outs

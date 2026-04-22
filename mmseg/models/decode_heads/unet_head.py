from ..custom_layers import *
from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead


class Bottleneck(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        modules = [
            ResnetBlock(in_ch=in_ch),
            AttnBlock(channels=in_ch),
            ResnetBlock(in_ch=in_ch),
        ]
        self.neck = nn.Sequential(*modules)

    def forward(self, x):
        return self.neck(x)


@MODELS.register_module()
class UNetHead(BaseDecodeHead):
    def __init__(
        self,
        stage_dims,
        num_blocks=2,
        norm_type="SyncBN",
        if_cat=True,
        feature_sizes=None,
        **kwargs
    ):
        super().__init__(in_channels=0, channels=0, **kwargs)
        self.stages = nn.ModuleList()
        self.if_cat = if_cat
        in_ch = stage_dims[-1]
        self.bottleneck = Bottleneck(in_ch)

        for i_stage, dim in enumerate(reversed(stage_dims)):
            layers = []
            in_ch = out_ch = dim
            for i_block in range(num_blocks):
                if i_block == num_blocks - 1 and i_stage != len(stage_dims) - 1:
                    out_ch = stage_dims[-(i_stage + 2)]
                if if_cat:
                    layers.append(
                        ResnetBlock(in_ch=in_ch * 2, out_ch=out_ch, norm_type=norm_type)
                    )
                    in_ch = max(out_ch // 2, 1)
                else:
                    layers.append(
                        ResnetBlock(in_ch=in_ch, out_ch=out_ch, norm_type=norm_type)
                    )

            if i_stage != len(stage_dims) - 1:
                if feature_sizes is None:
                    layers.append(
                        ResnetBlock(
                            in_ch=out_ch, out_ch=out_ch, up2=True, norm_type=norm_type
                        )
                    )
                else:
                    next_feature_size = feature_sizes[-(i_stage + 2)]
                    layers.append(
                        ResnetBlock(
                            in_ch=out_ch,
                            out_ch=out_ch,
                            any_resize=next_feature_size,
                            norm_type=norm_type,
                        )
                    )

            self.stages.append(nn.Sequential(*layers))

        self.cls_layer = conv3x3(stage_dims[0], self.num_classes)

    def forward(self, x, bottleneck_feature=None):
        if not isinstance(x, list) and not isinstance(x, tuple):
            raise ValueError("Input x should be a list of feature maps.")
        h = self.bottleneck(x[-1])
        for stage, feature in zip(self.stages, reversed(x)):
            if self.if_cat:
                feature_cat = torch.cat([h, feature], dim=1)
                h = stage(feature_cat)
            else:
                h = stage(feature + h)
        h = self.cls_layer(h)
        return h

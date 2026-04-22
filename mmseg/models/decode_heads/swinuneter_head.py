from monai.networks.nets import SwinUNETR
from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead


@MODELS.register_module()
class SwinUNETRHead(BaseDecodeHead):
    """reimplementation of ResidualEncoder in nnUnet"""

    def __init__(
        self,
        img_size,
        in_channels=3,
        depths=(2, 2, 2, 2),
        feature_size=48,
        spatial_dims=2,
        **kwargs
    ):
        super().__init__(in_channels=0, channels=0, **kwargs)
        self.model = SwinUNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=self.num_classes,
            num_heads=(3, 6, 12, 24),
            norm_name="instance",
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            normalize=True,
            use_checkpoint=False,
            spatial_dims=spatial_dims,
            downsample="merging",
            use_v2=False,
            depths=depths,
            feature_size=feature_size,
        )

    def forward(self, x):
        output = self.model(x)
        return output

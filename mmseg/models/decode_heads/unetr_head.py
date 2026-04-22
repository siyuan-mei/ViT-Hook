from monai.networks.nets import UNETR
from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead


@MODELS.register_module()
class UNETRHead(BaseDecodeHead):
    """reimplementation of ResidualEncoder in nnUnet"""

    def __init__(self, img_size, in_channels=3, spatial_dims=2, **kwargs):
        super().__init__(in_channels=0, channels=0, **kwargs)
        self.model = UNETR(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=self.num_classes,
            spatial_dims=spatial_dims,
        )

    def forward(self, x):
        output = self.model(x)
        return output

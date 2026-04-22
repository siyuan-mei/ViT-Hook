from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class UwMadisonDataset(BaseSegDataset):

    METAINFO = dict(
        classes=("background", "Stomach", "Small bowel", "Large bowel"),
        palette=[[0, 0, 0], [0, 0, 255], [0, 255, 0], [255, 0, 0]],
    )

    def __init__(
        self,
        img_suffix=".png",
        seg_map_suffix=".png",
        reduce_zero_label=False,
        **kwargs
    ) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs
        )

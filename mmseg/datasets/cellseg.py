from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class CellSegDataset(BaseSegDataset):

    METAINFO = dict(
        classes=("background", "cell", "boundary"),
        palette=[[0, 0, 0], [255, 0, 0], [0, 0, 255]],
    )

    def __init__(
        self,
        img_suffix="_0000.png",
        seg_map_suffix="_label.png",
        reduce_zero_label=False,
        **kwargs
    ) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs
        )

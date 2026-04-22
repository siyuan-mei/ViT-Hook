# dataset settings
dataset_type = "ADE20KDataset"
data_root = "data/ade/ADEChallengeData2016"
crop_size = (896, 896)  # The crop size during training.
train_pipeline = [
    dict(type="LoadImageFromFile"),  # First pipeline to load images from file path.
    dict(
        type="LoadAnnotations", reduce_zero_label=True
    ),  # Second pipeline to load annotations for current image.
    dict(
        type="RandomResize",  # Augmentation pipeline that resize the images and their annotations.
        scale=(3584, 896),
        ratio_range=(0.5, 2.0),  # The augmented scale range as ratio.
        keep_ratio=True,
    ),
    dict(type="RandomCrop", crop_size=crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),  # Pack the inputs data for the semantic segmentation.
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=(3584, 896), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type="LoadAnnotations", reduce_zero_label=True),
    dict(type="PackSegInputs"),
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(
        type="TestTimeAug",
        transforms=[
            [dict(type="Resize", scale_factor=r, keep_ratio=True) for r in img_ratios],
            [dict(type="ResizeToMultiple", size_divisor=32)],
            [
                dict(type="RandomFlip", prob=0.0, direction="horizontal"),
                dict(type="RandomFlip", prob=1.0, direction="horizontal"),
            ],
            [dict(type="LoadAnnotations")],
            [dict(type="PackSegInputs")],
        ],
    ),
]
train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,  # Shut down the worker processes after an epoch end, which can accelerate training speed.
    sampler=dict(
        type="InfiniteSampler", shuffle=True
    ),  # Randomly shuffle during training.
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path="images/training", seg_map_path="annotations/training"
        ),
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path="images/validation", seg_map_path="annotations/validation"
        ),
        pipeline=test_pipeline,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(type="IoUMetric", iou_metrics=["mIoU", "mDice", "mFscore"])
test_evaluator = val_evaluator

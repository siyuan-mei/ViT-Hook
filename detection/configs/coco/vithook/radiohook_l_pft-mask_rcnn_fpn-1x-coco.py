_base_ = [
    "../../_base_/models/mask-rcnn_r50_fpn_amp.py",
    "../../_base_/datasets/coco_instance.py",
    "../../_base_/schedules/schedule_1x.py",
    "../../_base_/default_runtime.py",
]

custom_imports = dict(
    imports=["mmdet_custom.models.backbones.vit_hooks.radio_hook"],
    allow_failed_imports=False,
)

# model settings
data_preprocessor = dict(
    type="DetDataPreprocessor",
    mean=[0, 0, 0],
    std=[255.0, 255.0, 255.0],
    bgr_to_rgb=True,
    pad_mask=True,
    pad_size_divisor=32,
)
model = dict(
    type="MaskRCNN_amp",
    data_preprocessor=data_preprocessor,
    backbone=dict(
        _delete_=True,
        type="RadioHook",
        model_version="radio_v2.5-l",
        is_frozen=True,
        num_hook=4,
        img_size=None,
        num_heads=16,
        with_self_attn=True,
        use_self_flash_attn=True,
        use_cross_flash_attn=False,
        use_extra_crossattn=False,
        self_attn_drop=True,
        use_abs_pos=False,
        active_block_indices=(5, 11, 17, 23),
        drop_path_rate=0.5,
        add_vit_feature=True,
        conv_inplane=64,
        n_points=4,
        msmlp_ratio=0.25,
        deform_ratio=0.5,
        with_msmlp=True,
    ),
    neck=dict(
        type="FPN", in_channels=[1024, 1024, 1024, 1024], out_channels=256, num_outs=5
    ),
)


optim_wrapper = dict(
    type="AmpOptimWrapper",
    optimizer=dict(type="AdamW", lr=1.0e-4, betas=(0.9, 0.999), weight_decay=0.05),
    dtype="float16",
    paramwise_cfg=dict(custom_keys={"backbone.blocks": dict(lr_mult=0.1)}),
)

data_root = "datasets/detection/COCO"

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    dataset=dict(
        data_root=data_root,
    ),
)

val_dataloader = dict(dataset=dict(data_root=data_root))
test_dataloader = val_dataloader

# This is needed to allow distributed training when some parameters
# have no gradient.
find_unused_parameters = True


vis_backends = [
    dict(type="LocalVisBackend", name="localvis"),
    dict(
        type="WandbVisBackend",
        init_kwargs=dict(
            project="coco",
            resume="allow",
            id=None,
        ),
        name="wandbvis",
    ),
]
visualizer = dict(
    type="DetLocalVisualizer",
    vis_backends=vis_backends,
    name="visualizer",
    # dataset_name='coco'
)

# _base_ = [
#     '../../_base_/models/co_dino_5scale_r50_lsj_8xb2_1x_coco.py'
# ]

_base_ = ["../../_base_/codino/co_dino_5scale_swin_l_lsj_16xb1_1x_coco.py"]

custom_imports = dict(
    imports=[
        "mmdet_custom.models.backbones.vit_hooks.radio_hook",
        "mmdet_custom.models.detectors.codetr.codetr",
    ],
    allow_failed_imports=False,
)

image_size = (1024, 1024)
batch_augments = [dict(type="BatchFixedSizePad", size=image_size, pad_mask=True)]

# model settings
data_preprocessor = dict(
    type="DetDataPreprocessor",
    mean=[0, 0, 0],
    std=[255.0, 255.0, 255.0],
    bgr_to_rgb=True,
    pad_mask=True,
    pad_size_divisor=32,
    batch_augments=batch_augments,
)

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        _delete_=True,
        type="RadioHook",
        model_version="radio_v2.5-h",
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
        active_block_indices=(7, 15, 23, 31),
        drop_path_rate=0.5,
        add_vit_feature=True,
        conv_inplane=64,
        n_points=4,
        msmlp_ratio=0.25,
        deform_ratio=0.5,
        with_msmlp=True,
    ),
    neck=dict(in_channels=[1280, 1280, 1280, 1280]),
    # neck=dict(
    #         type='SFP',
    #         in_channels=[1280],
    #         out_channels=256,
    #         num_outs=5,
    #         use_p2=True,
    #         use_act_checkpoint=False),
    query_head=dict(
        num_query=1500,
        dn_cfg=dict(
            label_noise_scale=0.5,
            box_noise_scale=0.4,
            group_cfg=dict(num_dn_queries=300),
        ),
        transformer=dict(encoder=dict(with_cp=6)),
    ),
)

train_cfg = dict(max_epochs=36)
param_scheduler = [
    dict(type="LinearLR", start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type="MultiStepLR", milestones=[30]),
]

optim_wrapper = dict(
    _delete_=True,
    type="AmpOptimWrapper",
    optimizer=dict(type="AdamW", lr=1.0e-4, betas=(0.9, 0.999), weight_decay=0.05),
    dtype="float16",
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={"backbone.blocks": dict(lr_mult=0.1)}),
)


data_root = "datasets/detection/COCO"

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        dataset=dict(
            data_root=data_root,
        )
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

_base_ = [
    "../../_base_/datasets/ade20k_512x512.py",
    "../../_base_/default_runtime.py",
    "../../_base_/schedules/schedule_160k.py",
]
# model settings
crop_size = (512, 512)
norm_cfg = dict(type="SyncBN", requires_grad=True)
data_preprocessor = dict(
    type="SegDataPreProcessor",
    # scale=1.0,
    mean=[0, 0, 0],
    std=[255.0, 255.0, 255.0],
    bgr_to_rgb=True,
    pad_val=0,  # Padding value. Default: 0.
    seg_pad_val=255,  # Padding value of segmentation map. Default: 255.
    size=crop_size,
)  # Fixed padding siz
num_classes = 150
model = dict(
    type="EncoderDecoder",
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type="RadioHook",
        model_version="radio_v2.5-b",
        is_frozen=True,
        num_hook=4,
        img_size=crop_size,
        with_self_attn=True,
        use_self_flash_attn=True,
        use_cross_flash_attn=False,
        use_abs_pos=True,
        active_mlp_indices=(2, 5, 8, 11),
        drop_path_rate=0.4,
        add_vit_feature=True,
        num_heads=12,
        conv_inplane=64,
        n_points=4,
        msmlp_ratio=0.25,
        deform_ratio=0.5,
        with_msmlp=True,
    ),
    decode_head=dict(
        type="UPerHead",
        pool_scales=(1, 2, 3, 6),
        in_channels=(768, 768, 768, 768),
        channels=512,
        dropout_ratio=0.1,
        in_index=(0, 1, 2, 3),
        norm_cfg=norm_cfg,
        # act_cfg=dict(type='GELU'),
        num_classes=150,
        align_corners=False,
        loss_decode=[
            dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
        ],
    ),
    auxiliary_head=dict(
        type="FCNHead",
        in_channels=768,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=0.4),
    ),
    # model training and testing settings
    train_cfg=dict(type="IterBasedTrainLoop", val_interval=2000),
    test_cfg=dict(
        type="TestLoop", mode="slide", crop_size=crop_size, stride=(341, 341)
    ),
)

optim_wrapper = dict(
    type="AmpOptimWrapper",
    optimizer=dict(
        type="AdamW", lr=1.0e-4, betas=(0.9, 0.999), weight_decay=0.05, eps=1e-8
    ),
    dtype="float16",
    paramwise_cfg=dict(custom_keys={"backbone.blocks": dict(lr_mult=0.1)}),
)
param_scheduler = [
    dict(type="LinearLR", start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type="PolyLR",
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    ),
]

data_root = "D:/datasets/segmentation/ADE/ADEChallengeData2016"

train_dataloader = dict(batch_size=4, num_workers=4, dataset=dict(data_root=data_root))
val_dataloader = dict(batch_size=1, num_workers=4, dataset=dict(data_root=data_root))
test_dataloader = val_dataloader

# This is needed to allow distributed training when some parameters
# have no gradient.
find_unused_parameters = True

vis_backends = [
    dict(type="LocalVisBackend", name="localvis"),
    dict(
        type="WandbVisBackend",
        init_kwargs=dict(
            project="seg_ade20k",
            resume="allow",
            id=None,
        ),
        name="wandbvis",
    ),
]

visualizer = dict(
    type="SegLocalVisualizer",
    vis_backends=vis_backends,
    name="visualizer",
    dataset_name="ade",
)

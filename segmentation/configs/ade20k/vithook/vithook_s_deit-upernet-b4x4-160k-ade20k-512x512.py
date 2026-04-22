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
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,  # Padding value. Default: 0.
    seg_pad_val=255,  # Padding value of segmentation map. Default: 255.
    size=crop_size,
)  # Fixed padding siz
num_classes = 150
pretrained = "pretrained_vit/deit_small_patch16_224-cd65a155.pth"
model = dict(
    type="EncoderDecoder",
    data_preprocessor=data_preprocessor,
    pretrained=pretrained,
    backbone=dict(
        type="ViTHook",
        crop_size=crop_size,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        drop_path_rate=0.2,
        num_hook=4,
        num_hook_heads=6,
        use_extra_crossattn=False,
        self_attn_drop=True,
        with_self_attn=True,
        with_cross_attn=True,
        use_self_flash_attn=True,
        use_cross_flash_attn=False,
        use_abs_pos=True,
        conv_inplane=64,
        n_points=4,
        msmlp_ratio=0.25,
        deform_ratio=1.0,
        with_msmlp=True,
        add_vit_feature=True,
        window_attn=[False] * 12,
        window_size=[None] * 12,
    ),
    decode_head=dict(
        type="UPerHead",
        pool_scales=(1, 2, 3, 6),
        in_channels=(384, 384, 384, 384),
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
        in_channels=384,
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
        type="AdamW", lr=6e-5, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8
    ),
    dtype="float16",
    constructor="CustomLayerDecayOptimizerConstructor",
    paramwise_cfg=dict(
        num_layers=12,
        layer_decay_rate=0.95,
    ),
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

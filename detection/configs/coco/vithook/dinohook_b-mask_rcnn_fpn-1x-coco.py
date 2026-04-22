_base_ = [
    "../../_base_/models/mask-rcnn_r50_fpn_amp.py",
    "../../_base_/datasets/coco_instance.py",
    "../../_base_/schedules/schedule_1x.py",
    "../../_base_/default_runtime.py",
]

custom_imports = dict(
    imports=[
        "mmdet_custom.models.backbones.vit_hooks.vit_hook",
        "mmdet_custom.optimizer_custom.custom_layer_decay_optimizer_constructor",
    ],
    allow_failed_imports=False,
)

# model settings
data_preprocessor = dict(
    type="DetDataPreprocessor",
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_mask=True,
    pad_size_divisor=32,
)
pretrained = "pretrained_vit/dinov2_vitb14_pretrain_14to16_torch17.pth"
model = dict(
    type="MaskRCNN_amp",
    data_preprocessor=data_preprocessor,
    backbone=dict(
        _delete_=True,
        type="ViTHook",
        pretrained=pretrained,
        crop_size=None,
        pretrain_size=592,
        img_size=592,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        # drop_path_rate=0.3,
        drop_path_rate=0.0,
        hook_drop_path_rate=0.3,
        is_frozen=True,
        ft_layers=(2, 5, 8, 11),
        num_hook=4,
        num_hook_heads=12,
        use_extra_crossattn=False,
        self_attn_drop=True,
        with_self_attn=True,
        use_self_flash_attn=True,
        use_cross_flash_attn=False,
        use_abs_pos=False,
        conv_inplane=64,
        n_points=4,
        msmlp_ratio=0.25,
        deform_ratio=0.5,
        with_msmlp=True,
        add_vit_feature=True,
        # window_attn=[True, True, False, True, True, False,
        #              True, True, False, True, True, False],
        # window_size=[14, 14, None, 14, 14, None,
        #              14, 14, None, 14, 14, None],
        window_attn=[False] * 12,
        window_size=[None] * 12,
    ),
    neck=dict(
        type="FPN", in_channels=[768, 768, 768, 768], out_channels=256, num_outs=5
    ),
)

optim_wrapper = dict(
    type="AmpOptimWrapper",
    optimizer=dict(type="AdamW", lr=1.0e-4, betas=(0.9, 0.999), weight_decay=0.05),
    dtype="float16",
    paramwise_cfg=dict(custom_keys={"backbone.blocks": dict(lr_mult=0.1)}),
    # constructor='CustomLayerDecayOptimizerConstructor',
    # paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.6,)
    # paramwise_cfg=dict(
    #     custom_keys={
    #     'backbone.level_embed': dict(decay_mult=0.),
    #     'backbone.pos_embed': dict(decay_mult=0.),
    #     'backbone.norm': dict(decay_mult=0.),
    #     'backbone.bias': dict(decay_mult=0.)}
    # )
)

data_root = "datasets/detection/COCO"

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
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

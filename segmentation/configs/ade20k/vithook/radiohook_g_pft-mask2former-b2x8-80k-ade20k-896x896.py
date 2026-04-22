_base_ = [
    "../../_base_/datasets/ade20k_896x896.py",
    "../../_base_/default_runtime.py",
    "../../_base_/schedules/schedule_80k.py",
]
custom_imports = dict(imports="mmdet.models", allow_failed_imports=False)
# model settings
crop_size = (896, 896)
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
num_things_classes = 100
num_stuff_classes = 50
num_classes = num_things_classes + num_stuff_classes

load_from = "./work_dir/ade20k/vithook/radiohook_g_frozen_activeblk4-mask2former-lr6e5-80k-drop05/iter_80000.pth"

model = dict(
    type="EncoderDecoder",
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type="RadioHook",
        model_version="radio_v2.5-g",
        is_frozen=True,
        num_hook=4,
        img_size=crop_size,
        with_self_attn=True,
        use_self_flash_attn=True,
        use_cross_flash_attn=False,
        use_extra_crossattn=False,
        use_abs_pos=True,
        active_block_indices=(9, 19, 29, 39),  # (3, 7, 11, 15, 19, 23, 27, 31)
        drop_path_rate=0.5,
        add_vit_feature=True,
        num_heads=24,
        conv_inplane=64,
        n_points=4,
        msmlp_ratio=0.25,
        deform_ratio=0.5,
        with_msmlp=True,
    ),
    decode_head=dict(
        type="Mask2FormerHead",
        in_channels=[1536, 1536, 1536, 1536],
        strides=[4, 8, 16, 32],
        feat_channels=1536,
        out_channels=1536,
        num_queries=200,
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        num_classes=num_classes,
        num_transformer_feat_level=3,
        align_corners=False,
        pixel_decoder=dict(
            type="mmdet.MSDeformAttnPixelDecoder",
            num_outs=3,
            norm_cfg=dict(type="GN", num_groups=32),
            act_cfg=dict(type="ReLU"),
            encoder=dict(  # DeformableDetrTransformerEncoder
                num_layers=6,
                layer_cfg=dict(  # DeformableDetrTransformerEncoderLayer
                    self_attn_cfg=dict(  # MultiScaleDeformableAttention
                        embed_dims=1536,
                        num_heads=24,
                        num_levels=3,
                        num_points=4,
                        im2col_step=64,
                        dropout=0.0,
                        batch_first=True,
                        norm_cfg=None,
                        init_cfg=None,
                    ),
                    ffn_cfg=dict(
                        embed_dims=1536,
                        feedforward_channels=4608,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type="ReLU", inplace=True),
                    ),
                ),
                init_cfg=None,
            ),
            positional_encoding=dict(  # SinePositionalEncoding
                num_feats=768, normalize=True
            ),
            init_cfg=None,
        ),
        enforce_decoder_input_project=False,
        positional_encoding=dict(  # SinePositionalEncoding
            num_feats=768, normalize=True
        ),
        transformer_decoder=dict(  # Mask2FormerTransformerDecoder
            return_intermediate=True,
            num_layers=9,
            layer_cfg=dict(  # Mask2FormerTransformerDecoderLayer
                self_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=1536,
                    num_heads=16,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=True,
                ),
                cross_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=1536,
                    num_heads=16,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=True,
                ),
                ffn_cfg=dict(
                    embed_dims=1536,
                    feedforward_channels=4608,
                    num_fcs=2,
                    act_cfg=dict(type="ReLU", inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    add_identity=True,
                ),
            ),
            init_cfg=None,
        ),
        loss_cls=dict(
            type="mmdet.CrossEntropyLoss",
            use_sigmoid=False,
            loss_weight=2.0,
            reduction="mean",
            class_weight=[1.0] * num_classes + [0.1],
        ),
        loss_mask=dict(
            type="mmdet.CrossEntropyLoss",
            use_sigmoid=True,
            reduction="mean",
            loss_weight=5.0,
        ),
        loss_dice=dict(
            type="mmdet.DiceLoss",
            use_sigmoid=True,
            activate=True,
            reduction="mean",
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0,
        ),
        train_cfg=dict(
            num_points=12544,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            assigner=dict(
                type="mmdet.HungarianAssigner",
                match_costs=[
                    dict(type="mmdet.ClassificationCost", weight=2.0),
                    dict(
                        type="mmdet.CrossEntropyLossCost", weight=5.0, use_sigmoid=True
                    ),
                    dict(type="mmdet.DiceCost", weight=5.0, pred_act=True, eps=1.0),
                ],
            ),
            sampler=dict(type="mmdet.MaskPseudoSampler"),
        ),
    ),
    # model training and testing settings
    train_cfg=dict(type="IterBasedTrainLoop", val_interval=2000),
    test_cfg=dict(
        type="TestLoop", mode="slide", crop_size=crop_size, stride=(512, 512)
    ),
)

optim_wrapper = dict(
    type="AmpOptimWrapper",
    optimizer=dict(
        type="AdamW", lr=1.2e-5, betas=(0.9, 0.999), weight_decay=1e-4, eps=1e-8
    ),
    dtype="float32",
    paramwise_cfg=dict(custom_keys={"backbone.blocks": dict(lr_mult=0.1)}),
)
param_scheduler = [
    dict(type="LinearLR", start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    # dict(
    #         type='CosineAnnealingLR',
    #         T_max=80000-1500,
    #         begin=1500,
    #         end=80000,
    #         by_epoch=False,
    #     )
    dict(
        type="PolyLR",
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=80000,
        by_epoch=False,
    ),
]

data_root = "D:/datasets/segmentation/ADE/ADEChallengeData2016"

train_dataloader = dict(batch_size=2, num_workers=2, dataset=dict(data_root=data_root))
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

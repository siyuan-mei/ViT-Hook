train_cfg = dict(
    type="EpochBasedTrainLoop", max_epochs=127, val_begin=1, val_interval=1
)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

log_processor = dict(by_epoch=True, log_with_hierarchy=True)
runner = dict(type="EpochBasedRunner", max_epochs=127)

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    runtime_info=dict(type="RuntimeInfoHook"),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    logger=dict(type="LoggerHook", interval=1, log_metric_by_epoch=True),
    checkpoint=dict(
        type="CheckpointHook",
        interval=1,
        by_epoch=True,
        save_best="mIoU",
        rule="greater",
        max_keep_ckpts=1,
    ),
    visualization=dict(
        type="SegVisualizationHook", draw=True, interval=100, show=False
    ),
)

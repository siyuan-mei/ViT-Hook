train_cfg = dict(type="IterBasedTrainLoop", max_iters=160000, val_interval=2000)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

log_processor = dict(by_epoch=False, log_with_hierarchy=True)
runner = dict(type="IterBasedRunner", max_iters=160000)

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    runtime_info=dict(type="RuntimeInfoHook"),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=False),
    checkpoint=dict(
        type="CheckpointHook",
        interval=1000,
        by_epoch=False,
        save_best="mIoU",
        rule="greater",
        max_keep_ckpts=1,
    ),
    visualization=dict(
        type="SegVisualizationHook", draw=True, interval=100, show=False
    ),
)

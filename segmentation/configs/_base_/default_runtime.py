default_scope = "mmseg"
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)
# log_processor = dict(by_epoch=True)
log_level = "INFO"
load_from = None  # Load checkpoint from file.
resume = False  # Whether to resume from existed mode

tta_model = dict(type="SegTTAModel")

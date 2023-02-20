_base_ = './yolov6_s_syncbn_fast_8xb32-300e_coco.py'

data_root = 'coco_tiny/'
dataset_type = 'YOLOv5CocoDataset'

deepen_factor = 0.33
widen_factor = 0.375

train_batch_size_per_gpu = 8
train_num_workers = 0

persistent_workers = False

# Base learning rate for optim_wrapper
base_lr = 0.01/8

model = dict(
    backbone=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    neck=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    bbox_head=dict(
        type='YOLOv6Head',
        head_module=dict(widen_factor=widen_factor),
        loss_bbox=dict(iou_mode='siou')))

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    persistent_workers=persistent_workers,
    dataset=dict(
        type=dataset_type,
        data_root=data_root
    )
)

val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root
    )
)

test_dataloader = val_dataloader


default_hooks = dict(
    logger=dict(interval=5)
)
# 开启混合精度训练
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        lr=base_lr
    )
)
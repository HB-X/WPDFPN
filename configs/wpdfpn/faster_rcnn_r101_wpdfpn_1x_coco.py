_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/default_runtime.py'
]
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
total_epochs = 12
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101),
             neck=dict(
                 type='WPDFPN',
                 in_channels=[256, 512, 1024, 2048],
                 out_channels=256,
                 num_outs=5)
             )

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    )
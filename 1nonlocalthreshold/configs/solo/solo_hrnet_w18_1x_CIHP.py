_base_=[
    '../_base_/models/solo_hrnet_w18.py',
    '../_base_/datasets/CIHP_humanparsing.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
    ]
model = dict(
    pretrained='open-mmlab://msra/hrnetv2_w18',
    backbone=dict(
        extra=dict(
            stage2=dict(num_channels=(18, 36)),
            stage3=dict(num_channels=(18, 36, 72)),
            stage4=dict(num_channels=(18, 36, 72, 144)))),
    neck=dict(type='HRFPN', in_channels=[18, 36, 72, 144], out_channels=256))

optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
total_epochs = 12
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8,11])
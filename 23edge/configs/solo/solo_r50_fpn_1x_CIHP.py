_base_=[
    '../_base_/models/solo_r50_fpn.py',
    '../_base_/datasets/CIHP_humanparsing.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
    ]

optimizer = dict(type='SGD', lr=0.004, momentum=0.9, weight_decay=0.0001)
total_epochs = 12
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8,11])
_base_=[
    '../_base_/models/solo_r50_fpn.py',
    '../_base_/datasets/MHP_humanparsing.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
    ]
model=dict(
    ins_head=dict(
        num_classes=58)#不算background
)
optimizer = dict(type='SGD', lr=0.008, momentum=0.9, weight_decay=0.0001)
total_epochs = 12
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.01,
    step=[9,11])
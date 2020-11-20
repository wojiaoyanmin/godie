_base_ = './solo_r50_fpn_1x_CIHP.py'
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))

optimizer = dict(type='SGD', lr=0.028, momentum=0.9, weight_decay=0.0001)
total_epochs = 12
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8,11])
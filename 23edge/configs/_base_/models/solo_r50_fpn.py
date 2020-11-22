model=dict(
    type='YMDetector',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0,1,2,3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        num_outs=5),#和下面的num_ints=4一起修改
    ins_head=dict(
        type='SOLOHead',
        num_classes=19,#不算background
        in_channels=256,
        stacked_convs=4,
        cate_stacked_convs=4,
        ins_feat_channels=512,
        cate_feat_channels=512,
        strides=[8, 8, 16, 32, 32],
        scale_ranges=((1, 48), (24, 96), (48, 192), (96, 384), (192,2048)),
        sigma=0.2,
        num_grids=[40, 36, 24, 16, 12],
        ins_out_channels=256,
        cate_feat_head=dict(
            type='CateFeatHead',
            in_channels=256,
            out_channels=512,
            out_edge_channels=256,
            start_level=0,
            end_level=4,
            num_classes=16*16,
            num_edge_classes=2,
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
            num_grid=64,
            stack_convs=2,
            stacked_edge_convs=1),
        mask_feat_head=dict(
            type='MaskFeatHead',
            in_channels=256,
            out_channels=128,
            start_level=0,
            end_level=3,
            num_classes=256,
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)),
        loss_ins=dict(
            type='DiceLoss',
            loss_weight=3.0),
        loss_cate=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_edge=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0),
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        ),
    
)
train_cfg=dict()
test_cfg=dict(
    nms_pre=500,
    score_thr=0.1,
    mask_thr=0.5,
    update_thr=0.1,#0.05,0.1
    kernel='gaussian',
    sigma=2.0,
    max_per_img=100)

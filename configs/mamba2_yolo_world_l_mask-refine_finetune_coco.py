_base_ = ['../third_party/mmyolo/configs/yolov8/'
          'yolov8_l_mask-refine_syncbn_fast_8xb16-500e_coco.py','mamba2_config.py']
custom_imports = dict(imports=['yolo_world'],
                      allow_failed_imports=False)

# hyper-parameters
num_classes = 80
num_training_classes = 80
max_epochs = 100  # Maximum training epochs
max_keep_ckpts = -1
close_mosaic_epochs = 10
save_epoch_intervals = 5
text_channels = _base_.TEXT_CHANNELS
text_expand = _base_.TEXT_EXPAND
base_lr = 2e-4
weight_decay = 0.05
train_batch_size_per_gpu = 16
val_batch_size_per_gpu = 4
val_num_workers = 0
copypaste_prob = 0.3

# model settings
model = dict(
    type='YOLOWorldDetector',
    mm_neck=True,
    num_train_classes=num_training_classes,
    num_test_classes=num_classes,
    data_preprocessor=dict(type='YOLOWDetDataPreprocessor'),
    backbone=dict(
        _delete_=True,
        type='MultiModalYOLOBackbone',
        image_model={{_base_.model.backbone}},
        text_model=dict(
            type='HuggingCLIPLanguageBackbone',
            model_name='openai/clip-vit-base-patch32',
            # frozen_modules=['all']
        ),
    ),
    neck=dict(type='MambaYOLOWorldPAFPN',
              guide_expand = text_expand,
              text_emb_dim = text_channels,
              text_extractor=_base_.mamba_cfg,
              block_cfg=dict(type='MambaFusionCSPLayerWithTwoConv2',
                             vss_cfg=_base_.vss_cfg,)
              ),
    bbox_head=dict(type='YOLOWorldHead',
                   head_module=dict(type='YOLOWorldHeadModule',
                                    use_bn_head=True,
                                    embed_dims=text_channels,
                                    num_classes=num_training_classes)),
    train_cfg=dict(assigner=dict(num_classes=num_training_classes))
)

# dataset settings
text_transform = [
    dict(type='RandomLoadText',
         num_neg_samples=(num_classes, num_classes),
         max_num_samples=num_training_classes,
         padding_to_max=True,
         padding_value=''),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                    'flip_direction', 'texts'))
]
train_pipeline = [
    *_base_.pre_transform,
    dict(type='MultiModalMosaic',
         img_scale=_base_.img_scale,
         pad_val=114.0,
         pre_transform=_base_.pre_transform),
    dict(type='YOLOv5CopyPaste', prob=copypaste_prob),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - _base_.affine_scale, 1 + _base_.affine_scale),
        max_aspect_ratio=_base_.max_aspect_ratio,
        border=(-_base_.img_scale[0] // 2, -_base_.img_scale[1] // 2),
        border_val=(114, 114, 114),
        min_area_ratio=_base_.min_area_ratio,
        use_mask_refine=_base_.use_mask2refine),
    *_base_.last_transform[:-1],
    *text_transform
]
train_pipeline_stage2 = [*_base_.train_pipeline_stage2[:-1], *text_transform]


coco_train_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(
        type='YOLOv5CocoDataset',
        data_root='data/coco',
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path='data/texts/coco_class_texts.json',
    pipeline=train_pipeline)

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    collate_fn=dict(type='yolow_collate'),
    dataset=coco_train_dataset)
test_pipeline = [
    *_base_.test_pipeline[:-1],
    dict(type='LoadText'),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'pad_param', 'texts'))
]
coco_val_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(
        type='YOLOv5CocoDataset',
        data_root='data/coco/',
        test_mode=True,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        batch_shapes_cfg=None,
        return_classes=True,
        ),
    class_text_path='data/texts/coco_class_texts.json',
    pipeline=test_pipeline,
)


val_dataloader = dict(dataset=coco_val_dataset,
                      pin_memory =False,
                      num_workers=val_num_workers,
                      persistent_workers=False,
                      batch_size = val_batch_size_per_gpu)
test_dataloader = val_dataloader


val_evaluator = dict(type='mmdet.CocoMetric',
                     ann_file='data/coco/annotations/instances_val2017.json',
                     metric='bbox')
test_evaluator = val_evaluator

# training settings
default_hooks = dict(
    param_scheduler=dict(
        scheduler_type='linear',
        lr_factor=0.01,
        max_epochs=max_epochs),
    checkpoint=dict(
        max_keep_ckpts=max_keep_ckpts,
        rule='greater',
        interval=save_epoch_intervals))
custom_hooks = [
    dict(type='EMAHook',
         ema_type='ExpMomentumEMA',
         momentum=0.0001,
         update_buffers=True,
         strict_load=False,
         priority=49),
    dict(type='mmdet.PipelineSwitchHook',
         switch_epoch=max_epochs - close_mosaic_epochs,
         switch_pipeline=train_pipeline_stage2)
]
train_cfg = dict(max_epochs=max_epochs,
                 val_interval=5,
                 dynamic_intervals=[((max_epochs - close_mosaic_epochs),
                                     _base_.val_interval_stage2)])


optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=base_lr,
        weight_decay=weight_decay,
        batch_size_per_gpu=train_batch_size_per_gpu),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'backbone.text_model': dict(lr_mult=0.01),
            'logit_scale': dict(weight_decay=0.0)
        }),
    constructor='YOLOWv5OptimizerConstructor',
    # clip_grad = dict(max_norm=1),
)

_base_ = ['../third_party/mmyolo/configs/yolov8/'
          'yolov8_s_syncbn_fast_8xb16-500e_coco.py','mamba2_config.py']
custom_imports = dict(imports=['yolo_world'],
                      allow_failed_imports=False)


# hyper-parameters
num_classes = 1203
num_training_classes = 80
max_epochs = 30  # Maximum training epochs
max_keep_ckpts = -1
close_mosaic_epochs = 2
save_epoch_intervals = 1
text_channels = _base_.TEXT_CHANNELS
text_expand = _base_.TEXT_EXPAND
base_lr = 2e-4
weight_decay = 0.05
train_batch_size_per_gpu = 16
val_batch_size_per_gpu = 4
val_num_workers = 0

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
            frozen_modules=['all']),
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
    train_cfg=dict(assigner=dict(num_classes=num_training_classes)),
    test_cfg=dict(max_per_img=1000)
)

# dataset settings
text_transform = [
    dict(type='RandomLoadText',
         num_neg_samples=(num_classes, num_classes),
         max_num_samples=num_training_classes,
         padding_to_max=True,
         padding_value=''),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip','img',
                    'flip_direction', 'texts'))
]
train_pipeline = [
    *_base_.pre_transform,
    dict(type='MultiModalMosaic',
         img_scale=_base_.img_scale,
         pad_val=114.0,
         pre_transform=_base_.pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - _base_.affine_scale, 1 + _base_.affine_scale),
        max_aspect_ratio=_base_.max_aspect_ratio,
        border=(-_base_.img_scale[0] // 2, -_base_.img_scale[1] // 2),
        border_val=(114, 114, 114)),
    *_base_.last_transform[:-1],
    *text_transform,
]
train_pipeline_stage2 = [*_base_.train_pipeline_stage2[:-1], *text_transform]

obj365v1_subset_train_dataset = dict(
    type='MultiModalDataset',
    dataset=dict(
        type='YOLOv5Objects365V1Dataset',
        data_root='data/objects365v1/',
        ann_file='objects365_train_subset.json',
        data_prefix=dict(img='train/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path='data/texts/obj365v1_class_texts.json',
    pipeline=train_pipeline)

obj365v1_train_dataset = dict(
    type='MultiModalDataset',
    dataset=dict(
        type='YOLOv5Objects365V1Dataset',
        data_root='data/objects365v1/',
        ann_file='objects365_train.json',
        data_prefix=dict(img='train/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path='data/texts/obj365v1_class_texts.json',
    pipeline=train_pipeline)

mg_train_dataset = dict(type='YOLOv5MixedGroundingDataset',
                        data_root='data/mixed_grounding/',
                        ann_file='final_mixed_train_no_coco.json',
                        data_prefix=dict(img='images/'),
                        filter_cfg=dict(filter_empty_gt=False, min_size=32),
                        pipeline=train_pipeline)

flickr_train_dataset = dict(
    type='YOLOv5MixedGroundingDataset',
    data_root='data/flickr/',
    ann_file='final_flickr_separateGT_train.json',
    data_prefix=dict(img='flickr30k_images/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=train_pipeline)

train_dataloader = dict(batch_size=train_batch_size_per_gpu,
                        collate_fn=dict(type='yolow_collate'),
                        dataset=dict(_delete_=True,
                                     type='ConcatDataset',
                                     datasets=[
                                         # obj365v1_subset_train_dataset,
                                         obj365v1_train_dataset,
                                         flickr_train_dataset,
                                         mg_train_dataset
                                     ],
                                     ignore_keys=['classes', 'palette']))

test_pipeline = [
    *_base_.test_pipeline[:-1],
    dict(type='LoadText'),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape','img',
                    'scale_factor', 'pad_param', 'texts'))
]


lvis_val_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(type='YOLOv5LVISV1Dataset',
                 data_root='data/coco/',
                 test_mode=True,
                 ann_file='lvis/lvis_v1_minival_inserted_image_name.json',
                 data_prefix=dict(img=''),
                 batch_shapes_cfg=None,
                 ),
    class_text_path='data/texts/lvis_v1_class_texts.json',
    pipeline=test_pipeline)

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

val_dataloader = dict(dataset=lvis_val_dataset,
                      pin_memory=False,
                      num_workers=val_num_workers,
                      persistent_workers = False,
                      batch_size = val_batch_size_per_gpu)
# val_dataloader = dict(dataset=coco_val_dataset,
#                       pin_memory =False,
#                       num_workers=val_num_workers,
#                       persistent_workers=False,
#                       batch_size = val_batch_size_per_gpu)
test_dataloader = val_dataloader

val_evaluator = dict(
    _delete_ = True,
    type='mmdet.LVISFixedAPMetric',
    ann_file='data/coco/lvis/lvis_v1_minival_inserted_image_name.json',
)
# val_evaluator = dict(type='mmdet.CocoMetric',
#                      ann_file='data/coco/annotations/instances_val2017.json',
#                      metric='bbox')
test_evaluator = val_evaluator

# training settings
default_hooks = dict(param_scheduler=dict(max_epochs=max_epochs),
                     checkpoint=dict(interval=save_epoch_intervals,
                                     rule='greater',
                                     max_keep_ckpts=max_keep_ckpts))
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
                 val_interval=1,
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
        bias_decay_mult=0.0,
        norm_decay_mult=0.0,
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'logit_scale': dict(weight_decay=0.0)
        }),
    constructor='YOLOWv5OptimizerConstructor',
    # clip_grad = dict(max_norm=1),
)

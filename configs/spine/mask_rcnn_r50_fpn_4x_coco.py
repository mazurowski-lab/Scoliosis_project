_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1)))

runner = dict(type='EpochBasedRunner', max_epochs=60)

# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('Spine',)

data = dict(
    train=dict(
        img_prefix='../detection_images/train/',
        classes=classes,
        ann_file='../raw_train_label_new.json'),
    val=dict(
        img_prefix='../detection_images/val/',
        classes=classes,
        ann_file='../raw_val_label_new.json'),
    test=dict(
        img_prefix='../detection_images/val/',
        classes=classes,
        ann_file='../raw_val_label_new.json'))
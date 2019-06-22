## GET STARTED AGAGIN

**NOTE**: This code is based on [HRNet-Object-Detecton](https://github.com/HRNet/HRNet-Object-Detection)

### Comparison with original code

- [X] **SyncBN**: using NVIDIA/apex SyncBN

- [X] **Multi-scale training (fixed)**: training detection models with SyncBN and multi-scale training will crash to terrible results (mAP=0.0 or no boxes are predicted). Fortunatedly, we've investigated a method to solve it by padding input images of different scales(`600*1000`, `800*1333`, `1000*1600`) to a fixed scale `**1000*1600**` and keeping the original aspect ratios.

- [X] **Multi-scale training (SimpleDet version)**: we've implemented multi-scale training strategy used in [SimpleDet](https://github.com/TuSimple/simpledet)

- [X] **Multi-node & multi-gpu training**: we've tested our code when training with multiple nodes (ONLY on AZURE!). Providing a MASTER IP and PORT. Training without SyncBN will reach normal results while training with SyncBN will fail.

### Start

##### SyncBN

* ResNet:

see [configs/syncbn/faster_rcnn_r50_fpn_sync_1x.py](configs/syncbn/faster_rcnn_r50_fpn_sync_1x.py)

change `normalize` in model config:

````python
normalize = dict(type='SyncBN', frozen=False)
````

* HRNet

HRNet in this repo doesn't support `normalize` but HRNet in [mmdetection](https//github.com/open-mmlab/mmdetection) supports it.

see [configs/hrnet/faster_rcnn_hrnetv2p_w18_sync_1x.py](configs/hrnet/faster_rcnn_hrnetv2p_w18_sync_1x.py)

change `backbone.type` to `SyncHighResolutionNet`.


#### Multi-scale training (fixed)

see [configs/hrnet/faster_rcnn_hrnetv2p_w18_syncbn_16batch_mstrain_pad_1x.py](configs/hrnet/faster_rcnn_hrnetv2p_w18_syncbn_16batch_mstrain_pad_1x.py)

1. set maximum padding size `pad_size`
2. set scales for multi-scale training.

````python
data = dict(
    imgs_per_gpu=4,
    workers_per_gpu=8,
    pad_size=(1600, 1024),
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'images/train2017.zip',
        img_scale=[(1600, 1000), (1000, 600), (1333, 800)],
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0.5,
        with_mask=False,
        with_crowd=True,
        with_label=True),
````

#### Multi-scale training (SimpleDet version)

see [configs/hrnet/faster_rcnn_hrnetv2p_w18_randresizecrop_1x.py](configs/hrnet/faster_rcnn_hrnetv2p_w18_randresizecrop_1x.py)

1. set maximum padding size `pad_size=(1200,800)`
2. set scales for multi-scale training.
3. add extra data augmentation

````python
    imgs_per_gpu=2,
    workers_per_gpu=4,
    pad_img=(1216, 800),
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017.zip',
        img_scale=(1200, 800),
        img_norm_cfg=img_norm_cfg,
        size_divisor=1,
        extra_aug=dict(
            rand_resize_crop=dict(
                scales=[[1400, 600], [1400, 800], [1400, 1000]],
                size=[1200, 800]
            )),
        flip_ratio=0.5,
        with_mask=False,
        with_crowd=True,
        with_label=True),
````

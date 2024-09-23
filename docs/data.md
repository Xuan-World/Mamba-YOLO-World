## Preparing Data for Mamba-YOLO-World

### Overview

For pre-training Mamba-YOLO-World, we adopt several datasets as listed in the below table:

| Data | Samples | Type | Boxes  |
| :-- | :-----: | :---:| :---: | 
| Objects365v1 | 609k | detection | 9,621k |
| GQA | 621k | grounding | 3,681k |
| Flickr | 149k | grounding | 641k |
 
### Dataset Directory

We put all data into the `data` directory, such as:

```bash
├── coco
│   ├── annotations
│   │   ├── instances_val2017.json
│   │   └── instances_train2017.json
│   ├── lvis
│   │   └── lvis_v1_minival_inserted_image_name.json
│   ├── train2017
│   └── val2017
├── flickr
│   ├── final_flickr_separateGT_train.json
│   └── images
├── mixed_grounding
│   ├── final_mixed_train_no_coco.json
│   ├── images
├── objects365v1
│   ├── objects365_train.json
│   └── train
└── texts
```
**NOTE**: We strongly suggest that you check the directories or paths in the dataset part of the config file, especially for the values `ann_file`, `data_root`, and `data_prefix`.

We provide the annotations of the pre-training data in the below table:

| Data | images | Annotation File |
| :--- | :------| :-------------- |
| Objects365v1 | [`Objects365 train`](https://opendatalab.com/OpenDataLab/Objects365_v1) | [`objects365_train.json`](https://opendatalab.com/OpenDataLab/Objects365_v1) |
| MixedGrounding | [`GQA`](https://nlp.stanford.edu/data/gqa/images.zip) | [`final_mixed_train_no_coco.json`](https://huggingface.co/GLIPModel/GLIP/tree/main/mdetr_annotations/final_mixed_train_no_coco.json) |
| Flickr30k | [`Flickr30k`](https://shannon.cs.illinois.edu/DenotationGraph/) |[`final_flickr_separateGT_train.json`](https://huggingface.co/GLIPModel/GLIP/tree/main/mdetr_annotations/final_flickr_separateGT_train.json) |
| LVIS-minival | [`COCO val2017`](https://cocodataset.org/) | [`lvis_v1_minival_inserted_image_name.json`](https://huggingface.co/GLIPModel/GLIP/blob/main/lvis_v1_minival_inserted_image_name.json) |

**Acknowledgement:** We sincerely thank [GLIP](https://github.com/microsoft/GLIP) and [mdetr](https://github.com/ashkamath/mdetr) for providing the annotation files for pre-training.


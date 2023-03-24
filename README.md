# Zero-Shot Everything Sketch-Based Image Retrieval, and in Explainable Style
![Python 3.6](https://img.shields.io/badge/python-3.6-green) ![Pytorch 1.10](https://img.shields.io/badge/pytorch-1.10-green) ![MIT License](https://img.shields.io/badge/licence-MIT-green)

In this repository, you can find the official PyTorch implementation of [Zero-Shot Everything Sketch-Based Image Retrieval, and in Explainable Style](), CVPR2023, Highlight.

Authors: [Fengyin Lin](https://github.com/buptLinfy), [Mingkang Li](https://github.com/buptlmk), [Da Li](https://scholar.google.co.uk/citations?user=RPvaE3oAAAAJ&hl=en), [Timothy Hospedales](https://scholar.google.co.uk/citations?user=nHhtvqkAAAAJ&amp;hl=en), [Yi-Zhe Song](https://scholar.google.co.uk/citations?hl=en&user=irZFP_AAAAAJ&view_op=list_works&sortby=pubdate) and [Yonggang Qi](https://qugank.github.io/), Beijing University of Posts and Telecommunications, Samsung AI Centre Cambridge, University of Edinburgh, SketchX CVSSP University of Surrey.

> Abstract: This paper studies the problem of zero-short sketch-based image retrieval (ZS-SBIR), however with two significant differentiators to prior art (i) we tackle all variants (inter-category, intra-category, and cross datasets) of ZS-SBIR with just one network (“everything”), and (ii) we would really like to understand how this sketch-photo matching operates (“explainable”). Our key innovation lies with the realization that such a cross-modal matching problem could be reduced to comparisons of groups of key local patches – akin to the seasoned “bag-of-words” paradigm. Just with this change, we are able to achieve both of the aforementioned goals, with the added benefit of no longer requiring external semantic knowledge. Technically, ours is a transformer-based cross-modal network, with three novel
components (i) a self-attention module with a learnable tokenizer to produce visual tokens that correspond to the most informative local regions, (ii) a cross-attention module to compute local correspondences between the visual tokens across two modalities, and finally (iii) a kernel-based relation network to assemble local putative matches and produce an overall similarity metric for a sketch-photo pair. Experiments show ours indeed delivers superior performances across all ZS-SBIR settings. The all important explainable goal is elegantly achieved by visualizing cross-modal token correspondences, and for the first time, via sketch to photo synthesis by universal replacement of all matched photo patches.

![Fig.1](./images/overview.png)

## Datasets
Please download SBIR datasets from the official websites and unzip each dataset to the corresponding directory in `./datasets`. We provide train and test splits for different datasets.

### Sketchy
Please go to the [Sketchy official website](https://sketchy.eye.gatech.edu/) to download the _Sketches and Photos_ datasets.

### TU-Berlin
Please go to the [TU-Berlin official website](http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/) to download the _Sketches and Photos_ datasets.

### QuickDraw
Please go to the [QuickDraw official website](https://github.com/googlecreativelab/quickdraw-dataset) to download the datasets. The original data format is vector, please convert it to `.png` or `.jpg` before use it.

## Installation
The requirements of this repo can be found in `requirements.txt`.

```bash
pip install -r requirements.txt
```

## Train

### Pretrained ViT backbone

The pre-trained ViT model on ImageNet-1K is provided on [Google Drive](https://drive.google.com/file/d/1bznKsXDM5-xaUR9suCBBc7J33lIa70zJ/view?usp=sharing). You should place `sam_ViT-B_16.pth` in `./model` and modify line 190 in `./model/sa.py` to absolute path if necessary.

### Haperparameters
Here is a list of full options for the model:
```bash
# dataset
data_path,            # path to load datasets.
dataset,              # choose a dataset for train or eval.
test_class,           # choose a zero-shot split of dataset.

# model
cls_number,           # class number if necessary, 100 as default.
d_model,              # feature dimension, 768 as default.
d_ff,                 # fead-forward layer dimension, 1024 as default.
head,                 # number of ca encoder head, 8 as default.
number,               # number of ca encoder layer, 1 as default.
pretrained,           # whether to use pretrained ViT model, true as default.
anchor_number,        # number of anchor in rn network, 49 as default.

# train
save, -s,             # path to save checkpoints.
batch,                # batch size, 15 as default.
epoch,                # train epoch, 30 as default.
datasetLen,           # data pair for train per epoch, 10000 as default.
learning_rate,        # learning rate, 1e-5 as default.
weight_decay,         # weight decay, 1e-2 as default.

# test
load, -l,             # path to load checkpoints.
retrieval, -r,        # test method, rn for ret-token and sa for cls-token, use rn as default.
testall,              # whether use all test data, suggesting false for train, true for test.
test_sk,              # number of sketches per loop during test, 20 as default.
test_im,              # number of images per loop during test, 20 as default.
num_workers,          # dataloader num workers, 4 as default.

# other
choose_cuda, -c,      # cuda to use, 0 as default.
seed,                 # random seed, 2021 as default.
```

### Train ZSE-SBIR

Here is a quick start for training the network on Sketchy Ext. Please pay attention to modifying data path and save path before run.
```bash
python -u train.py 
# or use nohup command
nohup python -u train.py > sketchy_ext.log 2>&1 &
```

Train model on Sketchy Ext.
```bash
python -u train.py --data_path [./datasets] \
                   --dataset sketchy_extend \ 
                   --test_class test_class_sketchy25 \ 
                   --batch 15 \ 
                   --epoch 30 \ 
                   -s [./checkpoints/sketchy_ext] \
                   -c 0 \ 
                   -r rn 
```

Train model on TU-Berlin Ext.
```bash
python -u train.py --data_path [./datasets] \
                   --dataset tu_berlin \ 
                   --test_class test_class_tuberlin30 \ 
                   --batch 15 \ 
                   --epoch 30 \ 
                   -s [./checkpoints/tuberlin_ext] \
                   -c 0 \ 
                   -r rn \ 
```

Train model on QuickDraw Ext.
```bash
python -u train.py --data_path [./datasets] \
                   --dataset Quickdraw \ 
                   --test_class Quickdraw \ 
                   --batch 15 \ 
                   --epoch 30 \ 
                   -s [./checkpoints/quickdraw_ext] \
                   -c 0 \ 
                   -r rn 
```

## Evaluation

### Our Trained Model
The trained model on Sketchy Ext is provided on [Google Drive](https://drive.google.com/file/d/16HAlzuibGoQhhozcz4_vVO3rGZEyLqcw/view?usp=sharing). You should place `best_checkpoint.pth` in `./checkpoint/sketchy_ext` and modify load path `--load` for example.


### Evaluate ZSE-SBIR

Here is a quick start for evaluating the network on Sketchy Ext. Please pay attention to modifying data path and save path before run.
```bash
# use ret-token for zs-sbir, and use all test data.
python -u test.py -r rn -- testall
# use cls-token for zs-sbir, which is quite faster.
python -u test.py -r sa -- testall
# or use nohup command
nohup python -u test.py -r rn --testall > test_sketchy_ext.log 2>&1 &
```

Evaluate model on Sketchy Ext.
```bash
python -u test.py --data_path [./datasets] \
                  --dataset sketchy_extend \
                  --test_class test_class_sketchy25 \ 
                  -l [./checkpoints/sketchy_ext/best_checkpoint.pth] \
                  -c 0 \ 
                  -r rn \ 
                  --testall
```

Evaluate model on TU-Berlin Ext.
```bash
python -u test.py --data_path [./datasets] \
                  --dataset tu_berlin \
                  --test_class test_class_tuberlin30 \ 
                  -l [./checkpoints/tuberlin_ext/best_checkpoint.pth] \
                  -c 0 \ 
                  -r rn \ 
                  --testall
```

Evaluate model on QuickDraw Ext.
```bash
python -u test.py --data_path [./datasets] \
                  --dataset Quickdraw \
                  --test_class Quickdraw \ 
                  -l [./checkpoints/quickdraw_ext/best_checkpoint.pth] \
                  -c 0 \ 
                  -r rn \ 
                  --testall
```

## License
This project is released under the [MIT License](./LICENSE).

## Citation
If you find this repository useful for your research, please use the following.
```
@inproceedings{
zse-sbir-cvpr2023,
title={Zero-Shot Everything Sketch-Based Image Retrieval, and in Explainable Style},
author={Fengyin Lin, Mingkang Li, Da Li, Timothy Hospedales, Yi-Zhe Song and Yonggang Qi},
booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
year={2023}
}
```

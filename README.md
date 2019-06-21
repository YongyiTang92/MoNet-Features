# Hallucinating Optical Flow Features for Video Classification
by Yongyi Tang, Lin Ma and Lianqiang Zhou. Accepted by IJCAI 2019.

### Introduction
Extracting motion information, specifically in the form of optical flow features, is extremely computationally expensive.
We propose a motion hallucination network to imagine the optical flow features from the appearance features for video classification.
For more details, please refer to our [paper](https://arxiv.org/abs/1905.11799).

![image](https://github.com/YongyiTang92/MoNet-Features/master/imgs/motivation.jpg)

### Citation

    @InProceedings{tang2019hallucinating,
      author = {Yongyi Tang and Ma, Lin and Lianqiang Zhou},
      title = {Hallucinating Optical Flow Features for Video Classification},
      booktitle = {IJCAI},
      year = {2019}
    }

### Overview

This repository contains trained models and feature reported in the our paper on the Kinetics-400 dataset.

### Sample code

Run the example code using

```
$ python extract_monet_features.py
```

With default flags, this builds the I3D-MoNet model which takes a video segment as input, and produce corresponding hallucinated motion features.

The correspoding checkpoint can be downed from [google-drive](https://drive.google.com/open?id=1ExYBlKP-j5Q_x9RL1md0nij_c8OR2M9F) or [weiyun](https://share.weiyun.com/5mmgLyU)

### Loading pre-extracted features

We provide the I3D-rgb I3D-flow and the MoNet-flow features in the form of [tfrecords](https://share.weiyun.com/53llmR0).

Run the example code of loading tfrecords using

```
$ python feature_reader.py
```

after modifing the file paths.

### Credits
Part of the code is from [kinetics-i3d](https://github.com/deepmind/kinetics-i3d)

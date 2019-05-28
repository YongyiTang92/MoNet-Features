# MoNet models and features trained on Kinetics-400

## Overview

This repository contains trained models and feature reported in the paper "Hallucinating Optical Flow Features for Video Classification" by Yongyi Tang, Lin Ma and Lianqiang Zhou.
The paper will be published as an IJCAI 2019 conference paper.

### Sample code

Run the example code using

`$ python extract_monet_features.py`

With default flags, this builds the I3D-MoNet model which takes a video segment as input, and produce corresponding hallucinated motion features.

The correspoding checkpoint can be downed from [google-drive](https://drive.google.com/open?id=1ExYBlKP-j5Q_x9RL1md0nij_c8OR2M9F) or [weiyun](https://share.weiyun.com/53llmR0)

### Loading pre-extracted features

We provide the I3D-rgb I3D-flow and the MoNet-flow features in the form of [tfrecords](https://share.weiyun.com/5mmgLyU).

Run the example code of loading tfrecords using

`$ python feature_reader.py`

after modifing the file paths.
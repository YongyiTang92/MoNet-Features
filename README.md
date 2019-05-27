# MoNet models and features trained on Kinetics-400

## Overview

This repository contains trained models and feature reported in the paper "Hallucinating Optical Flow Features From Appearance Features for VideoClassification" by Yongyi Tang, Lin Ma and Lianqiang Zhou.
The paper will be published as an IJCAI 2019 conference paper.

### Sample code

Run the example code using

`$ python extract_monet_features.py`

With default flags, this builds the I3D-MoNet model which takes a video segment as input, and produce corresponding hallucinated motion features.

### Loading pre-extracted features

We provide the I3D-rgb I3D-flow and the MoNet-flow features in the following files in the form of tfrecords.

Run the example code of loading tfrecords using

`$ python feature_reader.py`

after modifing the file paths.
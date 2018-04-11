# Transfer Learning Lab with VGG, Inception and ResNet
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

## Overview

We use Keras to explore feature extraction with the VGG, Inception and ResNet architectures. The models used were trained for days or weeks on the [ImageNet dataset](http://www.image-net.org/). Thus, the weights encapsulate higher-level features learned from training on thousands of classes.

To perform feature extration we use two datasets:

1. [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=news)
2. [Cifar10](https://www.cs.toronto.edu/~kriz/cifar.html)

Without a powerful GPU, running feature extraction on these models will take a significant amount of time. To make things easier we precomputed **bottleneck features** for each (network, dataset) pair, this will allow experimenting with feature extraction even on a modest CPU. Because the base network weights are frozen during feature extraction, the output for an image will always be the same. Thus, once the image has already been passed once through the network we can cache and reuse the output.

The training and validation files are encoded as such:

- {network}_{dataset}_bottleneck_features_train.p
- {network}_{dataset}_bottleneck_features_validation.p

Where:

- **network** can be one of 'vgg', 'inception', or 'resnet'

- **dataset** can be on of 'cifar10' or 'traffic'

## Getting Started

The project requires various libraries to be installed in your environment first, in particular:

- Python 3
- TensorFlow
- Keras
- NumPY
- SciPy

You'll also need the bottleneck features for the various networks:

- [VGG Features](http://video.udacity-data.com.s3.amazonaws.com/topher/2016/November/5834b432_vgg-100/vgg-100.zip)
- [ResNet Features](http://video.udacity-data.com.s3.amazonaws.com/topher/2016/November/5834b634_resnet-100/resnet-100.zip)
- [Inception Features](http://video.udacity-data.com.s3.amazonaws.com/topher/2016/November/5834b498_inception-100/inception-100.zip)

## Feature Extraction

The [feature_extraction.py]{./feature_extraction.py} will train feature extraction with the files in input, in particular it accepts the following parameters:

- --training_file: The pickle file to use for training
- --validation_file: The pickle file to use for validation
- --epochs: The number of epochs to train for (default 50)
- --batch_size: The size of the minibatch (default 256)

For example:

```bash
python feature_extraction.py \
--training_file vgg_cifar10_100_bottleneck_features_train.p \
--validation_file vgg_cifar10_bottleneck_features_validation.p
```

Would train the VGG network with the Cifar10 dataset bottleneck features. The 100 in vgg_cifar10_100 indicates this file has 100 examples per class (See [shrink.py](./shrink/py)):

```bash
Epoch 50/50
1000/1000 [==============================] - 0s - loss: 0.2358 - acc: 0.9510 - val_loss: 0.8924 - val_acc: 0.7102
```

Running the same with the inception and the resnet we would get a similar output:

```bash
python feature_extraction.py \
--training_file inception_cifar10_100_bottleneck_features_train.p \
--validation_file inception_cifar10_bottleneck_features_validation.p

Epoch 50/50
1000/1000 [==============================] - 0s - loss: 0.0897 - acc: 1.0000 - val_loss: 1.0422 - val_acc: 0.6608

python feature_extraction.py \
--training_file resnet_cifar10_100_bottleneck_features_train.p \
--validation_file resnet_cifar10_bottleneck_features_validation.p

Epoch 50/50
1000/1000 [==============================] - 0s - loss: 0.0755 - acc: 1.0000 - val_loss: 0.8027 - val_acc: 0.7315

```

We can do the same on the german traffic sign dataset (e.g. in the following using ResNet):

```bash
python feature_extraction.py \
--training_file resnet_traffic_100_bottleneck_features_train.p \
--validation_file resnet_traffic_bottleneck_features_validation.p

Epoch 50/50
4300/4300 [==============================] - 0s - loss: 0.0327 - acc: 1.0000 - val_loss: 0.6112 - val_acc: 0.8103
```

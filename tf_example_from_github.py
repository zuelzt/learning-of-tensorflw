#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 10:21:13 2018

@author: Rorschach
@mail: 188581221@qq.com
"""
import warnings
warnings.filterwarnings('ignore')

# import MNIST
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# split dataset
X_train = mnist.train.images
X_test = mnist.test.images
y_train = mnist.train.images
y_test = mnist.test.images

# batch size 64
batch_X, batch_y = mnist.train.next_batch(64)












































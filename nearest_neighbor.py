#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 09:59:55 2018

@author: Rorschach
@mail: 188581221@qq.com
"""
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# import data
mnist = input_data.read_data_sets("/Users/zt/Desktop/Master File/practice of python/MNIST_data/", one_hot=True)

# get part of data
train_X ,train_y = mnist.train.next_batch(10000)
test_X, test_y = mnist.test.next_batch(1000)

# imput
with tf.name_scope('Input'):
    tr_X = tf.placeholder(tf.float32, shape=[None, 28 * 28])
    te_X = tf.placeholder(tf.float32, shape=[28 * 28])
    y = tf.placeholder(tf.float32, shape=[10])

# distance
with tf.name_scope('Distance'):
    dis1 = tf.reduce_sum(tf.abs(tf.add(tr_X, tf.negative(te_X))), reduction_indices=1)
    pred_class = tf.arg_min(dis1, 0)


init = tf.global_variables_initializer()
sess = tf.Session()

# train
sess.run(init)
for step in range(len(test_y)):
   
    print('pred_class:{0}, true_class:{1}, accuracy:{2:3f}'.format(sess.run(pred_class), sess.run(true_class), accuracy))






































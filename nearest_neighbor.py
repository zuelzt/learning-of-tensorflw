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

# distance
with tf.name_scope('Distance'):
    dis_1 = tf.reduce_sum(tf.abs(tf.add(tr_X, tf.negative(te_X))), reduction_indices=1)  
    pred_class_1 = tf.arg_min(dis_1, 0)  # 曼哈顿距离
    
    dis_2 = tf.sqrt(tf.reduce_sum(tf.multiply(tf.add(tr_X, tf.negative(te_X)), tf.add(tr_X, tf.negative(te_X))), reduction_indices=1))
    pred_class_2 = tf.arg_min(dis_2, 0)  # 欧式距离
    
# =============================================================================
#     dis_3 = tf.div(tf.matmul(tr_X, te_X), tf.multiply(tf.sqrt(tf.reduce_sum(tf.pow(tr_X, 2), reduction_indices=1), tf.sqrt(tf.reduce_sum(tf.pow(te_X, 2))))))
#     pred_class_3 = tf.arg_min(dis_3, 0)  # 夹角余弦
# =============================================================================

init = tf.global_variables_initializer()
sess = tf.Session()
accuracy_1 = 0
accuracy_2 = 0

# train
sess.run(init)
for step in range(len(test_y)):
    p_class_1, p_class_2 = sess.run([pred_class_1, pred_class_2], feed_dict={tr_X: train_X,
                                              te_X: test_X[step, :]})
    prediction_1 = np.argmax(train_y[p_class_1])
    prediction_2 = np.argmax(train_y[p_class_2])

    true = np.argmax(test_y[step])  # 真实值
    
    accuracy_1 += int(prediction_1 == true)
    accuracy_2 += int(prediction_2 == true)

    print('step {0} ---pred_class:{1}, true_class:{2}, accuracy:{3:.3f}'.format(step + 1, prediction_2, true, accuracy_2 / (step + 1)))






































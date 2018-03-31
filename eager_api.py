#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 16:57:24 2018

@author: Rorschach
@mail: 188581221@qq.com
"""
import warnings
warnings.filterwarnings('ignore')


import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np

# set eager API
tfe.enable_eager_execution()  # only be called once at program startup.

# have a try
a = tf.constant(2)
print("a = {}".format(a))

b = tf.constant(3)
c = a + b
print("a + b = {}".format(c))

# we can use Tensors with Numpy Arrays.
a = tf.constant([[2, 1],
                 [1, 2]], dtype=tf.float32)
b = np.array([[1, 1],
              [1, 1]], dtype=np.float32)

c = a + b
d = tf.matmul(a, b)
print("c = {}".format(c))
print("d = {}".format(d))

# iterate through
for i in range(a.shape[0]):
    for j in range(a.shape[1]):
        print(a[i][j])




























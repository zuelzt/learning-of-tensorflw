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






























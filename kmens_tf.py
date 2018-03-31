#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 20:47:39 2018

@author: Rorschach
@mail: 188581221@qq.com
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf
from tensorflow.contrib.factorization import KMeans

# import dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
full_data_x = mnist.train.images

# set parameters
num_steps = 50  #step
batch_size = 1024  #batch
k = 25  # clsuters
num_classes = 10  # 0, 1, 2, ..... ,9, 10
num_features = 28 * 28  # features

# set input
X = tf.placeholder(tf.float32, shape=[None, num_features])
y = tf.placeholder(tf.float32, shape=[None, num_classes])

# K-Means
kmeans = KMeans(inputs=X, num_clusters=k, distance_metric='cosine', use_mini_batch=True)
# 夹角余弦距离， use_mini_batch 是一种每一次迭代随机抽取较小样本进行聚类的大数据处理方法，
# 在不损失过多精确度的前提下提高速度

# build
(all_scores, cluster_idx, scores, cluster_centers_initialized, 
 cluster_centers_vars,init_op,train_op) = kmeans.training_graph()
# all_scores: all distance
# cluster_idx: model predictions
#scores: losses
#cluster_centers_initialized: scalar indicating whether the initial cluster centers have been chosen
#cluster_centers_vars: a Variable containing the cluster centers
#init_op: an op to choose the initial cluster centers
#train_op: an op that runs an iteration of training
cluster_idx = cluster_idx[0]  # be a tuple
avg_distance = tf.reduce_mean(scores)



























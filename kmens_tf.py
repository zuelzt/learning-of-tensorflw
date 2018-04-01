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
mnist = input_data.read_data_sets("/Users/zt/Desktop/Master File/practice of python/MNIST_data", one_hot=True)
full_data_x = mnist.train.images

# set parameters
num_steps = 100  #step
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
(all_scores, cluster_idx, scores, cluster_centers_initialized,#cluster_centers_vars,
 init_op, train_op) = kmeans.training_graph()
# all_scores: all distance
# cluster_idx: model predictions
#scores: losses
#cluster_centers_initialized: scalar indicating whether the initial cluster centers have been chosen
#cluster_centers_vars: a Variable containing the cluster centers (now delete)
#init_op: an op to choose the initial cluster centers
#train_op: an op that runs an iteration of training
cluster_idx = cluster_idx[0]  # be a tuple
avg_distance = tf.reduce_mean(scores)

# initialize
init = tf.global_variables_initializer()  # all_variables_initializer() is delete

# start seession
sess = tf.Session()
sess.run(init)
sess.run(init_op, feed_dict={X: full_data_x})

# training
for i in range(num_steps + 1):
    _, d, idx = sess.run([train_op, avg_distance, cluster_idx],
                         feed_dict={X: full_data_x})
    if i % 10 == 0 or i == 1:
        print("Step {0} : Avg-distance = {1:.4f}".format(i, d))

counts = np.zeros(shape=(k, num_classes))
for i in range(len(idx)):
    counts[idx[i]] += mnist.train.labels[i]
labels_map = [np.argmax(c) for c in counts]
labels_map = tf.convert_to_tensor(labels_map)  # so we can use next.

# look up
cluster_label = tf.nn.embedding_lookup(labels_map, cluster_idx)  # find cluster_idx labels in map

# accuracy
correct_prediction = tf.equal(cluster_label, tf.cast(tf.arg_max(y, 1), dtype=tf.int32))  # bool
accuracy_value = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))

# test
test_X, test_y = mnist.test.images, mnist.test.labels
print("accuracy: {0:.4f}".format(sess.run(accuracy_value, feed_dict={X: test_X, y: test_y})))


##----------------------------------
# when k = 500, temperature of my Macbook Pro CPU core come to 100 ℃ , get accuracy 0.9345











#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 10:23:44 2018

@author: Rorschach
@mail: 188581221@qq.com
"""
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# parameters
learning_rate = 0.01
training_epochs = 5000
display_step = 50

# Data
train_X = np.array([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_y = np.array([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])
n_sample = train_X.shape[0]

# set graph
X = tf.placeholder(dtype=tf.float32, shape=[n_sample, ])
y = tf.placeholder(dtype=tf.float32, shape=[n_sample, ])
w = tf.Variable(np.random.randn(), name="weight")
b = tf.Variable(np.random.randn(), name="bias")

pred = tf.add(tf.multiply(X, w), b)

# cost
cost = tf.reduce_sum(tf.pow(tf.subtract(y, pred), 2) / (2 * n_sample))

# gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# initialize
init = tf.global_variables_initializer()

# training
sess = tf.Session()
sess.run(init)
for step in range(training_epochs):
    sess.run(optimizer, feed_dict={X: train_X, y: train_y})  # X.shape == train_X
    if (step + 1) % display_step == 0:
        c = sess.run(cost, feed_dict={X: train_X, y: train_y})
        print("Step {0}: cost = {1:.4f}, w = {2:.4f}, b = {3:.4f}"
              .format(step + 1, c, sess.run(w), sess.run(b)))

#  graphic
plt.plot(train_X, train_y, 'ro', label='Original data')  # 'ro' mean dot
plt.plot(train_X, sess.run(w) * train_X + sess.run(b), label='Fitted line')
plt.legend()  # show legend
plt.show()

















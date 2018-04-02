#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 16:28:39 2018

@author: Rorschach
@mail: 188581221@qq.com
"""
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# import
mnist = input_data.read_data_sets("/Users/zt/Desktop/Master File/practice of python/MNIST_data/",
                                  one_hot=True)

# parameters
learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_epoch = 1
logs_path = "/Users/zt/Desktop/TensorFlow/logs/"

# nn parameters
n_hidden_1 = 256
n_hidden_2 = 100
n_input = 28*28
n_output = 10

# set
with tf.name_scope('Input'):
    X = tf.placeholder(tf.float32, [None, 28*28], name='InputData')
    y = tf.placeholder(tf.float32, [None, 10], name='LabelData')

# create model
def multlayer_perceptron(X, weights, biases):
    # set layer 1
    layer_1 = tf.add(tf.multiply(X, weights['w1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    tf.summary.histogram('relu1', layer_1)
    # set layer 2
    layer_2 = tf.add(tf.multiply(layer_1, weights['w2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    tf.summary.histogram('relu2', layer_2)
    # set out layer
    out_layer = tf.add(tf.multiply(layer_2, weights['w3']), biases['b3'])
    return out_layer

# get w, b
weights = {
        'w1': tf.random_normal([n_input, n_hidden_1], name='w1'),
        'w2': tf.random_normal([n_hidden_1, n_hidden_2], name='w2'),
        'w3': tf.random_normal([n_hidden_2, n_output], name='w3')}

biases = {
        'b1': tf.random_normal([n_hidden_1], name='b1'),
        'b2': tf.random_normal([n_hidden_2], name='b2'),
        'b3': tf.random_normal([n_output], name='b3')}

# built
with tf.name_scope('Model'):
    pred = multlayer_perceptron(X, weights, biases)
    
with tf.name_scope('Loss'):
    # softmax cross entropy
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

with tf.name_scope('SGD'):
    # gradient descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    grads = tf.gradients(loss, tf.trainable_variables())
    grads = list(zip(grads, tf.trainable_variables()))  # can use next
    apply_grads = optimizer.apply_gradients(grads_and_vars=grads)
    
with tf.name_scope('Accuracy'):
    acc = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    acc = tf.reduce_mean(tf.cast(acc, dtype=tf.float32))

# initialize
init = tf.global_variables_initializer()






















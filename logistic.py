#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 10:53:57 2018

@author: Rorschach
@mail: 188581221@qq.com
"""
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time

# import data
mnist = input_data.read_data_sets("/Users/zt/Desktop/Master File/practice of python/MNIST_data/", one_hot=True)

# hyperparameter
learning_rate = 0.01
repeat = 300
batch_size = 100
display_step = 1
log_path = '/Users/zt/Desktop/logs'

# placeholder
with tf.name_scope('Placeholder'):
    X = tf.placeholder(dtype=tf.float32, shape=[None, 28*28], name='X')
    y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='y')

# all 1
E = tf.zeros([batch_size, 10]) + 1

# parameter
with tf.name_scope('Parameters'):
    with tf.name_scope('weights'):
        w = tf.Variable(tf.zeros([28*28, 10]), name='weight')
        tf.summary.histogram('weight', w)
    with tf.name_scope('biases'):
        b = tf.Variable(tf.zeros([10]), name='bias')
        tf.summary.histogram('bias', b)
        
# model
with tf.name_scope('Model'):
    pred = tf.nn.softmax(tf.add(tf.matmul(X, w), b))
    
# cost
with tf.name_scope('Cost'):  # 测试四种损失函数
    #cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=pred)) 
    #cost = - tf.reduce_mean(tf.reduce_sum(tf.add(y*tf.log(pred), (E - y)*tf.log(E - pred)), reduction_indices=1))
    #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
    cost = - tf.reduce_mean(tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
    tf.summary.scalar('cost', cost)
    
# optimizer
with tf.name_scope('Optimizer'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# accuracy
with tf.name_scope('Accuracy'):
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1)), dtype=tf.float32))
    tf.summary.scalar('accuracy', acc)

# train
init = tf.global_variables_initializer()
sess = tf.Session()
writer = tf.summary.FileWriter(log_path, sess.graph)
merged = tf.summary.merge_all()

sess.run(init)
begin = time.time()
for step in range(repeat):
    total = int(mnist.train.num_examples / batch_size)
    Cost = 0
    for i in range(total):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        _, costs, accuracy, mer = sess.run([optimizer, cost, acc, merged], feed_dict={X: batch_xs, 
                                           y: batch_ys})
        Cost += costs / total
    if step % display_step == 0:
        writer.add_summary(mer, step)
        print('Step{0} ------ cost:{1:.4f} ------ accuracy:{2:.4f}'.format(step+1, Cost, accuracy))
end = time.time()
print('Total: {} s'.format(end-begin))
        
# test
test_accuracy = sess.run(acc, feed_dict={X: mnist.test.images,
                                y: mnist.test.labels})
print('Test Accuracy: {0:.3f}'.format(test_accuracy))

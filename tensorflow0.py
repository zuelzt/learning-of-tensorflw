#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 23:43:15 2018

@author: Rorschach
@mail: 188581221@qq.com
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf

x_data = np.random.rand(100).astype("float32")
y_data = x_data * 0.1 + 0.3

#if we don't know 
#figure out
W = tf.Variable(tf.random_uniform([1], -1, 1))  #dim = 1, between -1 and 1
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

#Minimize the mean squard errors
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

#initialize the variables
init = tf.initialize_all_variables()

#launch
sess = tf.Session()
sess.run(init)

#fit the line
for step in range(201):  
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(b))
  
      
        
#The MNIST Data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#try1
x = tf.placeholder("float", [None, 784])  #占位符， 28 * 28 = 784 等待被赋值
W = tf.Variable(tf.zeros([784, 10]))  #设置变量，接收 784 输出 10
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)
#tf.matmul(x, W) = Wx

#loss by cross-entropy
y_ = tf.placeholder("float", [None, 10])
cross_entropy = - tf.reduce_sum(y_ * tf.log(y))
#tf.reduce_sum = sum

#train
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
#learning rate = 0.01

#ready
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

#train
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)  #每一步迭代加载 100 个样本
    sess.run(train_step, feed_dict={x:batch_xs, y_:batch_ys})

#test
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))  #bool
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))  #cast bool -> float
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_:mnist.test.labels}))




#try2 CNN
#initial
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)  #产生 0 均值， 0.1 方差的正态分布数
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)  #产生微小偏差
    return tf.Variable(initial)

#set
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')  #1 步长， 0 边距

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  #2 * 2

#layer 1
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#Layer 2
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#densely connected layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#output layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#train,test
cross_entropy = - tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess.run(tf.initialize_all_variables())

for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_:batch[1], keep_prob:1.0})
        #与 sess.run() 等价
        print("Step{0},training accuracy:{1:.4f}".format(i, train_accuracy))
    train_step.run(feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})
print("test accuracy:{0:.4f}"
      .format(accuracy.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0})))

#final accuracy: 0.99





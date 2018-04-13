#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 08:41:55 2018

@author: Rorschach
@mail: 188581221@qq.com
"""
import warnings
warnings.filterwarnings('ignore')
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# import data
mnist = input_data.read_data_sets('/Users/zt/Desktop/Master File/practice of python/MNIST_data', one_hot=True)

# hyperparameter
learning_rate = 0.01
train_step = 100
batch_size = 1000 
display_step = 50
log_path = '/Users/zt/Desktop/logs/'

# parameter
n_input = 28*28
n_layer_1 = 256
n_layer_2 = 128

# placeholder
with tf.name_scope('Placeholder'):
    X = tf.placeholder(tf.float32, [None, 28*28], name='X')
    X_image = tf.transpose(tf.reshape(X, [-1, 1, 28, 28]), perm=[0, 2, 3, 1])
    tf.summary.image('Input', X_image)
        
# weight, bias
with tf.name_scope('Parameter'):
    with tf.name_scope('weights'):
        weights = {'w_e1': tf.Variable(tf.random_normal([n_input, n_layer_1])),
                   'w_e2': tf.Variable(tf.random_normal([n_layer_1, n_layer_2])),
                   'w_d1': tf.Variable(tf.random_normal([n_layer_2, n_layer_1])),
                   'w_d2': tf.Variable(tf.random_normal([n_layer_1, n_input]))}
        for name in weights.keys():
            tf.summary.histogram(name, weights[name])
    with tf.name_scope('biases'):
        biases = {'b_e1': tf.Variable(tf.random_normal([n_layer_1])),
                  'b_e2': tf.Variable(tf.random_normal([n_layer_2])),
                  'b_d1': tf.Variable(tf.random_normal([n_layer_1])),
                  'b_d2': tf.Variable(tf.random_normal([n_input]))}
        for name in biases.keys():
            tf.summary.histogram(name, biases[name])
 
# model       
def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['w_e1']), biases['b_e1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['w_e2']), biases['b_e2']))
    return layer_2

def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['w_d1']), biases['b_d1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['w_d2']), biases['b_d2']))
    return layer_2

with tf.name_scope('Model'):
    op = encoder(X)
    pred = decoder(op)
    
# cost
y = X
with tf.name_scope('Cost'):
    cost = tf.reduce_mean(tf.pow(y-pred, 2))
    tf.summary.scalar('Cost', cost)
    
# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# accuracy
with tf.name_scope('Accuracy'):
    accuracy = tf.reduce_mean(tf.reduce_mean(tf.cast(tf.equal(pred > 0.5, y > 0.5), tf.float32), reduction_indices=1))
    tf.summary.scalar('Acuuracy', accuracy)
'''
If we use "tf.equal(pred, y)," accuracy will be close to zero.
So I try "tf.equal(pred > 0.5, y > 0.5)".
It works.
'''

# before
sess = tf.Session()
init = tf.global_variables_initializer()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter(log_path, sess.graph)
num = 0  # count

# train
sess.run(init)
begin = time.time()
for step in range(train_step):
    total = int(mnist.train.num_examples / batch_size)
    for i in range(total):
        batch_xs, notuse = mnist.train.next_batch(batch_size)
        _, mer, cos, acc = sess.run([optimizer, merged, cost, accuracy], feed_dict={X: batch_xs})
        num += 1
        if (num) % display_step == 0:
            writer.add_summary(mer, num)
            print('step:{0}------cost:{1:.4f}------accuracy:{2:.4f}'.format(num, cos, acc))
end = time.time()

# results
print('final accuracy:{0:.3f} ----- Total:{1} s!'.format(acc, end-begin))

# test
n = 10  # display 
true_image = np.empty((28*n, 28*n))
pred_image = np.empty((28*n, 28*n))

for i in range(n):
    batch_xss, _ = mnist.test.next_batch(n)
    pred_array = sess.run(pred, feed_dict={X: batch_xss})  
    for j in range(n):
        true_image[i*28:(i+1)*28,j*28:(j+1)*28] = batch_xss[j].reshape([28, 28])
        pred_image[i*28:(i+1)*28,j*28:(j+1)*28] = pred_array[j].reshape([28, 28])
        
# plot
print('original')
plt.figure(figsize=(n, n))
plt.imshow(true_image, cmap='gray',origin='upper')
plt.show()

print('coder')
plt.figure(figsize=(n, n))
plt.imshow(pred_image, cmap='gray', origin='upper')
plt.show()





        
    
    











































#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 08:42:15 2018

@author: Rorschach
@mail: 188581221@qq.com
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf

#Build
matrix1 = tf.constant([[3, 3]])   #op1
matrix2 = tf.constant([[1, 2], [3, 4]])   #op2
product = tf.matmul(matrix1, matrix2)   #op3

#Launch
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
result1 = sess.run(product)
sess.close()

#Or
with tf.Session() as sess:
    with tf.device("/cpu:8000"):
        result1_5 = sess.run(product)
    
#Test
tm1 = np.array([[3, 3]])
tm2 = np.array([[1, 2], [3, 4]])
result2 = np.matmul(tm1, tm2)

print(result1 == result2, result1_5 == result2)

#also can use
maybe_I_use_another_name = tf.InteractiveSession()  #避免只使用一个变量来维持会话
#Tensor.eval() and Operation.run()

x = tf.Variable([1, 2])
a = tf.constant([3, 3])

x.initializer.run()  #Operation.run()

sub = tf.sub(x, a)  # -
print(sub.eval())  #Tensor.eval()

maybe_I_use_another_name.close()

#we use Tensor to represent all data.
#We can think of a TensorFlow tensor as an n-dimensional array or list. 

#Variables & Fetches
a = tf.Variable(0, name="zt")  #we can use "print(a.name)" to get "zt:0"
b = tf.constant(12)
c = tf.constant(1)  
new1 = tf.add(a, c)  # a tensor
new2 = tf.mul(b, 2) 
update1 = tf.assign(a, new1)  #state <- new

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    #print(sess.run(a))
    for _ in range(5):
        #sess.run(new)
        sess.run(update1)
        print(sess.run([a, new2]))
#sess.run() 会自动运行 Tensor add ？

#Feed
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.mul(input1, input2)

with tf.Session() as sess:
    print(sess.run(output, feed_dict={input1:[2], input2:4}))  #only use once

















































 






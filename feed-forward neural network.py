#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 09:39:01 2018

@author: Rorschach
@mail: 188581221@qq.com
"""
# 用于 Python2 的兼容，此处可忽略 
# =============================================================================
# from __future__ import absolute_import  
# from __future__ import division
# from __future__ import print_function
# =============================================================================

#import
import warnings
warnings.filterwarnings('ignore')

import argparse  # 不是很清楚什么用
import os
import sys
import time

from six.moves import xrange  # 即 range
import tensorflow as tf 

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist

# input data
FLAGS = None  # 设置一个空参数

## 定义一个产生占位符的函数
def placeholder_inputs(batches_size):
    image_placeholder = tf.placeholder(tf.float32, shape=(batches_size, mnist.IMAGE_PIXELS))
    label_placeholder = tf.placeholder(tf.float32, shape=batches_size)
    return image_placeholder, label_placeholder

## 定义输入字典函数
def fill_feed_dict(data_set, images_pl, labels_pl):  # dataset: from input_data.read_data_sets()
  images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size,
                                                 FLAGS.fake_data)
  feed_dict = {
      images_pl: images_feed,
      labels_pl: labels_feed,
  }
  return feed_dict

## 定义计算准确率的函数
def do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_set):
  true_count = 0  # 计数
  steps_per_epoch = data_set.num_examples // FLAGS.batch_size  # 分几组
  num_examples = steps_per_epoch * FLAGS.batch_size  # 等量分组
  for step in range(steps_per_epoch):
    feed_dict = fill_feed_dict(data_set, images_placeholder, labels_placeholder)
    true_count += sess.run(eval_correct, feed_dict=feed_dict)
  precision = float(true_count) / num_examples
  print('Num examples: {0}  Num correct: {1}  Precision: {2:.4f}'
        .format(num_examples, true_count, precision))

# 定义运行函数
def run_training():
  data_sets = input_data.read_data_sets(FLAGS.input_data_dir, FLAGS.fake_data)

  with tf.Graph().as_default():  # 在默认图上运行
    images_placeholder, labels_placeholder = placeholder_inputs(
        FLAGS.batch_size)

    logits = mnist.inference(images_placeholder,
                             FLAGS.hidden1,
                             FLAGS.hidden2)  # 构建两个隐藏层的 softmax 输出

    # Add to the Graph the Ops for loss calculation.
    loss = mnist.loss(logits, labels_placeholder)

    # Add to the Graph the Ops that calculate and apply gradients.
    train_op = mnist.training(loss, FLAGS.learning_rate)

    # Add the Op to compare the logits to the labels during evaluation.
    eval_correct = mnist.evaluation(logits, labels_placeholder)

    # Build the summary Tensor based on the TF collection of Summaries.
    summary = tf.summary.merge_all()

    # Add the variable initializer Op.
    init = tf.global_variables_initializer()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

    # And then after everything is built:

    # Run the Op to initialize the variables.
    sess.run(init)

    # Start the training loop.
    for step in xrange(FLAGS.max_steps):
      start_time = time.time()

      # Fill a feed dictionary with the actual set of images and labels
      # for this particular training step.
      feed_dict = fill_feed_dict(data_sets.train,
                                 images_placeholder,
                                 labels_placeholder)

      # Run one step of the model.  The return values are the activations
      # from the `train_op` (which is discarded) and the `loss` Op.  To
      # inspect the values of your Ops or variables, you may include them
      # in the list passed to sess.run() and the value tensors will be
      # returned in the tuple from the call.
      _, loss_value = sess.run([train_op, loss],
                               feed_dict=feed_dict)

      duration = time.time() - start_time

      # Write the summaries and print an overview fairly often.
      if step % 100 == 0:
        # Print status to stdout.
        print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
        # Update the events file.
        summary_str = sess.run(summary, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()

      # Save a checkpoint and evaluate the model periodically.
      if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
        saver.save(sess, checkpoint_file, global_step=step)
        # Evaluate against the training set.
        print('Training Data Eval:')
        do_eval(sess,
                eval_correct,
                images_placeholder,
                labels_placeholder,
                data_sets.train)
        # Evaluate against the validation set.
        print('Validation Data Eval:')
        do_eval(sess,
                eval_correct,
                images_placeholder,
                labels_placeholder,
                data_sets.validation)
        # Evaluate against the test set.
        print('Test Data Eval:')
        do_eval(sess,
                eval_correct,
                images_placeholder,
                labels_placeholder,
                data_sets.test)












































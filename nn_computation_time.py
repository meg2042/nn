from tensorflow.examples.tutorials.mnist import input_data
from scipy import ndimage
from tkinter import *
import tensorflow as tf
import numpy as np
import math
import time
import inflect
import threading
import cv2

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

neurons_layer_h1 = 800
neurons_layer_h2 = 400
neurons_layer_h3 = 200

#examples tf.nn.relu tf.nn.sigmoid tf.nn.tanh
l1_activation_function = tf.nn.relu
l2_activation_function = tf.nn.relu
l3_activation_function = tf.nn.relu

W_fc1 = weight_variable([784, neurons_layer_h1])
b_fc1 = bias_variable([neurons_layer_h1])
h_fc1 = l1_activation_function(tf.matmul(x, W_fc1) + b_fc1)

W_fc2 = weight_variable([neurons_layer_h1, neurons_layer_h2])
b_fc2 = bias_variable([neurons_layer_h2])
h_fc2 = l2_activation_function(tf.matmul(h_fc1, W_fc2) + b_fc2)

W_fc3 = weight_variable([neurons_layer_h2, neurons_layer_h3])
b_fc3 = bias_variable([neurons_layer_h3])
h_fc3 = l3_activation_function(tf.matmul(h_fc2, W_fc3) + b_fc3)

W_o = weight_variable([neurons_layer_h3, 10])
b_o = bias_variable([10])
y = tf.matmul(h_fc3, W_o) + b_o

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
train_step = tf.train.RMSPropOptimizer(0.001).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
elapsed_list = []
iterations = 300

for n in range(0, 10):
  tf.global_variables_initializer().run()
  with tf.Session() as sess:
    start = time.perf_counter()
    sess.run(tf.global_variables_initializer())
    for i in range(iterations):
      batch = mnist.train.next_batch(128)
      if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1]})
        print('Step %d, training accuracy %g' % (i, train_accuracy))
      train_step.run(feed_dict={x: batch[0], y_: batch[1]})
    elapsed_list.append(time.perf_counter() - start)
    #print('Elapsed %.3f seconds.' % elapsed)
    print('Test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

print('Elapsed times for', len(elapsed_list), 'runs with', iterations, 'iterations')
print("Average:", sum(elapsed_list)/len(elapsed_list))
print("Standard deviation:", np.std(np.array(elapsed_list)))

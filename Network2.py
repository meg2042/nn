from tensorflow.examples.tutorials.mnist import input_data
from scipy import ndimage
from tkinter import *
import tensorflow as tf
import numpy as np
import math
import inflect
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

W_fc1 = weight_variable([784, neurons_layer_h1])
b_fc1 = bias_variable([neurons_layer_h1])
h_fc1 = tf.sigmoid(tf.matmul(x, W_fc1) + b_fc1)

W_fc2 = weight_variable([neurons_layer_h1, neurons_layer_h2])
b_fc2 = bias_variable([neurons_layer_h2])
h_fc2 = tf.sigmoid(tf.matmul(h_fc1, W_fc2) + b_fc2)

W_fc3 = weight_variable([neurons_layer_h2, neurons_layer_h3])
b_fc3 = bias_variable([neurons_layer_h3])
h_fc3 = tf.sigmoid(tf.matmul(h_fc2, W_fc3) + b_fc3)

W_o = weight_variable([neurons_layer_h3, 10])
b_o = bias_variable([10])
y = tf.matmul(h_fc3, W_o) + b_o

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
#train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(0.003).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

def getBestShift(img):
  cy, cx = ndimage.measurements.center_of_mass(img)
  rows, cols = img.shape
  shiftx = np.round(cols / 2.0 - cx).astype(int)
  shifty = np.round(rows / 2.0 - cy).astype(int)
  return shiftx, shifty


def shift(img, sx, sy):
  rows, cols = img.shape
  M = np.float32([[1, 0, sx], [0, 1, sy]])
  shifted = cv2.warpAffine(img, M, (cols, rows))
  return shifted

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(120000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
      train_accuracy = accuracy.eval(feed_dict={
          x: batch[0], y_: batch[1]})
      print('Step %d, training accuracy %g' % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

  print('Test accuracy %g' % accuracy.eval(feed_dict={
      x: mnist.test.images, y_: mnist.test.labels}))

  images = np.zeros((1,784))
  correct_vals = np.zeros((1,10))

  while True:
    def preddigit():
      gray = cv2.imread("img1.png", 0)
      gray = cv2.resize(255-gray, (28, 28))
      (thresh, gray) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

      while np.sum(gray[0]) == 0:
          gray = gray[1:]

      while np.sum(gray[:,0]) == 0:
          gray = np.delete(gray,0,1)

      while np.sum(gray[-1]) == 0:
          gray = gray[:-1]

      while np.sum(gray[:,-1]) == 0:
          gray = np.delete(gray,-1,1)

      rows,cols = gray.shape

      if rows > cols:
          factor = 20.0/rows
          rows = 20
          cols = int(round(cols*factor))
          gray = cv2.resize(gray, (cols,rows))
      else:
          factor = 20.0/cols
          cols = 20
          rows = int(round(rows*factor))
          gray = cv2.resize(gray, (cols, rows))

      colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
      rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
      gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant')

      shiftx,shifty = getBestShift(gray)
      shifted = shift(gray,shiftx,shifty)
      gray = shifted

      p = inflect.engine()

      flatten = gray.flatten() / 255.0
      images[0] = flatten
      prediction = tf.argmax(y, 1)
      predictionprint = sess.run(prediction, feed_dict={x: images, y_: correct_vals})
      labelText.set(p.number_to_words(predictionprint[0]).title())

    root = Tk()
    root.title("Digit predictor - AdamOptimizer")
    root.bind('<Key>', lambda e: preddigit())
    frame = Frame(root, width=1000, height=1200)
    frame.pack()

    labelText = StringVar()
    lab = Label(frame, textvariable=labelText)
    lab.config(font=("Times", 180))
    lab.pack(side="top", fill='both', padx=4, pady=4)

    button_1 = Button(root, font="Times", bg='#98fb98', fg='#000000', text='Predict', width=10, command=preddigit)
    button_1.pack(side="bottom", fill='x', padx=4, pady=4)

    root.mainloop()
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import argparse
import sys
import tensorflow as tf

alpha = 0.1
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W, stride):
  return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')

def max_pool(x,pool_size, stride):
  return tf.nn.max_pool(x, ksize=[1, pool_size, pool_size, 1],strides=[1, stride, stride, 1], padding='SAME')

img = cv2.imread('test.jpg',-1)

img = cv2.resize(img, (448, 448))

inputs = np.zeros((1,448,448,3),dtype='float32')

inputs[0,:,:,:] = (img/255.0)*2.0 - 1.0


x = tf.placeholder(tf.float32,[None,448,448,3])

W_conv1 = weight_variable([7,7,3,64])
b_conv1 = bias_variable([64])
h_conv1 = conv2d(x, W_conv1,2) + b_conv1
h_conv1 = tf.maximum(alpha*h_conv1,h_conv1)
h_pool1 = max_pool(h_conv1,2,2)

W_conv2 = weight_variable([3,3,64,192])
b_conv2 = bias_variable([192])
h_conv2 = conv2d(h_pool1, W_conv2,1) + b_conv2
h_conv2 = tf.maximum(alpha*h_conv2,h_conv2)
h_pool2 = max_pool(h_conv2,2,2)


W_conv3 = weight_variable([1,1,192,128])
b_conv3 = bias_variable([128])
h_conv3 = conv2d(h_pool2, W_conv3,1) + b_conv3
h_conv3 = tf.maximum(alpha*h_conv3,h_conv3)

W_conv4 = weight_variable([3,3,128,256])
b_conv4 = bias_variable([256])
h_conv4 = conv2d(h_conv3, W_conv4,1) + b_conv4
h_conv4 = tf.maximum(alpha*h_conv4,h_conv4)

W_conv5 = weight_variable([1,1,256,256])
b_conv5 = bias_variable([256])
h_conv5 = conv2d(h_conv4, W_conv5,1) + b_conv5
h_conv5 = tf.maximum(alpha*h_conv5,h_conv5)

W_conv6 = weight_variable([3,3,256,512])
b_conv6 = bias_variable([512])
h_conv6 = conv2d(h_conv5, W_conv6,1) + b_conv6
h_conv6 = tf.maximum(alpha*h_conv6,h_conv6)
h_pool6 = max_pool(h_conv6,2,2)

W_conv7 = weight_variable([1,1,512,256])
b_conv7 = bias_variable([256])
h_conv7 = conv2d(h_pool6, W_conv7,1) + b_conv7
h_conv7 = tf.maximum(alpha*h_conv7,h_conv7)

W_conv8 = weight_variable([3,3,256,512])
b_conv8 = bias_variable([512])
h_conv8 = conv2d(h_conv7, W_conv8,1) + b_conv8
h_conv8 = tf.maximum(alpha*h_conv8,h_conv8)

W_conv9 = weight_variable([1,1,512,256])
b_conv9 = bias_variable([256])
h_conv9 = conv2d(h_conv8, W_conv9,1) + b_conv9
h_conv9 = tf.maximum(alpha*h_conv9,h_conv9)

W_conv10 = weight_variable([3,3,256,512])
b_conv10 = bias_variable([512])
h_conv10 = conv2d(h_conv9, W_conv10,1) + b_conv10
h_conv10 = tf.maximum(alpha*h_conv10,h_conv10)

W_conv11 = weight_variable([1,1,512,256])
b_conv11 = bias_variable([256])
h_conv11 = conv2d(h_conv10, W_conv11,1) + b_conv11
h_conv11 = tf.maximum(alpha*h_conv11,h_conv11)

W_conv12 = weight_variable([3,3,256,512])
b_conv12 = bias_variable([512])
h_conv12 = conv2d(h_conv11, W_conv12,1) + b_conv12
h_conv12 = tf.maximum(alpha*h_conv12,h_conv12)

W_conv13 = weight_variable([1,1,512,256])
b_conv13 = bias_variable([256])
h_conv13 = conv2d(h_conv12, W_conv13,1) + b_conv13
h_conv13 = tf.maximum(alpha*h_conv13,h_conv13)

W_conv14 = weight_variable([3,3,256,512])
b_conv14 = bias_variable([512])
h_conv14 = conv2d(h_conv13, W_conv14,1) + b_conv14
h_conv14 = tf.maximum(alpha*h_conv14,h_conv14)

W_conv15 = weight_variable([1,1,512,512])
b_conv15 = bias_variable([512])
h_conv15 = conv2d(h_conv14, W_conv15,1) + b_conv15
h_conv15 = tf.maximum(alpha*h_conv15,h_conv15)

W_conv16 = weight_variable([3,3,512,1024])
b_conv16 = bias_variable([1024])
h_conv16 = conv2d(h_conv15, W_conv16,1) + b_conv16
h_conv16 = tf.maximum(alpha*h_conv16,h_conv16)
h_pool16 = max_pool(h_conv16,2,2)

W_conv17 = weight_variable([1,1,1024,512])
b_conv17 = bias_variable([512])
h_conv17 = conv2d(h_pool16, W_conv17,1) + b_conv17
h_conv17 = tf.maximum(alpha*h_conv17,h_conv17)

W_conv18 = weight_variable([3,3,512,1024])
b_conv18 = bias_variable([1024])
h_conv18 = conv2d(h_conv17, W_conv18,1) + b_conv18
h_conv18 = tf.maximum(alpha*h_conv18,h_conv18)

W_conv19 = weight_variable([1,1,1024,512])
b_conv19 = bias_variable([512])
h_conv19 = conv2d(h_conv18, W_conv19,1) + b_conv19
h_conv19 = tf.maximum(alpha*h_conv19,h_conv19)

W_conv20 = weight_variable([3,3,512,1024])
b_conv20 = bias_variable([1024])
h_conv20 = conv2d(h_conv19, W_conv20,1) + b_conv20
h_conv20 = tf.maximum(alpha*h_conv20,h_conv20)

W_conv21 = weight_variable([3,3,1024,1024])
b_conv21 = bias_variable([1024])
h_conv21 = conv2d(h_conv20, W_conv21,1) + b_conv21
h_conv21 = tf.maximum(alpha*h_conv21,h_conv21)

W_conv22 = weight_variable([3,3,1024,1024])
b_conv22 = bias_variable([1024])
h_conv22 = conv2d(h_conv21, W_conv22,2) + b_conv22
h_conv22 = tf.maximum(alpha*h_conv22,h_conv22)

W_conv23 = weight_variable([3,3,1024,1024])
b_conv23 = bias_variable([1024])
h_conv23 = conv2d(h_conv22, W_conv23,1) + b_conv23
h_conv23 = tf.maximum(alpha*h_conv23,h_conv23)

W_conv24 = weight_variable([3,3,1024,1024])
b_conv24 = bias_variable([1024])
h_conv24 = conv2d(h_conv23, W_conv24,1) + b_conv24
h_conv24 = tf.maximum(alpha*h_conv24,h_conv24)

W_fc25 = weight_variable([7 * 7 * 1024 , 4096])
b_fc25 = bias_variable([4096])
h_pool25_flat = tf.reshape(h_conv24, [-1, 7*7*1024])
h_fc25 = tf.add(tf.matmul(h_pool25_flat, W_fc25), b_fc25)
h_fc25 = tf.maximum(alpha*h_fc25,h_fc25)

h_fc25 = tf.nn.dropout(h_fc25, 0.5)

W_fc26 = weight_variable([4096, 7*7*30])
b_fc26 = bias_variable([7*7*30])
h_fc26 = tf.add(tf.matmul(h_fc25, W_fc26), b_fc26)
h_fc26 = tf.reshape(h_fc26,[7,7,30])


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

sess.run([h_fc26],feed_dict={x:inputs})

















print inputs.shape

# plt.imshow(img, cmap=cm.Greys_r)
# plt.show()
# cv2.waitKey(0)
# cv2.destroyAllWindows()



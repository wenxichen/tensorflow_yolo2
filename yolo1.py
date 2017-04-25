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

def conv_layer(x,filter_size,input_chl,output_chl,stride):
  W_conv = weight_variable([filter_size, filter_size, input_chl, output_chl])
  b_conv = bias_variable([output_chl])
  h_conv = conv2d(x, W_conv, stride) + b_conv
  return tf.maximum(alpha * h_conv, h_conv)

def fc_layer(x,input_dim,output_dim,flat=False,linear=False):
  W_fc = weight_variable([input_dim, output_dim])
  b_fc = bias_variable([output_dim])
  if flat: x = tf.reshape(x, [-1, input_dim])
  h_fc = tf.add(tf.matmul(x, W_fc), b_fc)
  if linear: return h_fc
  return tf.maximum(alpha * h_fc, h_fc)


img = cv2.imread('test.jpg',-1)

img = cv2.resize(img, (448, 448))

inputs = np.zeros((1,448,448,3),dtype='float32')

inputs[0,:,:,:] = (img/255.0)*2.0 - 1.0


x = tf.placeholder(tf.float32,[None,448,448,3])

h_conv1 = conv_layer(x,7,3,64,2)
h_pool1 = max_pool(h_conv1,2,2)
print h_pool1.get_shape()
h_conv2 = conv_layer(h_pool1,3,64,192,1)
h_pool2 = max_pool(h_conv2,2,2)
h_conv3 = conv_layer(h_pool2,1,192,128,1)
h_conv4 = conv_layer(h_conv3,3,128,256,1)
h_conv5 = conv_layer(h_conv4,1,256,256,1)
h_conv6 = conv_layer(h_conv5,3,256,512,1)
h_pool6 = max_pool(h_conv6,2,2)
h_conv7 = conv_layer(h_pool6,1,512,256,1)
h_conv8 = conv_layer(h_conv7,3,256,512,1)
h_conv9 = conv_layer(h_conv8,1,512,256,1)
h_conv10 = conv_layer(h_conv9,3,256,512,1)
h_conv11 = conv_layer(h_conv10,1,512,256,1)
h_conv12 = conv_layer(h_conv11,3,256,512,1)
h_conv13 = conv_layer(h_conv12,1,512,256,1)
h_conv14 = conv_layer(h_conv13,3,256,512,1)
h_conv15 = conv_layer(h_conv14,1,512,512,1)
h_conv16 = conv_layer(h_conv15,3,512,1024,1)
h_pool16 = max_pool(h_conv16,2,2)
h_conv17 = conv_layer(h_pool16,1,1024,512,1)
h_conv18 = conv_layer(h_conv17,3,512,1024,1)
h_conv19 = conv_layer(h_conv18,1,1024,512,1)
h_conv20 = conv_layer(h_conv19,3,512,1024,1)
h_conv21 = conv_layer(h_conv20,3,1024,1024,1)
h_conv22 = conv_layer(h_conv21,3,1024,1024,2)
h_conv23 = conv_layer(h_conv22,3,1024,1024,1)
h_conv24 = conv_layer(h_conv23,3,1024,1024,1)
h_fc25 = fc_layer(h_conv24,7 * 7 * 1024,4096,flat=True)
h_fc25 = tf.nn.dropout(h_fc25, 0.5)
h_fc26 = fc_layer(h_fc25,4096,7*7*30,flat=False,linear=True)
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



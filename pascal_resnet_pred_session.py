# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import glob
import cv2
import os, sys
sys.path.append('/home/wenxi/Projects/tensorflow_yolo2/src')
from slim_dir.nets import resnet_v1
import config as cfg
from img_dataset.pascal_voc import pascal_voc
from utils.timer import Timer
from yolo2_nets.net_utils import get_resnet_tf_variables
from yolo2_nets.tf_resnet import resnet_v1_50

slim = tf.contrib.slim
NUM_CLASS = 20
IMAGE_SIZE = cfg.IMAGE_SIZE
S = cfg.S
B = cfg.B

# create database instance
imdb = pascal_voc('trainval')
CKPTS_DIR = cfg.get_ckpts_dir('resnet50', imdb.name)

input_data = tf.placeholder(tf.float32, [None, 224, 224, 3])
# labels = tf.placeholder(tf.float32, [None, S, S, 5 + NUM_CLASS])

# read in the test image
image = cv2.imread(
    '/home/wenxi/Projects/tensorflow_yolo2/experiments/fig1.jpg')
image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
image = image.astype(np.float32)
image = (image / 255.0) * 2.0 - 1.0
image = image.reshape((1, 224, 224, 3))
# get the right arg_scope in order to load weights
with slim.arg_scope(resnet_v1.resnet_arg_scope()):
    # net is shape [batch_size, S, S, 2048] if input size is 244 x 244
    net, end_points = resnet_v1_50(input_data, is_training=False)

net = slim.flatten(net)

# TODO: bug need fix
fcnet = slim.fully_connected(net, 4096, scope='yolo_fc1')
fcnet = tf.nn.dropout(fcnet, 1)

# in this case 7x7x30
fcnet = slim.fully_connected(
    net, S * S * (5 * B + NUM_CLASS), scope='yolo_fc2')

grid_net = tf.reshape(fcnet, [-1, S, S, (5 * B + NUM_CLASS)])
tfconfig = tf.ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth = True
sess = tf.Session(config=tfconfig)

# Load checkpoint
_ = get_resnet_tf_variables(sess, imdb, 'resnet50', save_epoch=False)

predicts = sess.run(grid_net, {input_data: image})
predicts.shape
predicts[0,:,:,20:22]
predicts[0,:,:,20:21]
predicts[0,:,:,20:21].shape
predicts[0,:,:,20:21][3,:,:]
predicts[0,:,:,20:21][3,3,0]
predicts[0,3,3,22:26]
offset = np.array(range(S) * S * B)
    offset = np.reshape(offset, (B, S, S))
    offset = np.transpose(offset, (1, 2, 0))  # [Y,X,B]
offset = np.array(range(S) * S * B)
offset = np.reshape(offset, (B, S, S))
offset = np.transpose(offset, (1, 2, 0))  # [Y,X,B]
offset
predicts[0,3,3,22:26]
pred_x = (predicts[0,3,3,22] + 3)/7.0
pred_x
pred_y = (predicts[0,3,3,23] + 3)/7.0
pred_w = (predicts[0,3,3,24])
pred_h = (predicts[0,3,3,25])
x = pred_x * 353
y = pred_y * 500
x
y
w = pred_w * 353
h = pred_h * 500
[x, y, w, h]
predicts[0,3,3,:20]
predicts[0,3,3,12]
predicts[0,3,3,14]
imdb
imdb.classes[14]

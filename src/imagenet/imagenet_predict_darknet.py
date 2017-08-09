"""Predict image class using ILSVRC2017 trained model"""

import cv2
import os
import math
import tensorflow as tf
import numpy as np

import os, sys
FILE_DIR = os.path.dirname(__file__)
sys.path.append(FILE_DIR+'/../')

import config as cfg
from img_dataset.ilsvrc2017_cls_multithread import ilsvrc_cls
from yolo2_nets.darknet import darknet19
from yolo2_nets.net_utils import get_ordered_ckpts
from utils.timer import Timer

imdb = ilsvrc_cls('val', batch_size=1)

input_data = tf.placeholder(tf.float32, [None, 224, 224, 3])
is_training = tf.placeholder(tf.bool)

logits = darknet19(input_data, is_training=is_training)
# pred = tf.argmax(logits, axis=1)
values, idxs = tf.nn.top_k(logits, k=5)

######################
# Initialize Session #
######################
tfconfig = tf.ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth = True
sess = tf.Session(config=tfconfig)

# # initialize variables, assume all vars are new now
# init_op = tf.global_variables_initializer()
# sess.run(init_op)
# load previous models
ckpts = get_ordered_ckpts(sess, imdb, 'darknet19')
print('Restorining model snapshots from {:s}'.format(ckpts[-1]))
old_saver = tf.train.Saver()
old_saver.restore(sess, str(ckpts[-1]))
print('Restored.')

T = Timer()
# while True:
# f = raw_input("image to predict: ")
f = "/home/wenxi/Projects/tensorflow_yolo2/data/ILSVRC/Data/CLS-LOC/train/n01537544/n01537544_16.JPEG"
# if f.lower() == "quit":
    # break
image = cv2.imread(f)
image = cv2.resize(image, (224, 224))


cv2.imshow("img", image)
k = cv2.waitKey(0)

image = image.reshape((1, 224, 224, 3))

T.tic()
probs, preds, logitss = sess.run([values, idxs, logits], {input_data: image, is_training: 0})
_time = T.toc(average=False)
# print logitss
print "predictions:", [imdb.classes[i] for i in preds[0]]
print "probabilities:", probs[0]
print "takes time:", _time
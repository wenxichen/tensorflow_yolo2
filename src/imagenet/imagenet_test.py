"""Predict image class using ILSVRC2017 trained model"""

import cv2
import os
import math
import tensorflow as tf
import numpy as np
from tqdm import tqdm

import os
import sys
FILE_DIR = os.path.dirname(__file__)
sys.path.append(FILE_DIR + '/../')

import config as cfg
from img_dataset.ilsvrc2017_cls_multithread import ilsvrc_cls
from yolo2_nets.darknet import darknet19
from yolo2_nets.net_utils import get_ordered_ckpts
from utils.timer import Timer

imdb = ilsvrc_cls('val', batch_size=50)
assert 0 == (imdb.image_num % imdb.batch_size)
print "######image number:", imdb.image_num
print "######batch number:", imdb.total_batch

input_data = tf.placeholder(tf.float32, [None, 224, 224, 3])
label_data = tf.placeholder(tf.int32, None)
is_training = tf.placeholder(tf.bool)

logits = darknet19(input_data, is_training=is_training)
# values, idxs = tf.nn.top_k(logits, k=5)

correct_pred = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), label_data)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

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
accumulated_acc = 0.0
accumulated_time = 0.0
for i in range(imdb.total_batch):
    images, labels = imdb.get()
    T.tic()
    logits_value, accuracy_value = sess.run([logits, accuracy],
                                            {input_data: images, label_data: labels, is_training: 0})
    _time = T.toc(average=False)
    print("batch {:d}/{:d}, acc: {:3f}, time: {:2f}sec"
          .format(i + 1, imdb.total_batch, accuracy_value, _time))
    accumulated_acc += accuracy_value
    accumulated_time += _time

print "###########validation accuracy:", (accumulated_acc / float(imdb.total_batch))
print "###########average time per batch:", (accumulated_time / float(imdb.total_batch))

"""Use trained darknet19 to detect"""

import tensorflow as tf
import numpy as np
import cv2

import os
import sys
FILE_DIR = os.path.dirname(__file__)
sys.path.append(FILE_DIR + '/../')

import config as cfg
from img_dataset.pascal_voc import pascal_voc
from utils.timer import Timer
from yolo2_nets.net_utils import restore_darknet19_variables, show_yolo_detection
from yolo2_nets.darknet import darknet19_core, darknet19_detection

slim = tf.contrib.slim

# TODO: make the image path to be user input
image_path = '/home/wenxi/Projects/tensorflow_yolo2/tests/testImg2.jpg'

IMAGE_SIZE = cfg.IMAGE_SIZE
S = cfg.S
B = cfg.B
# create database instance
imdb = pascal_voc('trainval')
NUM_CLASS = imdb.num_class
CKPTS_DIR = cfg.get_ckpts_dir('darknet19', imdb.name)

input_data = tf.placeholder(tf.float32, [None, 224, 224, 3])

# read in the test image
image = cv2.imread(image_path)
image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
image = image.astype(np.float32)
image = (image / 255.0) * 2.0 - 1.0
image = image.reshape((1, 224, 224, 3))


core_net = darknet19_core(input_data, is_training=False)
final_conv_layer = darknet19_detection(core_net, 30)
grid_net = tf.reshape(final_conv_layer, [-1, S, S, (5 * B + NUM_CLASS)])

######################
# Initialize Session #
######################
tfconfig = tf.ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth = True
sess = tf.Session(config=tfconfig)

# Load from weight file or checkpoint
# TODO: may need to put this in net_utils.py
if os.path.isfile(cfg.darknet_pascal_weight_path + ".meta"):
    print 'Restorining model from weight file {:s}'.format(cfg.darknet_pascal_weight_path)
    saver = tf.train.Saver()
    saver.restore(sess, cfg.darknet_pascal_weight_path)
    print 'Restored.'
else:
    _ = restore_darknet19_variables(sess, imdb, 'darknet19', save_epoch=False)

predicts = sess.run(grid_net, {input_data: image})
show_yolo_detection(image_path, predicts, imdb)

"""Train ILSVRC2017 Data using homemade scripts."""

import cv2
import os
import math
import tensorflow as tf
import numpy as np

import os
import sys
FILE_DIR = os.path.dirname(__file__)
sys.path.append(FILE_DIR + '/../')

import config as cfg
from img_dataset.pascal_voc import pascal_voc
from utils.timer import Timer
from yolo2_nets.net_utils import get_resnet_tf_variables, get_loss_new
from yolo2_nets.darknet import darknet19_core, darknet19_detection

slim = tf.contrib.slim

# set hyper parameters
ADD_ITER = 80000
NUM_CLASS = 20
IMAGE_SIZE = cfg.IMAGE_SIZE
S = cfg.S
SIZE_PER_CELL = 1.0 / S
B = cfg.B
BATCH_SIZE = 24

OFFSET = np.array(range(S) * S * B)
OFFSET = np.reshape(OFFSET, (B, S, S))
OFFSET = np.transpose(OFFSET, (1, 2, 0))  # [Y,X,B]

# create database instance
imdb = pascal_voc('trainval', batch_size=BATCH_SIZE, rebuild=cfg.REBUILD)
CKPTS_DIR = cfg.get_ckpts_dir('darknet19', imdb.name)

input_data = tf.placeholder(tf.float32, [None, 224, 224, 3])
label_data = tf.placeholder(tf.float32, [None, S, S, 5 + NUM_CLASS])
is_training = tf.placeholder(tf.bool)


core_net = darknet19_core(input_data, is_training=is_training)
final_conv_layer = darknet19_detection(core_net, 30)

grid_net = tf.reshape(final_conv_layer, [-1, S, S, (5 * B + NUM_CLASS)])

loss, ious, object_mask = get_loss_new(grid_net, label_data, num_class=NUM_CLASS,
                                   batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, 
                                   S=S, B=B, OFFSET=OFFSET)
tf.summary.scalar('total_loss', loss)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = tf.train.AdamOptimizer().minimize(loss)
    # train_op = tf.train.MomentumOptimizer(0.00005, 0.98).minimize(loss)

######################
# Initialize Session #
######################
tfconfig = tf.ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth = True
sess = tf.Session(config=tfconfig)

#####################################################################################
# loading from imagenet trained model
LOAD_CKPTS_DIR = cfg.get_ckpts_dir('darknet19', 'ilsvrc_2017_cls')
# graph_vars = [n.name for n in tf.get_default_graph().as_graph_def().node]
ckpt_vars = [t[0] for t in tf.contrib.framework.list_variables(LOAD_CKPTS_DIR)]
vars_to_restore = []
vars_to_init = []
for v in tf.global_variables():
    if v.name[:-2] in ckpt_vars:
        vars_to_restore.append(v)
    else:
        vars_to_init.append(v)
print 'vars_to_restore:', len(vars_to_restore)
print 'vars_to_init:', len(vars_to_init)

init_op = tf.variables_initializer(vars_to_init)
saver = tf.train.Saver(vars_to_restore)

print('Initializing new variables to train from imagenet trained model')
sess.run(init_op)
saver.restore(sess, os.path.join(LOAD_CKPTS_DIR, 'train_epoch_98.ckpt'))
####################################################################################

cur_saver = tf.train.Saver()

# generate summary on tensorboard
merged = tf.summary.merge_all()
tb_dir, _ = cfg.get_output_tb_dir('darknet19', imdb.name, val=False)
train_writer = tf.summary.FileWriter(tb_dir, sess.graph)

last_iter_num = 0
TOTAL_ITER = ADD_ITER + last_iter_num
T = Timer()
T.tic()
for i in range(last_iter_num + 1, TOTAL_ITER + 1):

    image, gt_labels = imdb.get()

    summary, loss_value, _, ious_value, object_mask_value = \
        sess.run([merged, loss, train_op, ious, object_mask],
                 {input_data: image, label_data: gt_labels, is_training: 1})
    # if i>10:
    train_writer.add_summary(summary, i)
    if i % 10 == 0:
        _time = T.toc(average=False)
        print('iter {:d}/{:d}, total loss: {:.3}, take {:.2}s'.
              format(i, TOTAL_ITER, loss_value, _time))
        T.tic()

    if i % 40000 == 0:
        save_path = cur_saver.save(sess, os.path.join(
            CKPTS_DIR, cfg.TRAIN_SNAPSHOT_PREFIX + '_iter_' + str(i) + '.ckpt'))
        print("Model saved in file: %s" % save_path)

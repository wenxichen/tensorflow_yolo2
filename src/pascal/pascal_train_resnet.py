"""Use pretained resnet50 on tensorflow to imitate YOLOv1"""

import tensorflow as tf
import numpy as np

import os
import sys
FILE_DIR = os.path.dirname(__file__)
sys.path.append(FILE_DIR + '/../')

from slim_dir.nets import resnet_v1
import config as cfg
from img_dataset.pascal_voc import pascal_voc
from utils.timer import Timer
from yolo2_nets.net_utils import restore_resnet_tf_variables, get_loss
from yolo2_nets.tf_resnet import resnet_v1_50

slim = tf.contrib.slim

# set hyper parameters
ADD_ITER = 200000
IMAGE_SIZE = cfg.IMAGE_SIZE
S = cfg.S
SIZE_PER_CELL = 1.0 / S
B = cfg.B
BATCH_SIZE = 4
# create database instance
imdb = pascal_voc('trainval', batch_size=BATCH_SIZE, rebuild=cfg.REBUILD)
NUM_CLASS = imdb.num_class
CKPTS_DIR = cfg.get_ckpts_dir('resnet50', imdb.name)

input_data = tf.placeholder(tf.float32, [None, 224, 224, 3])
label_data = tf.placeholder(tf.float32, [None, S, S, 5 + NUM_CLASS])


# get the right arg_scope in order to load weights
with slim.arg_scope(resnet_v1.resnet_arg_scope()):
    # net is shape [batch_size, S, S, 2048] if input size is 244 x 244
    net, end_points = resnet_v1_50(input_data)

net = slim.flatten(net)

fcnet1 = slim.fully_connected(net, 4096, scope='yolo_fc1')
fcnet1 = tf.nn.dropout(fcnet1, 0.5)

# in this case 7x7x30
fcnet2 = slim.fully_connected(
    fcnet1, S * S * (5 * B + NUM_CLASS), scope='yolo_fc2')

grid_net = tf.reshape(fcnet2, [-1, S, S, (5 * B + NUM_CLASS)])

loss, ious, object_mask = get_loss(grid_net, label_data, num_class=NUM_CLASS,
                                   batch_size=BATCH_SIZE, image_size=IMAGE_SIZE,
                                   S=S, B=B, OFFSET=cfg.YOLO_GRID_OFFSET)
tf.summary.scalar('total_loss', loss)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = tf.train.AdamOptimizer(0.0005).minimize(loss)
    # train_op = tf.train.MomentumOptimizer(0.00005, 0.98).minimize(loss)

######################
# Initialize Session #
######################
tfconfig = tf.ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth = True
sess = tf.Session(config=tfconfig)

last_iter_num = restore_resnet_tf_variables(
    sess, imdb, 'resnet50', save_epoch=False)

cur_saver = tf.train.Saver()

# generate summary on tensorboard
merged = tf.summary.merge_all()
tb_dir, _ = cfg.get_output_tb_dir('resnet50', imdb.name, val=False)
train_writer = tf.summary.FileWriter(tb_dir, sess.graph)

TOTAL_ITER = ADD_ITER + last_iter_num
T = Timer()
T.tic()
for i in range(last_iter_num + 1, TOTAL_ITER + 1):

    image, gt_labels = imdb.get()

    summary, loss_value, _, ious_value, object_mask_value = \
        sess.run([merged, loss, train_op, ious, object_mask],
                 {input_data: image, label_data: gt_labels})
    # if i>10:
    train_writer.add_summary(summary, i)
    if i % 10 == 0:
        _time = T.toc(average=False)
        print 'iter {:d}/{:d}, total loss: {:.3}, take {:.2}s'\
              .format(i, TOTAL_ITER, loss_value, _time))
        T.tic()

    if i % 40000 == 0:
        save_path=cur_saver.save(sess, os.path.join(
            CKPTS_DIR, cfg.TRAIN_SNAPSHOT_PREFIX + '_iter_' + str(i) + '.ckpt'))
        print "Model saved in file: %s" % save_path

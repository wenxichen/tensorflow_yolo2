"""Use pretained resnet50 on tensorflow to imitate YOLOv1"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import os
import glob

from tensorflow.python.ops import control_flow_ops
from slim_dir.datasets import dataset_factory
from slim_dir.deployment import model_deploy
from slim_dir.nets import nets_factory, resnet_v1, resnet_utils
from slim_dir.preprocessing import preprocessing_factory

import config as cfg
from img_dataset.pascal_voc import pascal_voc
from timer import Timer

slim = tf.contrib.slim

# set hyper parameters
ADD_ITER = 10000
NUM_CLASS = 20
IMAGE_SIZE = cfg.IMAGE_SIZE
S = cfg.S
SIZE_PER_CELL = 1.0 / S
B = cfg.B
LAMBDA_COORD = 5
LAMBDA_NOOBJ = 0.5
BATCH_SIZE = cfg.BATCH_SIZE
OFFSET = np.array(range(S) * S * B)
OFFSET = np.reshape(OFFSET, (B, S, S))
OFFSET = np.transpose(OFFSET, (1,2,0)) #[Y,X,B]

# create database instance
imdb = pascal_voc('trainval', rebuild=cfg.REBUILD)
CKPTS_DIR = cfg.get_ckpts_dir('resnet50', imdb.name)

input_data = tf.placeholder(tf.float32,[None, 224, 224, 3])
label_data = tf.placeholder(tf.float32, [None, S, S, 5 + NUM_CLASS])

def resnet_v1_50(inputs,
                 num_classes=None,
                 is_training=True,
                 global_pool=False,
                 output_stride=None,
                 reuse=None,
                 scope='resnet_v1_50'):
  """ResNet-50 model of [1]. See resnet_v1() for arg and return description."""
  blocks = [
      resnet_utils.Block(
          'block1', resnet_v1.bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
      resnet_utils.Block(
          'block2', resnet_v1.bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
      resnet_utils.Block(
          'block3', resnet_v1.bottleneck, [(1024, 256, 1)] * 5 + [(1024, 256, 2)]),
      resnet_utils.Block(
          'block4', resnet_v1.bottleneck, [(2048, 512, 1)] * 3)
  ]
  return resnet_v1.resnet_v1(inputs, blocks, num_classes, is_training,
                    global_pool=global_pool, output_stride=output_stride,
                    include_root_block=True, spatial_squeeze=False, reuse=reuse,
                    scope=scope)


def get_iou(boxes1, boxes2, scope='iou'):
    """calculate IOUs between boxes1 and boxes2.
    Args:
        boxes1: 5-D tensor [BATCH_SIZE, S, S, B, 4] with last dimension: (x_center, y_center, w, h)
        boxes2: 5-D tensor [BATCH_SIZE, S, S, B, 4] with last dimension: (x_center, y_center, w, h)
    Return:
        iou: 4-D tensor [BATCH_SIZE, S, S, B]
    """
    with tf.variable_scope(scope):
        boxes1 = tf.stack([boxes1[:, :, :, :, 0] - boxes1[:, :, :, :, 2] / 2.0,
                            boxes1[:, :, :, :, 1] - boxes1[:, :, :, :, 3] / 2.0,
                            boxes1[:, :, :, :, 0] + boxes1[:, :, :, :, 2] / 2.0,
                            boxes1[:, :, :, :, 1] + boxes1[:, :, :, :, 3] / 2.0])
        boxes1 = tf.transpose(boxes1, [1, 2, 3, 4, 0])

        boxes2 = tf.stack([boxes2[:, :, :, :, 0] - boxes2[:, :, :, :, 2] / 2.0,
                            boxes2[:, :, :, :, 1] - boxes2[:, :, :, :, 3] / 2.0,
                            boxes2[:, :, :, :, 0] + boxes2[:, :, :, :, 2] / 2.0,
                            boxes2[:, :, :, :, 1] + boxes2[:, :, :, :, 3] / 2.0])
        boxes2 = tf.transpose(boxes2, [1, 2, 3, 4, 0])

        # calculate the left up point & right down point
        lu = tf.maximum(boxes1[:, :, :, :, :2], boxes2[:, :, :, :, :2])
        rd = tf.minimum(boxes1[:, :, :, :, 2:], boxes2[:, :, :, :, 2:])

        # intersection
        intersection = tf.maximum(0.0, rd - lu)
        inter_square = intersection[:, :, :, :, 0] * intersection[:, :, :, :, 1]

        # calculate the boxs1 square and boxs2 square
        square1 = (boxes1[:, :, :, :, 2] - boxes1[:, :, :, :, 0]) * \
            (boxes1[:, :, :, :, 3] - boxes1[:, :, :, :, 1])
        square2 = (boxes2[:, :, :, :, 2] - boxes2[:, :, :, :, 0]) * \
            (boxes2[:, :, :, :, 3] - boxes2[:, :, :, :, 1])

        union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)

    return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)


def get_loss(net, labels, scope='loss_layer'):
    """Create loss from the last fc layer.
    
    Args:
        net: the last fc layer reshaped to (BATCH_SIZE, S, S, 5B+NUM_CLASS).
        labels: the ground truth of shape (BATCH_SIZE, S, S, 5+NUMCLASS) with the following content:
                labels[:,:,:,0] : ground truth of responsibility of the predictor
                labels[:,:,:,1:5] : ground truth of bounding box coordinates in reshaped size
                labels[:,:,:,5:] : ground truth of classes

    Return:
        loss: class loss + object loss + noobject loss + coordinate loss
              with shape (BATCH_SIZE)
    """

    with tf.variable_scope(scope):
        predict_classes = net[:, :, :, :NUM_CLASS]
        # confidence is defined as Pr(Object) * IOU
        predict_confidence = net[:, :, :, NUM_CLASS:NUM_CLASS+B]
        # predict_boxes has last dimenion has [x, y, w, h] * B
        # where (x, y) "represent the center of the box relative to the bounds of the grid cell"
        predict_boxes = tf.reshape(net[:, :, :, NUM_CLASS+B:], [BATCH_SIZE, S, S, B, 4])

        ########################
        # calculate class loss #
        ########################
        responsible =  tf.reshape(labels[:, :, :, 0], [BATCH_SIZE, S, S, 1]) # [BATCH_SIZE, S, S]
        classes = labels[:, :, :, 5:]

        class_delta = responsible * (predict_classes - classes) # [:,S,S,NUM_CLASS]
        class_loss = tf.reduce_mean(tf.reduce_sum(tf.square(class_delta), axis=[1, 2, 3]), name='class_loss')

        #############################
        # calculate coordinate loss #
        #############################
        gt_boxes = tf.reshape(labels[:, :, :, 1:5], [BATCH_SIZE, S, S, 1, 4])
        gt_boxes = tf.tile(gt_boxes, [1, 1, 1, B, 1]) / float(IMAGE_SIZE)

        # add offsets to the predicted box and ground truth box coordinates to get absolute coordinates between 0 and 1
        offset = tf.constant(OFFSET, dtype=tf.float32)
        offset = tf.reshape(offset, [1, S, S, B])
        offset = tf.tile(offset, [BATCH_SIZE, 1, 1, 1])
        predict_xs = (predict_boxes[:, :, :, :, 0] + offset) * IMAGE_SIZE / float(S)
        gt_rel_xs = gt_boxes[:, :, :, :, 0] * S - offset
        offset = tf.transpose(offset, (0, 2, 1, 3))
        predict_ys = predict_boxes[:, :, :, :, 1] + offset * IMAGE_SIZE / float(S)
        gt_rel_ys = gt_boxes[:, :, :, :, 1] * S - offset
        predict_ws = predict_boxes[:, :, :, :, 2] * IMAGE_SIZE
        gt_rel_ws = tf.sqrt(gt_boxes[:, :, :, :, 2])
        predict_hs = predict_boxes[:, :, :, :, 3] * IMAGE_SIZE
        gt_rel_hs = tf.sqrt(gt_boxes[:, :, :, :, 3])
        predict_boxes_offset = tf.stack([predict_xs, predict_ys, predict_ws, predict_hs], axis=4)
        # gt_boxes_offset = tf.stack([gt_xs, gt_ys, gt_ws, gt_hs], axis=4)
        

        # calculate IOUs
        ious = get_iou(predict_boxes_offset, gt_boxes)
        
        # calculate object masks and nonobject masks tensor [BATCH_SIZE, S, S, B]
        object_mask = tf.reduce_max(ious, 3, keep_dims=True)
        object_mask = tf.cast((ious >= object_mask), tf.float32) * responsible
        noobject_mask = tf.ones_like(object_mask, dtype=tf.float32) - object_mask

        # coordinate loss
        coord_mask = tf.expand_dims(object_mask, 4)
        boxes_delta_xs = predict_boxes[:, :, :, :, 0] - gt_rel_xs
        boxes_delta_ys = predict_boxes[:, :, :, :, 1] - gt_rel_ys
        boxes_delta_ws = tf.sqrt(predict_boxes[:, :, :, :, 2]) - gt_rel_ws
        boxes_delta_hs = tf.sqrt(predict_boxes[:, :, :, :, 3]) - gt_rel_hs
        boxes_delta = tf.stack([boxes_delta_xs, boxes_delta_ys, boxes_delta_ws, boxes_delta_hs], axis=4)
        boxes_delta = coord_mask * boxes_delta
        coord_loss = tf.reduce_mean(tf.reduce_sum(tf.square(boxes_delta), axis=[1, 2, 3, 4]), name='coord_loss') * LAMBDA_COORD

        #########################
        # calculate object loss #
        #########################
        # object loss
        object_delta = object_mask * (predict_confidence - ious)
        object_loss = tf.reduce_mean(tf.reduce_sum(tf.square(object_delta), axis=[1, 2, 3]), name='object_loss')
        # noobject loss
        noobject_delta = noobject_mask * predict_confidence
        noobject_loss = tf.reduce_mean(tf.reduce_sum(tf.square(noobject_delta), axis=[1, 2, 3]), name='noobject_loss') * LAMBDA_NOOBJ

        tf.summary.scalar('class_loss', class_loss)
        tf.summary.scalar('object_loss', object_loss)
        tf.summary.scalar('noobject_loss', noobject_loss)
        tf.summary.scalar('coord_loss', coord_loss)

        tf.summary.histogram('boxes_delta_x', boxes_delta_xs)
        tf.summary.histogram('boxes_delta_y', boxes_delta_ys)
        tf.summary.histogram('boxes_delta_w', boxes_delta_ws)
        tf.summary.histogram('boxes_delta_h', boxes_delta_hs)
        tf.summary.histogram('iou', ious)

    return class_loss + object_loss + noobject_loss + coord_loss, ious, object_mask


def get_tf_variables(sess, retrain=False):
    """Get the tensorflow variables needed for the network.
    Initialize variables or read from weights file if there is no checkpoint to restore,
    otherise restore from the latest checkpoint.
    """

    # Find previous snapshots if there is any to restore from
    sfiles = os.path.join(CKPTS_DIR, cfg.TRAIN_SNAPSHOT_PREFIX + '_iter_*.ckpt.meta')
    sfiles = glob.glob(sfiles)
    sfiles.sort(key=os.path.getmtime)
    # Get the snapshot name in TensorFlow
    sfiles = [ss.replace('.meta', '') for ss in sfiles]

    nfiles = os.path.join(CKPTS_DIR, cfg.TRAIN_SNAPSHOT_PREFIX + '_iter_*.pkl')
    nfiles = glob.glob(nfiles)
    nfiles.sort(key=os.path.getmtime)

    lsf = len(sfiles)

    if lsf == 0:
        #####################################################################
        # initialize new variables and restore all the pretrained variables #
        #####################################################################
        # get all variable names
        # variable_names = [n.name for n in tf.get_default_graph().as_graph_def().node]

        # get tensor by name
        # t = tf.get_default_graph().get_tensor_by_name("tensor_name")

        # get variables by scope
        # vars_in_scope = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='scope_name')

        # op to initialized variables that does not have pretrained weights
        adam_vars = [var for var in tf.global_variables() if 'Adam' in var.name or 'beta1_power' in var.name or 'beta2_power' in var.name]
        uninit_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='yolo_fc1') \
                    + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='yolo_fc2') \
                    + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='loss_layer') \
                    + adam_vars
        init_op = tf.variables_initializer(uninit_vars)

        # Restore only the convolutional layers:
        variables_to_restore = slim.get_variables_to_restore(exclude=['yolo_fc1', 'yolo_fc2', 'loss_layer'])
        for var in uninit_vars:
            if var in variables_to_restore:
                variables_to_restore.remove(var)
        saver = tf.train.Saver(variables_to_restore)

        print('Initializing new variables to train from downloaded resnet50 weights')
        sess.run(init_op)
        saver.restore(sess, os.path.join(cfg.WEIGHTS_PATH, 'resnet_v1_50.ckpt'))

        return 0

    else:
        print('Restorining model snapshots from {:s}'.format(sfiles[-1]))
        saver = tf.train.Saver()
        saver.restore(sess, str(sfiles[-1]))
        print('Restored.')

        fnames = sfiles[-1].split('_')
        return int(fnames[-1][:-5])


# get the right arg_scope in order to load weights
with slim.arg_scope(resnet_v1.resnet_arg_scope()):
    # net is shape [batch_size, S, S, 2048] if input size is 244 x 244
    net, end_points = resnet_v1_50(input_data)

net = slim.flatten(net)

fcnet = slim.fully_connected(net, 4096, scope='yolo_fc1')

fcnet = tf.nn.dropout(fcnet, 0.5)

# in this case 7x7x30
fcnet = slim.fully_connected(net, S*S*(5*B+NUM_CLASS), scope='yolo_fc2')

grid_net = tf.reshape(fcnet,[-1, S, S, (5*B+NUM_CLASS)])

loss, ious, object_mask = get_loss(grid_net, label_data)
tf.summary.scalar('total_loss', loss)

train_op = tf.train.AdamOptimizer().minimize(loss)

######################
# Initialize Session #
######################
tfconfig = tf.ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth = True
sess = tf.Session(config=tfconfig)

last_iter_num = get_tf_variables(sess)

cur_saver = tf.train.Saver()

# generate summary on tensorboard
merged = tf.summary.merge_all()
tb_dir = cfg.get_output_tb_dir('resnet50', imdb.name)
train_writer = tf.summary.FileWriter(tb_dir, sess.graph)

TOTAL_ITER = ADD_ITER + last_iter_num
T = Timer()
T.tic()
for i in range(last_iter_num + 1, TOTAL_ITER + 1):

    image, gt_labels = imdb.get()

    summary, loss_value, _, ious_value, object_mask_value = \
        sess.run([merged, loss, train_op, ious, object_mask], {input_data:image, label_data:gt_labels})
    # if i>10:
    train_writer.add_summary(summary, i)
    if i % 10 == 0:
        _time = T.toc(average=False)
        print('iter {:d}/{:d}, total loss: {:.3}, take {:.2}s'.format(i, TOTAL_ITER, loss_value, _time))
        T.tic()

    if i % 2000 == 0:
        save_path = cur_saver.save(sess, os.path.join(CKPTS_DIR, cfg.TRAIN_SNAPSHOT_PREFIX + '_iter_' + str(i) + '.ckpt'))
        print("Model saved in file: %s" % save_path)


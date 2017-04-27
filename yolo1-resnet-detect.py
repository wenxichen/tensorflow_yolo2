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
import sys

from tensorflow.python.ops import control_flow_ops
from slim_dir.datasets import dataset_factory
from slim_dir.deployment import model_deploy
from slim_dir.nets import nets_factory, resnet_v1, resnet_utils
from slim_dir.preprocessing import preprocessing_factory

import config as cfg
from img_dataset.pascal_voc import pascal_voc
from timer import Timer

slim = tf.contrib.slim


ADD_EPOCH = 4000
NUM_CLASS = 20
IMAGE_SIZE = cfg.IMAGE_SIZE
S = cfg.S
B = cfg.B
LAMBDA_COORD = 5
LAMBDA_NOOBJ = 0.5
BATCH_SIZE = cfg.BATCH_SIZE
OFFSET = np.array(range(S) * S * B)
OFFSET = np.reshape(OFFSET, (B, S, S))
OFFSET = np.transpose(OFFSET, (1,2,0)) #[Y,X,B]

# create database instance
imdb = pascal_voc('trainval')
CKPTS_DIR = cfg.get_ckpts_dir('resnet50', imdb.name)

input_data = tf.placeholder(tf.float32,[None, 224, 224, 3])
# labels = tf.placeholder(tf.float32, [None, S, S, 5 + NUM_CLASS])

# read in the test image
image = cv2.imread('/Users/wenxichen/Desktop/dl/dl_yolo2/tests/testImg5.jpg')
image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
image = image.astype(np.float32)
image = (image / 255.0) * 2.0 - 1.0
image = image.reshape((1, 224, 224, 3))

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
                labels[:,:,:,1:5] : ground truth bounding box coordinates
                labels[:,:,:,5:] : ground truth classes

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
        # TODO: need to make the ground truth labels last dimension [x, y, w, h]
        # with the same rule as predict_boxes
        gt_boxes = tf.reshape(labels[:, :, :, 1:5], [BATCH_SIZE, S, S, 1, 4])
        gt_boxes = tf.tile(gt_boxes, [1, 1, 1, B, 1]) / float(IMAGE_SIZE)

        # add offsets to the predicted box and ground truth box coordinates to get absolute coordinates between 0 and 1
        offset = tf.constant(OFFSET, dtype=tf.float32)
        offset = tf.reshape(offset, [1, S, S, B])
        offset = tf.tile(offset, [BATCH_SIZE, 1, 1, 1])
        predict_xs = predict_boxes[:, :, :, :, 0] + (offset / float(S))
        gt_xs = gt_boxes[:, :, :, :, 0] + (offset / float(S))
        offset = tf.transpose(offset, (0, 2, 1, 3))
        predict_ys = predict_boxes[:, :, :, :, 1] + (offset / float(S))
        gt_ys = gt_boxes[:, :, :, :, 1] + (offset / float(S))
        predict_ws = predict_boxes[:, :, :, :, 2]
        gt_ws = gt_boxes[:, :, :, :, 2]
        predict_hs = predict_boxes[:, :, :, :, 3]
        gt_hs = gt_boxes[:, :, :, :, 3]
        predict_boxes_offset = tf.stack([predict_xs, predict_ys, predict_ws, predict_hs], axis=4)
        gt_boxes_offset = tf.stack([gt_xs, gt_ys, gt_ws, gt_hs], axis=4)
        

        # calculate IOUs
        ious = get_iou(predict_boxes_offset, gt_boxes_offset)
        
        # calculate object masks and nonobject masks tensor [BATCH_SIZE, S, S, B]
        object_mask = tf.reduce_max(ious, 3, keep_dims=True)
        object_mask = tf.cast((ious >= object_mask), tf.float32) * responsible
        noobject_mask = tf.ones_like(object_mask, dtype=tf.float32) - object_mask

        # coordinate loss
        coord_mask = tf.expand_dims(object_mask, 4)
        boxes_delta_xs = predict_boxes[:, :, :, :, 0] - gt_boxes[:, :, :, :, 0]
        boxes_delta_ys = predict_boxes[:, :, :, :, 1] - gt_boxes[:, :, :, :, 1]
        boxes_delta_ws = tf.sqrt(predict_boxes[:, :, :, :, 2]) - tf.sqrt(gt_boxes[:, :, :, :, 2])
        boxes_delta_hs = tf.sqrt(predict_boxes[:, :, :, :, 3]) - tf.sqrt(gt_boxes[:, :, :, :, 3])
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

    return class_loss + object_loss + noobject_loss + coord_loss


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

######################
# Initialize Session #
######################
tfconfig = tf.ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth = True
sess = tf.Session(config=tfconfig)

# Load existing checkpoints
# Find previous snapshots if there is any to restore from
sfiles = os.path.join(CKPTS_DIR, cfg.TRAIN_SNAPSHOT_PREFIX + '_epoch_*.ckpt.meta')
sfiles = glob.glob(sfiles)
sfiles.sort(key=os.path.getmtime)
# Get the snapshot name in TensorFlow
sfiles = [ss.replace('.meta', '') for ss in sfiles]
lsf = len(sfiles)

if lsf == 0:
    print('No checkpoints to load')
    sys.exit(-1)

else:
    print('Restorining model snapshots from {:s}'.format(sfiles[-1]))
    saver = tf.train.Saver()
    saver.restore(sess, str(sfiles[-1]))
    print('Restored.')

predicts = sess.run(grid_net, {input_data:image})


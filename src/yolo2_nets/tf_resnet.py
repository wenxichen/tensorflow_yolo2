"""Use pretained resnet50 from tensorflow slim"""

import tensorflow as tf
import numpy as np
import os

from slim_dir.nets import resnet_v1, resnet_utils

import config as cfg
from utils.timer import Timer
from yolo2_nets.net_utils import get_resnet_tf_variables

slim = tf.contrib.slim

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
                               include_root_block=True, spatial_squeeze=global_pool, reuse=reuse,
                               scope=scope)
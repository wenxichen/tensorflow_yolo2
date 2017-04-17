from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.ops import control_flow_ops
from datasets import dataset_factory
from deployment import model_deploy
from nets import nets_factory, resnet_v1, resnet_utils
from preprocessing import preprocessing_factory

slim = tf.contrib.slim


# ALPHA = 0.1

x = tf.placeholder(tf.float32,[None,448,448,3])

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


# get the right arg_scope in order to load weights
with slim.arg_scope(resnet_v1.resnet_arg_scope()):
  # net is shape [batch_size, 7, 7, 2048] if input size is 244 x 244
  net, end_points = resnet_v1_50(x)

net = slim.flatten(net)

fcnet = slim.fully_connected(net, 4096, scope='yolo_fc1')

fcnet = tf.nn.dropout(fcnet, 0.5)

fcnet = slim.fully_connected(net, 7*7*30, scope='yolo_fc2')

output = tf.reshape(fcnet,[7,7,30])

# get all variable names
# variable_names = [n.name for n in tf.get_default_graph().as_graph_def().node]

# get tensor by name
# t = tf.get_default_graph().get_tensor_by_name("tensor_name")

# get variables by scope
# vars_in_scope = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='scope_name')

# op to initialized variables that does not have pretrained weights
uninit_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='yolo_fc1') + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='yolo_fc2')
init_op = tf.variables_initializer(uninit_vars)


########op restore all the pretrained variables###########
# Restore only the convolutional layers:
variables_to_restore = slim.get_variables_to_restore(exclude=['yolo_fc1', 'yolo_fc2'])
saver = tf.train.Saver(variables_to_restore)

with tf.Session() as sess:
  sess.run(init_op)
  
  saver.restore(sess, '/Users/wenxichen/Desktop/TensorFlow/ckpts/resnet_v1_50.ckpt')
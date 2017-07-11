import os
import numpy as np
import tensorflow as tf

alpha = 0.1


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')


def max_pool(x, pool_size, stride):
    return tf.nn.max_pool(x, ksize=[1, pool_size, pool_size, 1], strides=[1, stride, stride, 1], padding='SAME')


def avg_pool(x, pool_size, stride):
    return tf.nn.avg_pool(x, ksize=[1, pool_size, pool_size, 1], strides=[1, stride, stride, 1], padding='SAME')


def conv_layer(x, filter_size, input_chl, output_chl, stride):
    W_conv = weight_variable([filter_size, filter_size, input_chl, output_chl])
    b_conv = bias_variable([output_chl])
    h_conv = conv2d(x, W_conv, stride) + b_conv
    return h_conv


def conv_bn_layer(x, filter_size, input_chl, output_chl, stride, is_training):
    h_conv_activated = conv_layer(
        x, filter_size, input_chl, output_chl, stride)
    h_bn_conv = tf.layers.batch_normalization(h_conv_activated,
                                         center=True, scale=True,
                                         training=is_training)
    h_activated = tf.maximum(alpha * h_bn_conv, h_bn_conv)
    return h_activated


def fc_layer(x, input_dim, output_dim, flat=False, linear=False):
    W_fc = weight_variable([input_dim, output_dim])
    b_fc = bias_variable([output_dim])
    if flat:
        x = tf.reshape(x, [-1, input_dim])
    h_fc = tf.add(tf.matmul(x, W_fc), b_fc)
    if linear:
        return h_fc
    return tf.maximum(alpha * h_fc, h_fc)


def darknet19(inputs,
              num_classes=None,
              is_training=True,
              global_pool=True,
              output_stride=None,
              reuse=None,
              scope='darknet19'):
    """Darknet19 architecture built according to 
    YOLO9000: Better, Faster, Stronger by J. Redmon and A. Farhadi.

    Args:
        inputs: 4D numpy array

    Returns:
        logits: 
    """
    # TODO: may need to change num_classes in the implementation to make the net more flexible

    with tf.variable_scope(scope, 'darknet19', [inputs], reuse=reuse) as sc:
        # end_points_collection = sc.name + '_end_points'
        # with slim.arg_scope([],
        #                     outputs_collections=end_points_collection):
        #   with slim.arg_scope([], is_training=is_training):

        h_conv1 = conv_bn_layer(inputs, 3, 3, 32, 1, is_training)
        h_pool1 = max_pool(h_conv1, 2, 2)

        h_conv2 = conv_bn_layer(h_pool1, 3, 32, 64, 1, is_training)
        h_pool2 = max_pool(h_conv2, 2, 2)

        h_conv3 = conv_bn_layer(h_pool2, 3, 64, 128, 1, is_training)
        h_conv4 = conv_bn_layer(h_conv3, 3, 128, 64, 1, is_training)
        h_conv5 = conv_bn_layer(h_conv4, 3, 64, 128, 1, is_training)
        h_pool3 = max_pool(h_conv5, 2, 2)

        h_conv6 = conv_bn_layer(h_pool3, 3, 128, 256, 1, is_training)
        h_conv7 = conv_bn_layer(h_conv6, 1, 256, 128, 1, is_training)
        h_conv8 = conv_bn_layer(h_conv7, 3, 128, 256, 1, is_training)
        h_pool4 = max_pool(h_conv8, 2, 2)

        h_conv9 = conv_bn_layer(h_pool4, 3, 256, 512, 1, is_training)
        h_conv10 = conv_bn_layer(h_conv9, 1, 512, 256, 1, is_training)
        h_conv11 = conv_bn_layer(h_conv10, 3, 256, 512, 1, is_training)
        h_conv12 = conv_bn_layer(h_conv11, 1, 512, 256, 1, is_training)
        h_conv13 = conv_bn_layer(h_conv12, 3, 256, 512, 1, is_training)
        h_pool5 = max_pool(h_conv13, 2, 2)

        h_conv14 = conv_bn_layer(h_pool5, 3, 512, 1024, 1, is_training)
        h_conv15 = conv_bn_layer(h_conv14, 1, 1024, 512, 1, is_training)
        h_conv16 = conv_bn_layer(h_conv15, 3, 512, 1024, 1, is_training)
        h_conv17 = conv_bn_layer(h_conv16, 1, 1024, 512, 1, is_training)
        h_conv18 = conv_bn_layer(h_conv17, 3, 512, 1024, 1, is_training)

        # ======
        h_conv19 = conv_bn_layer(h_conv18, 1, 1024, 1000, 1, is_training)
        h_avgpool = tf.layers.average_pooling2d(h_conv19, [7, 7], [7, 7])
        logits = tf.reshape(h_avgpool, [-1, 1000])
        # loss = tf.nn.softmax(h_avgpool)

        # end_points = slim.utils.convert_collection_to_dict(end_points_collection)
        # if num_classes is not None:
        #   end_points['predictions'] = slim.softmax(logits, scope='predictions')
        return logits

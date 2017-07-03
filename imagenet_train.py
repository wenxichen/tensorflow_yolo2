"""Train ILSVRC2017 Data using homemade scripts."""

import cv2
import tensorflow as tf

import config as cfg
from img_dataset.ilsvrc2017_cls import ilsvrc_cls
from darknet import darknet19
from utils.timer import Timer

imdb = ilsvrc_cls('train')


input_data = tf.placeholder(tf.float32, [None, 224, 224, 3])
label_data = tf.placeholder(tf.int32, None)

logits = darknet19(input_data)
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_data, logits=logits)
loss = tf.reduce_mean(loss)
train_op = tf.train.AdamOptimizer().minimize(loss)

######################
# Initialize Session #
######################
tfconfig = tf.ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth = True
sess = tf.Session(config=tfconfig)

# initialize variables, assume all vars are new now
init_op = tf.global_variables_initializer()
sess.run(init_op)

T = Timer()
for i in range(1000):
    T.tic()
    images, labels = imdb.get()
    _, loss_val = sess.run([train_op, loss], {input_data: images, label_data: labels})
    _time = T.toc(average=False)

    print('iter {:d}/{:d}, total loss: {:.3}, take {:.2}s'.format(i+1, 1000, loss_val, _time))
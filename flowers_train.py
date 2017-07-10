"""Train ILSVRC2017 Data using homemade scripts."""

import cv2
import os
import math
import tensorflow as tf

import config as cfg
from yolo2_nets.darknet import darknet19
from yolo2_nets.net_utils import get_ordered_ckpts
from utils.timer import Timer
from img_dataset.TF_flowers import tf_flowers

slim = tf.contrib.slim

imdb = tf_flowers(0.2, data_aug=True)
CKPTS_DIR = cfg.get_ckpts_dir('darknet19', imdb.name)


input_data = tf.placeholder(tf.float32, [None, 224, 224, 3])
label_data = tf.placeholder(tf.int64, None)
is_training = tf.placeholder(tf.bool)

logits = darknet19(input_data, is_training=is_training)
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=label_data, logits=logits)
loss = tf.reduce_mean(loss)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = tf.train.AdamOptimizer(learning_rate=0.0002).minimize(loss)

correct_pred = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int64), label_data)
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
fnames = ckpts[-1].split('_')
old_iter = int(fnames[-1][:-5])

# simple model saver
cur_saver = tf.train.Saver()

T = Timer()
for i in range(old_iter, old_iter + 50000):
    T.tic()
    images, labels = imdb.get_train()
    _, loss_value, acc_value = sess.run([train_op, loss, accuracy],
                                        {input_data: images, label_data: labels, is_training: 1})

    val_images, val_labels = imdb.get_val()
    val_loss_value, val_acc_value = sess.run([loss, accuracy],
                                             {input_data: val_images, label_data: val_labels, is_training: 0})

    _time = T.toc(average=False)
    print('iter {:d}/{:d}, train_loss: {:.3}, trian_acc: {:.3}, val_loss: {:.3} val_acc: {:.3} take {:.2}s'
          .format(i + 1, old_iter + 50000, loss_value, acc_value, val_loss_value, val_acc_value, _time))


save_path = cur_saver.save(sess, os.path.join(
    CKPTS_DIR, cfg.TRAIN_SNAPSHOT_PREFIX + '_iter_' + str(old_iter + 50000) + '.ckpt'))
print("Model saved in file: %s" % save_path)

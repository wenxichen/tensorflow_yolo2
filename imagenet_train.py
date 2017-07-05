"""Train ILSVRC2017 Data using homemade scripts."""

import cv2
import os
import math
import tensorflow as tf

import config as cfg
# from img_dataset.ilsvrc2017_cls import ilsvrc_cls
from img_dataset.ilsvrc2017_cls_multithread import ilsvrc_cls
from darknet import darknet19
from utils.timer import Timer


imdb = ilsvrc_cls('train')
CKPTS_DIR = cfg.get_ckpts_dir('darknet19', imdb.name)
total_batch = int(math.ceil(1281167 / float(cfg.BATCH_SIZE))) 


input_data = tf.placeholder(tf.float32, [None, 224, 224, 3])
label_data = tf.placeholder(tf.int32, None)

logits = darknet19(input_data)
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=label_data, logits=logits)
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

# simple model saver
cur_saver = tf.train.Saver()

T = Timer()
for i in range(total_batch * 2 + 1):
    T.tic()
    images, labels = imdb.get()
    _, loss_val = sess.run(
        [train_op, loss], {input_data: images, label_data: labels})
    _time = T.toc(average=False)

    print('epoch {:d}, iter {:d}/{:d}, total loss: {:.3}, take {:.2}s'
          .format(imdb.epoch, i + 1, total_batch, loss_val, _time))


save_path = cur_saver.save(sess, os.path.join(
    CKPTS_DIR, cfg.TRAIN_SNAPSHOT_PREFIX + '_iter_' + '10000' + '.ckpt'))
print("Model saved in file: %s" % save_path)

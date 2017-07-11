"""Train ILSVRC2017 Data using homemade scripts."""

import cv2
import os
import math
import tensorflow as tf
from multiprocessing import Process, Queue

import config as cfg
from img_dataset.ilsvrc2017_cls_multithread import ilsvrc_cls
from yolo2_nets.darknet import darknet19
from utils.timer import Timer


def get_validation_process(imdb, queue_in, queue_out):
    """Get validation dataset. Run in a child process."""
    while True:
        queue_in.get()
        images, labels = imdb.get()
        queue_out.put([images, labels])


imdb = ilsvrc_cls('train', data_aug=True, multithread=cfg.MULTITHREAD)
val_imdb = ilsvrc_cls('val', batch_size=128)
# set up child process for getting validation data
queue_in = Queue()
queue_out = Queue()
val_data_process = Process(target=get_validation_process,
                           args=(val_imdb, queue_in, queue_out))
val_data_process.start()
queue_in.put(True)  # start getting the first batch

CKPTS_DIR = cfg.get_ckpts_dir('darknet19', imdb.name)


input_data = tf.placeholder(tf.float32, [None, 224, 224, 3])
label_data = tf.placeholder(tf.int32, None)
is_training = tf.placeholder(tf.bool)

logits = darknet19(input_data)
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=label_data, logits=logits)
loss = tf.reduce_mean(loss)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = tf.train.AdamOptimizer().minimize(loss)

correct_pred = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), label_data)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

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
for i in range(imdb.total_batch * 2 + 1):
    T.tic()
    images, labels = imdb.get()
    _, loss_value, acc_value = sess.run(
        [train_op, loss, accuracy], {input_data: images, label_data: labels, is_training: 1})
    _time = T.toc(average=False)

    print('epoch {:d}, iter {:d}/{:d}, training loss: {:.3}, training acc: {:.3}, take {:.2}s'
          .format(imdb.epoch, (i + 1) % imdb.total_batch, imdb.total_batch, loss_value, acc_value, _time))

    if (i + 1) % 25 == 0:
        T.tic()
        val_images, val_labels = queue_out.get()
        val_loss_value, val_acc_value = sess.run(
            [loss, accuracy], {input_data: val_images, label_data: val_labels, is_training: 0})
        _val_time = T.toc(average=False)
        print('###validation loss: {:.3}, validation acc: {:.3}, take {:.2}s'
              .format(val_loss_value, val_acc_value, _val_time))
        queue_in.put(True) 

save_path = cur_saver.save(sess, os.path.join(
    CKPTS_DIR, cfg.TRAIN_SNAPSHOT_PREFIX + '_epoch_' + str(imdb.epoch-1) + '.ckpt'))
print("Model saved in file: %s" % save_path)

# terminate child processes
if cfg.MULTITHREAD:
    imdb.close_all_processes()
queue_in.cancel_join_thread()
queue_out.cancel_join_thread()
val_data_process.terminate()

"""Train resnet50 on ILSVRC2017 Data using homemade scripts."""

import tensorflow as tf
from multiprocessing import Process, Queue

import os
import sys
FILE_DIR = os.path.dirname(__file__)
sys.path.append(FILE_DIR + '/../')

import config as cfg
from img_dataset.ilsvrc_cls_multithread_scipy import ilsvrc_cls
from yolo2_nets import inception_resnet_v2
from yolo2_nets.net_utils import restore_inception_resnet_variables_from_weight
from utils.timer import Timer
from utils.helpers import add_contrast_on_batch
from scipy import ndimage

slim = tf.contrib.slim

TRAIN_BATCH_SIZE = 18

##############################################################
# Use Inception v3 to generate adversarial examples to train #
##############################################################
from cleverhans.attacks import FastGradientMethod
import numpy as np
from tensorflow.contrib.slim.nets import inception

checkpoint_path = "/home/wenxi/Projects/tensorflow_yolo2/weights/inception_v3.ckpt"
max_epsilon = 16.0
eps = 2.0 * max_epsilon / 255.0
batch_shape = [TRAIN_BATCH_SIZE, 299, 299, 3]
tensorflow_master = ""


class InceptionModel(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.built = False

    def __call__(self, x_input):
        """Constructs model and return probabilities for given input."""
        reuse = True if self.built else None
        with slim.arg_scope(inception.inception_v3_arg_scope()):
            _, end_points = inception.inception_v3(
                x_input, num_classes=self.num_classes, is_training=False,
                reuse=reuse)
        self.built = True
        output = end_points['Predictions']
        probs = output.op.inputs[0]
        return probs


g_inception_v3 = tf.Graph()
with g_inception_v3.as_default():
    x_input = tf.placeholder(tf.float32, shape=batch_shape)
    model = InceptionModel(1001)

    fgsm = FastGradientMethod(model)
    x_adv = fgsm.generate(x_input, eps=eps, clip_min=-1., clip_max=1.)

    saver = tf.train.Saver(slim.get_model_variables())
    session_creator = tf.train.ChiefSessionCreator(
        scaffold=tf.train.Scaffold(saver=saver),
        checkpoint_filename_with_path=checkpoint_path,
        master=tensorflow_master)
    sess_inception_v3 = tf.train.MonitoredSession(
        session_creator=session_creator)


##############################
# inception resnet predictor #
##############################
inception_imagenet_labels = ['background']
with open(os.path.join(cfg.SRC_DIR, 'img_dataset', 'imagenet_lsvrc_2015_synsets.txt'), 'r') as f:
    for line in f.readlines():
        if line.strip():
            inception_imagenet_labels.append(line.strip())
assert len(inception_imagenet_labels) == 1001
synset_to_ind = dict(list(zip(inception_imagenet_labels, list(range(1001)))))


def get_validation_process(imdb, queue_in, queue_out):
    """Get validation dataset. Run in a child process."""
    while True:
        queue_in.get()
        images, labels = imdb.get()
        queue_out.put([images, labels])


# NOTE: check fix the data imread and label
imdb = ilsvrc_cls('train',
                  multithread=cfg.MULTITHREAD, batch_size=TRAIN_BATCH_SIZE, image_size=299, random_noise=True)
val_imdb = ilsvrc_cls('val', batch_size=18, image_size=299, random_noise=True)
# set up child process for getting validation data
queue_in = Queue()
queue_out = Queue()
val_data_process = Process(target=get_validation_process,
                           args=(val_imdb, queue_in, queue_out))
val_data_process.start()
queue_in.put(True)  # start getting the first batch

CKPTS_DIR = cfg.get_ckpts_dir('inception_resnet', imdb.name)
TENSORBOARD_TRAIN_DIR, TENSORBOARD_VAL_DIR = cfg.get_output_tb_dir(
    'inception_resnet', imdb.name)
TENSORBOARD_TRAIN_ADV_DIR = os.path.abspath(os.path.join(
    cfg.ROOT_DIR, 'tensorboard', 'inception_resnet', imdb.name, 'train_adv'))
if not os.path.exists(TENSORBOARD_TRAIN_ADV_DIR):
    os.makedirs(TENSORBOARD_TRAIN_ADV_DIR)
TENSORBOARD_VAL_ADV_DIR = os.path.abspath(os.path.join(
    cfg.ROOT_DIR, 'tensorboard', 'inception_resnet', imdb.name, 'val_adv'))
if not os.path.exists(TENSORBOARD_VAL_ADV_DIR):
    os.makedirs(TENSORBOARD_VAL_ADV_DIR)

g_inception_resnet = tf.Graph()
with g_inception_resnet.as_default():
    input_data = tf.placeholder(tf.float32, [None, 299, 299, 15])
    label_data = tf.placeholder(tf.int32, None)
    is_training = tf.placeholder(tf.bool)

    with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
        # NOTE: check fix the number of classes
        logits, end_points = inception_resnet_v2.inception_resnet_v2(input_data, num_classes=1001,
                                                                     is_training=is_training)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=label_data, logits=logits)
    loss = tf.reduce_mean(loss)

    # NOTE: check fix the variable to train
    vars_to_train_trf = tf.get_collection(
    tf.GraphKeys.GLOBAL_VARIABLES, scope='InceptionResnetV2/Conv2d_tr_3x3')
    vars_to_train_1 = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope='InceptionResnetV2/Conv2d_1a_3x3')
    vars_to_train_2 = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope='InceptionResnetV2/Conv2d_2a_3x3')
    # vars_to_train_3 = tf.get_collection(
    #     tf.GraphKeys.GLOBAL_VARIABLES, scope='InceptionResnetV2/Conv2d_2b_3x3')
    # vars_to_train_4 = tf.get_collection(
    #     tf.GraphKeys.GLOBAL_VARIABLES, scope='InceptionResnetV2/Conv2d_3b_1x1')
    # vars_to_train_5 = tf.get_collection(
    #     tf.GraphKeys.GLOBAL_VARIABLES, scope='InceptionResnetV2/Conv2d_4a_3x3')
    # assert len(vars_to_train_1) != 0
    # assert len(vars_to_train_2) != 0
    # assert len(vars_to_train_trf) != 0
    # print "###vars to train###:", vars_to_train
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        # train_op_trf = tf.train.AdamOptimizer().minimize(loss, var_list=vars_to_train_trf)
        train_op_1 = tf.train.AdamOptimizer(
            0.00001).minimize(loss, var_list=(vars_to_train_1 + vars_to_train_2))
        # train_op_2 = tf.train.AdamOptimizer(
        #     0.00005).minimize(loss, var_list=vars_to_train_2)
        train_op_2 = tf.train.AdamOptimizer().minimize(loss, var_list=(vars_to_train_trf))
        # train_op_4 = tf.train.AdamOptimizer(
        #     0.00001).minimize(loss, var_list=vars_to_train_4)
        # train_op_5 = tf.train.AdamOptimizer(
        #     0.00001).minimize(loss, var_list=vars_to_train_5)
        train_op = tf.group(train_op_1, train_op_2)

    correct_pred = tf.equal(
        tf.cast(tf.argmax(logits, 1), tf.int32), label_data)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)

    ######################
    # Initialize Session #
    ######################
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    sess_inception_resent = tf.Session(config=tfconfig)

    # TODO: fix restore
    # old_epoch = restore_inception_resnet_variables_from_weight(
    #     sess_inception_resent, os.path.join(cfg.WEIGHTS_PATH, 'ens_adv_inception_resnet_v2.ckpt'))
    # imdb.epoch = old_epoch + 1

    # TODO: not sure why adam needs to be reinitialized
    adam_vars = [var for var in tf.global_variables()
                 if 'Adam' in var.name or
                 'beta1_power' in var.name or
                 'beta2_power' in var.name]
    uninit_vars = adam_vars \
        + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                            scope='InceptionResnetV2/Conv2d_tr_3x3')
    init_op = tf.variables_initializer(uninit_vars)
    variables_to_restore = slim.get_variables_to_restore()
    for var in uninit_vars:
        if var in variables_to_restore:
            variables_to_restore.remove(var)
    ckpt_file = os.path.join(CKPTS_DIR, "train_iter_47500.ckpt")
    print 'Restorining model snapshots from {:s}'.format(ckpt_file)
    saver = tf.train.Saver(variables_to_restore)
    sess_inception_resent.run(init_op)
    saver.restore(sess_inception_resent, ckpt_file)
    print 'Restored.'

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(TENSORBOARD_TRAIN_DIR)
    val_writer = tf.summary.FileWriter(TENSORBOARD_VAL_DIR)
    train_adv_writer = tf.summary.FileWriter(TENSORBOARD_TRAIN_ADV_DIR)
    val_adv_writer = tf.summary.FileWriter(TENSORBOARD_VAL_ADV_DIR)

    # simple model saver
    cur_saver = tf.train.Saver()

T = Timer()
for i in range(47500, 200000):
    T.tic()
    images, labels = imdb.get()
    # images = ndimage.gaussian_filter(
    #     images, sigma=(0, 1, 1, 0), order=0, truncate=2.0)
    contrast_images = add_contrast_on_batch(images)
    labels = [synset_to_ind[imdb.classes[int(item)]] for item in labels]
    _, loss_value, acc_value, train_summary = sess_inception_resent.run(
        [train_op, loss, accuracy, merged], {input_data: contrast_images, label_data: labels, is_training: 1})
    _time = T.toc(average=False)

    print('iter {:d}, training loss: {:.3}, training acc: {:.3}, take {:.2}s'
          .format(i + 1, loss_value, acc_value, _time))

    # Training on fgsm on inception v3 adversarial examples
    T.tic()
    nontargeted_images = sess_inception_v3.run(
        x_adv, feed_dict={x_input: images})
    adv_contrast_images = add_contrast_on_batch(nontargeted_images)
    _, loss_adv_value, acc_adv_value, train_adv_summary = sess_inception_resent.run(
        [train_op, loss, accuracy, merged], {input_data: adv_contrast_images, label_data: labels, is_training: 1})
    _time = T.toc(average=False)

    print('iter {:d}, adv training loss: {:.3}, adv training acc: {:.3}, take {:.2}s'
          .format(i + 1, loss_adv_value, acc_adv_value, _time))

    if (i + 1) % 25 == 0:
        T.tic()
        val_images, val_labels = queue_out.get()
        # val_images = ndimage.gaussian_filter(
        #     val_images, sigma=(0, 1, 1, 0), order=0, truncate=2.0)
        val_contrast_images = add_contrast_on_batch(val_images)
        val_labels = [
            synset_to_ind[imdb.classes[int(item)]] for item in val_labels]
        val_loss_value, val_acc_value, val_summary = sess_inception_resent.run(
            [loss, accuracy, merged], {input_data: val_contrast_images, label_data: val_labels, is_training: 0})
        _val_time = T.toc(average=False)
        print('###validation loss: {:.3}, validation acc: {:.3}, take {:.2}s'
              .format(val_loss_value, val_acc_value, _val_time))

        nontargeted_val_images = sess_inception_v3.run(
            x_adv, feed_dict={x_input: val_images})
        val_adv_contrast_images = add_contrast_on_batch(nontargeted_val_images)
        val_adv_loss_value, val_adv_acc_value, val_adv_summary = sess_inception_resent.run(
            [loss, accuracy, merged], {input_data: val_adv_contrast_images, label_data: val_labels, is_training: 0})
        _val_time = T.toc(average=False)
        print('###adv validation loss: {:.3}, adv validation acc: {:.3}, take {:.2}s'
              .format(val_adv_loss_value, val_adv_acc_value, _val_time))
        queue_in.put(True)

        global_step = i + 1
        train_writer.add_summary(train_summary, global_step)
        train_adv_writer.add_summary(train_adv_summary, global_step)
        val_writer.add_summary(val_summary, global_step)
        val_adv_writer.add_summary(val_adv_summary, global_step)

    if ((i + 1) % 2500 == 0):
        save_path = cur_saver.save(sess_inception_resent, os.path.join(
            CKPTS_DIR,
            cfg.TRAIN_SNAPSHOT_PREFIX + '_iter_' + str(i + 1) + '.ckpt'))
        print("Model saved in file: %s" % save_path)

# terminate child processes
if cfg.MULTITHREAD:
    imdb.close_all_processes()
queue_in.cancel_join_thread()
queue_out.cancel_join_thread()
val_data_process.terminate()

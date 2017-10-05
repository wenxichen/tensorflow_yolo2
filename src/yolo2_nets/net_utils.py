import os
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

import tensorflow as tf
import numpy as np
import config as cfg

slim = tf.contrib.slim


def get_ordered_ckpts(sess, imdb, net_name, save_epoch=True):
    """Get the ckpts for particular network on certain dataset.
    The ckpts is ordered in ascending order of time.

    Returns: sorted list of ckpt names.
    """

    # Find previous snapshots if there is any to restore from
    ckpts_dir = cfg.get_ckpts_dir(net_name, imdb.name)
    if save_epoch:
        save_interval = 'epoch'
    else:
        save_interval = 'iter'
    sfiles = os.path.join(ckpts_dir,
                          cfg.TRAIN_SNAPSHOT_PREFIX + '_' + save_interval + '_*.ckpt.meta')
    sfiles = glob.glob(sfiles)
    sfiles.sort(key=os.path.getmtime)
    # Get the snapshot name in TensorFlow
    sfiles = [ss.replace('.meta', '') for ss in sfiles]

    return sfiles

    # TODO: double check what this is doing
    # nfiles = os.path.join(ckpts_dir, cfg.TRAIN_SNAPSHOT_PREFIX + '_iter_*.pkl')
    # nfiles = glob.glob(nfiles)
    # nfiles.sort(key=os.path.getmtime)


def restore_darknet19_variables(sess, imdb, net_name, save_epoch=True):
    """Initialize or restore the varialbes in darknet19."""

    sfiles = get_ordered_ckpts(sess, imdb, net_name, save_epoch=save_epoch)
    lsf = len(sfiles)
    # TODO: add case when lsf is 0
    if lsf > 0:
        print 'Restorining model snapshots from {:s}'.format(sfiles[-1])
        saver = tf.train.Saver()
        saver.restore(sess, str(sfiles[-1]))
        print 'Restored.'

        fnames = sfiles[-1].split('_')
        return int(fnames[-1][:-5])


def restore_inception_resnet_variables_from_weight(sess, weights_path):

    adam_vars = [var for var in tf.global_variables()
                 if 'Adam' in var.name or
                 'beta1_power' in var.name or
                 'beta2_power' in var.name]
    uninit_vars = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope='InceptionResnetV2/Conv2d_1a_3x3') + adam_vars
    init_op = tf.variables_initializer(uninit_vars)

    variables_to_restore = slim.get_variables_to_restore(
        exclude=['InceptionResnetV2/Conv2d_1a_3x3'])
    for var in uninit_vars:
        if var in variables_to_restore:
            variables_to_restore.remove(var)
    saver = tf.train.Saver(variables_to_restore)

    print 'Initializing new variables to train from downloaded inception resnet weights'
    sess.run(init_op)
    saver.restore(sess, weights_path)

    return 0


def restore_resnet_tf_variables(sess, imdb, net_name, retrain=False, detection=True,
                                save_epoch=True, new_optmizer=None):
    """Get the tensorflow variables needed for the network.
    Initialize variables or read from weights file if there is no checkpoint to restore,
    otherise restore from the latest checkpoint.

    Args:
      sess: Tensorflow session
      net_name: name of the network
      retrain: True to retrain the variables starting from the pretrained weights
      detection: True if the weights are for image detection not image classification
      save_epoch: True if the ckpts are saved in terms of epoch numbers
      new_optimizer: None if optimizer stay the same as in the ckpts, 
                     otherwise, set to the name of the new optimizer (String)

    Returns: last saved iteration number
    """

    sfiles = get_ordered_ckpts(
        sess, imdb, net_name, save_epoch=save_epoch)
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
        adam_vars = [var for var in tf.global_variables()
                     if 'Adam' in var.name or
                        'beta1_power' in var.name or
                        'beta2_power' in var.name]
        if detection:
            uninit_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='yolo_fc1') \
                + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='yolo_fc2') \
                + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='loss_layer') \
                + adam_vars
        else:
            uninit_vars = adam_vars
        init_op = tf.variables_initializer(uninit_vars)

        # Restore only the convolutional layers:
        variables_to_restore = slim.get_variables_to_restore(
            exclude=['yolo_fc1', 'yolo_fc2', 'loss_layer'])
        for var in uninit_vars:
            if var in variables_to_restore:
                variables_to_restore.remove(var)
        saver = tf.train.Saver(variables_to_restore)

        print 'Initializing new variables to train from downloaded resnet50 weights'
        sess.run(init_op)
        saver.restore(sess, os.path.join(
            cfg.WEIGHTS_PATH, 'resnet_v1_50.ckpt'))

        return 0

    else:
        variables_to_restore = slim.get_variables_to_restore()
        if new_optmizer is not None:
            print 'Initializing variables for the new optimizer'
            optimzer_vars = [var for var in tf.global_variables()
                             if new_optmizer in var.name]
            init_op = tf.variables_initializer(optimzer_vars)
            sess.run(init_op)

            for var in optimzer_vars:
                if var in variables_to_restore:
                    variables_to_restore.remove(var)
        print 'Restorining model snapshots from {:s}'.format(sfiles[-1])
        saver = tf.train.Saver(variables_to_restore)
        saver.restore(sess, str(sfiles[-1]))
        print 'Restored.'

        fnames = sfiles[-1].split('_')
        return int(fnames[-1][:-5])


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
        inter_square = intersection[:, :, :,
                                    :, 0] * intersection[:, :, :, :, 1]

        # calculate the boxs1 square and boxs2 square
        square1 = (boxes1[:, :, :, :, 2] - boxes1[:, :, :, :, 0]) * \
            (boxes1[:, :, :, :, 3] - boxes1[:, :, :, :, 1])
        square2 = (boxes2[:, :, :, :, 2] - boxes2[:, :, :, :, 0]) * \
            (boxes2[:, :, :, :, 3] - boxes2[:, :, :, :, 1])

        union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)

    return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)


def get_loss(net, labels, num_class, batch_size, image_size, S, B, OFFSET, scope='loss_layer'):
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
        predict_classes = net[:, :, :, :num_class]
        # confidence is defined as Pr(Object) * IOU
        predict_confidence = net[:, :, :, num_class:num_class + B]
        # predict_boxes has last dimenion has [x, y, w, h] * B
        # where (x, y) "represent the center of the box relative to the bounds of the grid cell"
        predict_boxes = tf.reshape(
            net[:, :, :, num_class + B:], [batch_size, S, S, B, 4])

        ########################
        # calculate class loss #
        ########################
        responsible = tf.reshape(labels[:, :, :, 0], [
                                 batch_size, S, S, 1])  # [BATCH_SIZE, S, S]
        classes = labels[:, :, :, 5:]

        class_delta = responsible * \
            (predict_classes - classes)  # [:,S,S,NUM_CLASS]
        class_loss = tf.reduce_mean(tf.reduce_sum(
            tf.square(class_delta), axis=[1, 2, 3]), name='class_loss')

        #############################
        # calculate coordinate loss #
        #############################
        gt_boxes = tf.reshape(labels[:, :, :, 1:5], [batch_size, S, S, 1, 4])
        gt_boxes = tf.tile(gt_boxes, [1, 1, 1, B, 1]) / float(image_size)

        # add offsets to the predicted box coordinates
        # to get absolute coordinates between 0 and 1
        offset = tf.constant(OFFSET, dtype=tf.float32)
        offset = tf.reshape(offset, [1, S, S, B])
        offset = tf.tile(offset, [batch_size, 1, 1, 1])
        predict_xs = (predict_boxes[:, :, :, :, 0] + offset) / float(S)
        predict_ys = (predict_boxes[:, :, :, :, 1] +
                      tf.transpose(offset, (0, 2, 1, 3))) / float(S)
        predict_ws = tf.square(predict_boxes[:, :, :, :, 2])
        predict_hs = tf.square(predict_boxes[:, :, :, :, 3])
        predict_boxes_offset = tf.stack(
            [predict_xs, predict_ys, predict_ws, predict_hs], axis=4)
        # gt_boxes_offset = tf.stack([gt_xs, gt_ys, gt_ws, gt_hs], axis=4)

        # calculate IOUs
        ious = get_iou(predict_boxes_offset, gt_boxes)

        # calculate object masks and nonobject masks tensor [BATCH_SIZE, S, S, B]
        object_mask = tf.reduce_max(ious, 3, keep_dims=True)
        object_mask = tf.cast((ious >= object_mask), tf.float32) * responsible
        noobject_mask = tf.ones_like(
            object_mask, dtype=tf.float32) - object_mask

        # add offsets to the ground truth box coordinates
        # to get absolute coordinates between 0 and 1
        gt_rel_xs = gt_boxes[:, :, :, :, 0] * S - offset
        gt_rel_ys = gt_boxes[:, :, :, :, 1] * \
            S - tf.transpose(offset, (0, 2, 1, 3))
        gt_rel_ws = tf.sqrt(gt_boxes[:, :, :, :, 2])
        gt_rel_hs = tf.sqrt(gt_boxes[:, :, :, :, 3])

        # coordinate loss
        coord_mask = tf.expand_dims(object_mask, 4)
        boxes_delta_xs = predict_boxes[:, :, :, :, 0] - gt_rel_xs
        boxes_delta_ys = predict_boxes[:, :, :, :, 1] - gt_rel_ys
        # TODO: double check if it is ok to get rid of the sqrt here
        boxes_delta_ws = predict_boxes[:, :, :, :, 2] - gt_rel_ws
        boxes_delta_hs = predict_boxes[:, :, :, :, 3] - gt_rel_hs
        boxes_delta = tf.stack(
            [boxes_delta_xs, boxes_delta_ys, boxes_delta_ws, boxes_delta_hs], axis=4)
        boxes_delta = coord_mask * boxes_delta
        coord_loss = tf.reduce_mean(tf.reduce_sum(tf.square(boxes_delta), axis=[
                                    1, 2, 3, 4]), name='coord_loss') * cfg.LAMBDA_COORD

        #########################
        # calculate object loss #
        #########################
        # object loss
        object_delta = object_mask * (predict_confidence - ious)
        object_loss = tf.reduce_mean(tf.reduce_sum(
            tf.square(object_delta), axis=[1, 2, 3]), name='object_loss')
        # noobject loss
        noobject_delta = noobject_mask * predict_confidence
        noobject_loss = tf.reduce_mean(tf.reduce_sum(tf.square(noobject_delta), axis=[
                                       1, 2, 3]), name='noobject_loss') * cfg.LAMBDA_NOOBJ

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


def show_yolo_detection(image_path, predict_output, imdb, object_thresh=0.5):
    """Compute bounding boxes from the yolo detection network prediction.
    Show the image and draw the bounding boxes 
    with threshold higher than object_thresh."""

    im = np.array(Image.open(image_path), dtype=np.uint8)
    im_h, im_w, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    ax.imshow(im)

    num_class = imdb.num_class
    offset = cfg.YOLO_GRID_OFFSET
    S = cfg.S
    B = cfg.B

    # get different type of prediction info
    predicts = predict_output.reshape([S, S, num_class + B * 5])
    predict_classes = predicts[:, :, :num_class]
    predict_confidences = predicts[:, :, num_class:num_class + B]
    predict_boxes = np.reshape(
        predicts[:, :, num_class + B:], [S, S, B, 4])
    predict_objects = predict_confidences > object_thresh
    # predict_objects = np.repeat(predict_objects, 4, 2).reshape(S, S, B, 4)
    # predict_boxes = predict_objects * predict_boxes

    # get predicted bounding boxes
    predict_xs = (predict_boxes[:, :, :, 0] + offset) / float(S)
    predict_ys = (predict_boxes[:, :, :, 1] +
                  np.transpose(offset, (1, 0, 2))) / float(S)
    predict_ws = np.square(predict_boxes[:, :, :, 2])
    predict_hs = np.square(predict_boxes[:, :, :, 3])

    # draw predicted bounding boxes
    for c in range(S):
        for r in range(S):
            for i in range(B):
                if predict_objects[c, r, i]:
                    predict_x = int(predict_xs[c, r, i] * im_w)
                    predict_y = int(predict_ys[c, r, i] * im_h)
                    predict_w = int(predict_ws[c, r, i] * im_w)
                    predict_h = int(predict_hs[c, r, i] * im_h)
                    predict_class = np.argmax(predict_classes[c, r])
                    predict_confidence = predict_confidences[c, r, i]
                    upper_left_x = predict_x - predict_w / 2
                    upper_left_y = predict_y - predict_h / 2
                    print "predicted bounding boxes: ({:d}, {:d}), width:{:d}, height:{:d}"\
                        .format(upper_left_x, upper_left_y, predict_w, predict_h)
                    ax.add_patch(
                        patches.Rectangle(
                            (upper_left_x, upper_left_y),
                            predict_w,
                            predict_h,
                            linewidth=1,
                            edgecolor='r',
                            facecolor='none'
                        )
                    )
                    ax.text(upper_left_x,
                            upper_left_y,
                            imdb.classes[int(predict_class)] +
                            ":" + str(predict_confidence),
                            color='r')
    plt.show()

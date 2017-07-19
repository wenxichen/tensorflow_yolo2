import os
import glob
import config
import tensorflow as tf
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


def get_resnet_tf_variables(sess, imdb, net_name, retrain=False):
    """Get the tensorflow variables needed for the network.
    Initialize variables or read from weights file if there is no checkpoint to restore,
    otherise restore from the latest checkpoint.

    Args:
      sess: Tensorflow session
      net_name: name of the network
      retrain: True to retrain the variables starting from the pretrained weights

    Returns: last saved iteration number
    """

    sfiles = get_ordered_ckpts(sess, imdb, net_name)
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
        adam_vars = [var for var in tf.global_variables(
        ) if 'Adam' in var.name or 'beta1_power' in var.name or 'beta2_power' in var.name]
        uninit_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='yolo_fc1') \
            + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='yolo_fc2') \
            + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='loss_layer') \
            + adam_vars
        init_op = tf.variables_initializer(uninit_vars)

        # Restore only the convolutional layers:
        variables_to_restore = slim.get_variables_to_restore(
            exclude=['yolo_fc1', 'yolo_fc2', 'loss_layer'])
        for var in uninit_vars:
            if var in variables_to_restore:
                variables_to_restore.remove(var)
        saver = tf.train.Saver(variables_to_restore)

        print('Initializing new variables to train from downloaded resnet50 weights')
        sess.run(init_op)
        saver.restore(sess, os.path.join(
            cfg.WEIGHTS_PATH, 'resnet_v1_50.ckpt'))

        return 0

    else:
        print('Restorining model snapshots from {:s}'.format(sfiles[-1]))
        saver = tf.train.Saver()
        saver.restore(sess, str(sfiles[-1]))
        print('Restored.')

        fnames = sfiles[-1].split('_')
        return int(fnames[-1][:-5])

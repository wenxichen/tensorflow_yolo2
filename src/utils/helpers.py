"""Miscellaneous helpers for the library."""

import numpy as np

def get_length(l):
    """Get the length of l.
    l can be 1d list or array."""
    if isinstance(l, list):
        assert not isinstance(l[0], list)
        return len(l)
    elif isinstance(l, np.ndarray):
        assert l.ndim == 1
        return l.shape[0]
    else:
        raise Exception

def compare_label_values(preds, labels):
    """Compare the prediction with labels.
    Assume they are in the same order and both are 1d list or array.
    
    Return:
        count: an integer of number of correct predictions
        accracy: an float of accuracy
    """

    assert get_length(preds) == get_length(labels)
    n = get_length(preds)
    count = 0
    for i in range(n):
        if preds[i] == labels[i]:
            count+=1
    return count, float(count) / n


#################################
# Helpers for adversarial tasks #
#################################
def add_contrast_on_batch(images):
    """Add the 4 side contrast using add_4_side_contrast_mtr to the whole batch.
    TODO: not sure if this is the most efficient way.
    """
    batch_num = images.shape[0]
    contrast_images = np.zeros([batch_num, 299, 299, 15])
    for i in range(batch_num):
        contrast_images[i,:,:,:] = add_4_side_contrast_mtr(images[i,:,:,:])
    return contrast_images

def add_4_side_contrast_mtr(x):
    """Add the contrast of the 4 sides of each pixel value for each rgb channel.
    Thus, for each rgb channel, 4 new channels is created s.t. for pixle (r,c):
        1) absolute difference between (r,c) and (r-1,c)
        2) absolute difference between (r,c) and (r+1,c)
        3) absolute difference between (r,c) and (r,c-1)
        4) absolute difference between (r,c) and (r,c+1)
    """
    x_with_contrast = np.zeros([299, 299, 15])
    x_with_contrast[:,:,0:3] = x
    x_with_contrast[1:,:,3:6] = np.abs(x[1:,:,:] - x[:-1,:,:])
    x_with_contrast[:-1,:,6:9] = np.abs(x[:-1,:,:] - x[1:,:,:])
    x_with_contrast[:,1:,9:12] = np.abs(x[:,1:,:] - x[:,:-1,:])
    x_with_contrast[:,:-1,12:] = np.abs(x[:,:-1,:] - x[:,1:,:])
    return x_with_contrast
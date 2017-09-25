"""Miscellaneous helpers for the library."""

import numpy as np

def get_length(l):
    """Get the length of l.
    l can be 1d list or array."""
    if isinstance(l, list):
        assert isinstance(l[0], list)
        return len(l)
    else if isinstance(l, np.array):
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

from pdb import set_trace

import numpy as np


def sortrows(a):
    """
    :param a:
    :return: Compare each row in ascending order
    """
    return a[np.lexsort(np.rot90(a))]


def refine_cell(scale_pos, cell, numbers, symprec=4):
    # scale_pos = np.dot(pos, np.linalg.inv(cell))
    scale_pos, _ = np.round(np.modf(scale_pos), symprec)
    scale_pos[scale_pos < 0] = scale_pos[scale_pos < 0] + 1
    # set_trace()
    pos, index = np.unique(scale_pos, axis=0, return_index=True)
    numbers = numbers[index]
    # set_trace()
    return pos, numbers

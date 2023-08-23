from pdb import set_trace

import numpy as np


def sortrows(a):
    """
    :param a:
    :return: Compare each row in ascending order
    """
    return a[np.lexsort(np.rot90(a))]


def refine_cell(scale_pos, numbers, symprec=4):
    """refine the scale position between 0-1, and remove duplicates

    Args:
        scale_pos: scale position of the structure
        numbers: atom_type
        symprec: system precise

    Returns: scale position after refine, the correspond atom type

    """
    scale_pos, _ = np.round(np.modf(scale_pos), symprec)
    scale_pos[scale_pos < 0] = scale_pos[scale_pos < 0] + 1
    pos, index = np.unique(scale_pos, axis=0, return_index=True)
    numbers = numbers[index]
    return pos, numbers

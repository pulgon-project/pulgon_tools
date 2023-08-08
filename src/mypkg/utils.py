import numpy as np


def sortrows(a):
    """
    :param a:
    :return: Compare each row in ascending order
    """
    return a[np.lexsort(np.rot90(a))]

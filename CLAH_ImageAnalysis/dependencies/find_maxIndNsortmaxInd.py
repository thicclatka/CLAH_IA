import numpy as np


def find_maxIndNsortmaxInd(array_of_int: np.ndarray, axis: int = 1) -> tuple:
    """
    Find the indices of the maximum values along the specified axis and sort them in ascending order.

    Parameters:
    - array_of_int: numpy.ndarray
        The input array of integers.
    - axis: int, optional
        The axis along which to find the maximum values. Default is 1.

    Returns:
    - maxInd: numpy.ndarray
        The indices of the maximum values along the specified axis.
    - sortMaxInd: numpy.ndarray
        The sorted indices of the maximum values.

    Example usage:
    >>> array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> maxInd, sortMaxInd = find_maxIndNsortmaxInd(array)
    >>> print(maxInd)
    [2 2 2]
    >>> print(sortMaxInd)
    [0 1 2]
    """

    maxInd = np.argmax(array_of_int, axis=axis)
    sortMaxInd = np.argsort(maxInd)
    return maxInd, sortMaxInd

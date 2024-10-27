import numpy as np


def local_minima(
    x: np.ndarray, not_closer_than: float | None = None, less_than: float | None = None
) -> np.ndarray:
    """
    Find the local minima in a 1D array.

    Args:
        x (ndarray): The input 1D array.
        not_closer_than (float, optional): The minimum distance between two consecutive local minima. Defaults to None.
        less_than (float, optional): The maximum value for a local minimum. Defaults to None.

    Returns:
        ndarray: An array containing the indices of the local minima.

    Raises:
        ValueError: If x is not a 1D array.
    """
    if x.ndim > 1:
        raise ValueError("x should be a 1D array")

    if less_than is not None:
        indices = np.where(x < less_than)[0]
    else:
        indices = np.arange(len(x))

    if len(indices) == 0:
        return np.array([])

    x_below = x[indices]
    gapToLeft = np.where(np.diff(np.insert(indices, 0, 0)) > 1)[0]
    gapToRight = np.where(np.diff(np.append(indices, len(x))) > 1)[0]

    sDiff = np.sign(np.diff(x_below))
    left_sign = np.insert(sDiff, 0, 1)
    left_sign[gapToLeft] = -1
    right_sign = np.append(sDiff, -1)
    right_sign[gapToRight] = 1

    zeros = np.where(right_sign == 0)[0]
    for i in reversed(zeros):
        right_sign[i] = right_sign[i + 1]

    mins = indices[np.where((left_sign < 0) & (right_sign > 0))[0]]

    if not_closer_than is not None and len(mins) > 1:
        while True:
            too_close = np.where(np.diff(mins) < not_closer_than)[0]
            if len(too_close) == 0:
                break
            vals = np.array([x[mins[too_close]], x[mins[too_close + 1]]])
            offset = np.argmax(vals, axis=0)
            delete = too_close + offset
            mins = np.delete(mins, np.unique(delete))

    return mins

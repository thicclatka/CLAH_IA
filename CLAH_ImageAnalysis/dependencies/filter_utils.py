import numpy as np
import cv2
from skimage.morphology import disk


def create_exponential_decay_filter(
    tau: float, start: int = 1, tail_factor: int = 6
) -> np.ndarray:
    """
    Creates an exponential decay filter.

    Parameters:
    tau (float): The decay constant used in the exponential function.
    start (int): The starting point of the range for the filter. Default is 1.
    tail_factor (int): Factor to determine the length of the filter's tail. Default is 6.

    Returns:
    numpy.ndarray: An exponential decay filter.
    """
    x = np.arange(start, tail_factor * tau + start)
    yexp = np.exp(-x / tau)
    return yexp


def apply_morphology_tophat_filter(
    array_stack: np.ndarray, window_size: int
) -> np.ndarray:
    """
    Apply morphology tophat filter to array stack.

    Args:
        array_stack (np.ndarray): _description_
        window_size (int): _description_

    Returns:
        np.ndarray: _description_
    """

    kernel2use = disk(window_size)
    return cv2.morphologyEx(array_stack, cv2.MORPH_TOPHAT, kernel2use)


def apply_median_blur_filter(array_stack: np.ndarray, window_size: int) -> np.ndarray:
    """
    Apply median blur filter to array stack.
    """
    if array_stack.dtype != np.uint8:
        array_stack = cv2.normalize(array_stack, None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )
    return cv2.medianBlur(array_stack, window_size)

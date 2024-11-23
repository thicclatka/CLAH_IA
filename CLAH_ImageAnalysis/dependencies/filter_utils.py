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

    Parameters:
        array_stack (numpy.ndarray): Input array stack.
        window_size (int): Window size for the median blur filter.

    Returns:
        numpy.ndarray: Median blurred array stack.
    """
    return cv2.medianBlur(array_stack, window_size)


def apply_high_pass_filter(img: np.ndarray, gSig_filt: tuple = (2, 2)) -> np.ndarray:
    """
    Apply high-pass filter using Gaussian kernel subtraction.

    Parameters:
        img: np.ndarray
            Input image (2D array)
        gSig_filt: tuple
            Standard deviation for Gaussian kernel in (y, x)

    Returns:
        np.ndarray: Filtered image
    """
    # Create kernel size (3 times sigma, rounded to next odd number)
    ksize = tuple([(3 * i) // 2 * 2 + 1 for i in gSig_filt])

    # Create 2D Gaussian kernel
    kernel_x = cv2.getGaussianKernel(ksize[0], gSig_filt[0])
    kernel_y = cv2.getGaussianKernel(ksize[1], gSig_filt[1])
    kernel_2d = kernel_x.dot(kernel_y.T)

    nonzero = np.nonzero(kernel_2d >= kernel_2d[:, 0].max())
    zero = np.nonzero(kernel_2d < kernel_2d[:, 0].max())
    kernel_2d[nonzero] -= kernel_2d[nonzero].mean()
    kernel_2d[zero] = 0

    # # Normalize kernel to act as high-pass
    # kernel_2d = kernel_2d - kernel_2d.mean()

    # Apply filter
    filtered = cv2.filter2D(
        np.array(img, dtype=np.float32), -1, kernel_2d, borderType=cv2.BORDER_REFLECT
    )

    return filtered

import numpy as np
import cv2
from skimage.morphology import disk
from scipy.signal import convolve
from scipy.fft import fft2
from scipy.fft import fftshift
from scipy.fft import ifft2
from scipy.fft import ifftshift


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


def apply_filter2D(
    array2filter: np.ndarray,
    kernel: np.ndarray,
    ddepth: int = -1,
    borderType: int = cv2.BORDER_REFLECT,
) -> np.ndarray:
    """
    Apply a 2D filter to an array using OpenCV's filter2D function.

    Parameters:
        array (np.ndarray): Input array to be filtered
        kernel (np.ndarray): Filter kernel to apply
        ddepth (int, optional): OpenCV depth for the destination array.
            Defaults to -1, which means the same depth as the source.
        border_type (int, optional): OpenCV border type for handling edges.
            Defaults to cv2.BORDER_REFLECT.

    Returns:
        np.ndarray: Filtered array
    """
    return cv2.filter2D(
        src=array2filter, ddepth=ddepth, kernel=kernel, borderType=borderType
    )


def apply_convolution(
    array2convolve: np.ndarray,
    kernel: np.ndarray | None = None,
    window_size: int | None = None,
) -> np.ndarray:
    """
    Apply convolution to an array.

    Parameters:
        array2convolve (np.ndarray): Array to convolve.
        kernel (np.ndarray | None): Kernel to convolve with.
        window_size (int | None): Window size for Gaussian kernel. If None, a Gaussian kernel is created.

    Returns:
        np.ndarray: Convolved array.
    """
    if kernel is None and window_size is not None:
        kernel = create_kernel_from_window_size(window_size)
    elif kernel is None and window_size is None:
        raise ValueError("Either kernel or window_size must be provided.")

    return convolve(array2convolve, kernel, mode="same")


def create_kernel_from_window_size(window_size: int) -> np.ndarray:
    """
    Create a Gaussian kernel from a window size.
    """
    x = np.linspace(-(window_size - 1) / 2, (window_size - 1) / 2, window_size)
    sigma = window_size / (2 * np.sqrt(2 * np.log(2)))
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()  # normalize to sum to 1
    return kernel


def create_bandpass_filter(shape, low_cutoff_freq, high_cutoff_freq):
    """
    Create a bandpass filter for a given shape.

    Parameters:
        shape (tuple): Shape of the filter.
        low_cutoff_freq (float): Low cutoff frequency. Units are pixels^-1.
        high_cutoff_freq (float): High cutoff frequency. Units are pixels^-1.

    Returns:
        np.ndarray: Bandpass filter.
    """
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    low_cutoff = low_cutoff_freq * min(rows, cols)
    high_cutoff = high_cutoff_freq * min(rows, cols)
    y, x = np.ogrid[:rows, :cols]
    distance = np.sqrt((y - crow) ** 2 + (x - ccol) ** 2)
    mask = (distance >= low_cutoff) & (distance <= high_cutoff)
    return mask


def apply_bandpass_filter(image, low_cutoff_freq, high_cutoff_freq):
    """
    Apply a bandpass filter to an image.

    Parameters:
        image (np.ndarray): Image to filter.
        low_cutoff_freq (float): Low cutoff frequency. Units are pixels^-1.
        high_cutoff_freq (float): High cutoff frequency. Units are pixels^-1.

    Returns:
        np.ndarray: Filtered image.
    """
    f = fft2(image)
    fshift = fftshift(f)
    mask = create_bandpass_filter(image.shape, low_cutoff_freq, high_cutoff_freq)
    fshift_filtered = fshift * mask
    f_ishift = ifftshift(fshift_filtered)
    filtered_image = np.real(ifft2(f_ishift))
    return filtered_image

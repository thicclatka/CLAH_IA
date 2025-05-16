import cv2
import numpy as np
from rich import print as rprint
from skimage.util import img_as_ubyte, img_as_uint
from sklearn.preprocessing import MinMaxScaler

from CLAH_ImageAnalysis.utils import color_dict


def normalize_array(
    array: np.ndarray,
    dtype: np.dtype = np.uint8,
    alpha: int = 0,
    beta: int | None = None,
    normalize_type: int = cv2.NORM_MINMAX,
) -> np.ndarray:
    """
    Normalize an array to a specified data type and range.

    Parameters:
        array (np.ndarray): Input array to normalize
        dtype (np.dtype): Target data type for output array. Defaults to np.uint8.
        alpha (int): Lower bound of normalization range. Defaults to 0.
        beta (int | None): Upper bound of normalization range. If None, uses maximum value
            possible for dtype. Defaults to None.
        normalize_type (int): OpenCV normalization type. Defaults to cv2.NORM_MINMAX.

    Returns:
        np.ndarray: Normalized array converted to specified dtype
    """
    if beta is None:
        beta = np.iinfo(dtype).max
    return cv2.normalize(
        src=array,
        dst=None,
        alpha=alpha,
        beta=beta,
        norm_type=normalize_type,
    ).astype(dtype)


def apply_CLAHE(
    array: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)
) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to an array.

    Parameters:
        array (np.ndarray): Input array to apply CLAHE to
        clip_limit (float): Threshold for contrast limiting. Higher values give more contrast.
            Defaults to 2.0.
        tile_grid_size (tuple): Size of grid for histogram equalization.
            Defaults to (8, 8).

    Returns:
        np.ndarray: Array with CLAHE applied
    """
    return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size).apply(
        array
    )


def apply_colormap(array: np.ndarray, colormap: int = cv2.COLORMAP_TURBO) -> np.ndarray:
    """
    Apply a colormap to an array.

    Parameters:
        array (np.ndarray): Input array to apply colormap to
        colormap (int): OpenCV colormap to apply. Defaults to cv2.COLORMAP_PLASMA.
            See https://docs.opencv.org/4.x/d3/d50/group__imgproc__colormap.html for all available colormaps.

    Returns:
        np.ndarray: Array with colormap applied
    """
    return cv2.applyColorMap(array, colormap)


def convert_bit_range(array: np.ndarray, bit_range: int = 8) -> np.ndarray:
    """
    Convert an array to a specified bit range.

    Parameters:
        array (np.ndarray): Input array to convert
        bit_range (int): Bit range to convert to. Defaults to 8.

    Returns:
        np.ndarray: Array converted to specified bit range
    """
    if bit_range <= 0:
        raise ValueError(f"Bit range must be positive, got {bit_range}")

    # Special cases for 8 and 16 bit using scikit-image functions
    if bit_range == 8:
        return img_as_ubyte(array)
    elif bit_range == 16:
        return img_as_uint(array)

    # For other bit ranges, scale the array manually
    max_val = (2**bit_range) - 1
    array_min = array.min()
    array_max = array.max()

    # Avoid division by zero
    if array_max == array_min:
        return np.full_like(array, max_val if array_max > 0 else 0)

    # Scale to [0, 1] then to target range
    scaled = (array - array_min) / (array_max - array_min)
    converted = (scaled * max_val).round()

    # Choose appropriate dtype based on bit range
    if bit_range <= 8:
        dtype = np.uint8
    elif bit_range <= 16:
        dtype = np.uint16
    elif bit_range <= 32:
        dtype = np.uint32
    else:
        dtype = np.uint64

    return converted.astype(dtype)


def zScore_signal(
    signal: np.ndarray, fps: int, window_seconds: float = 7.0
) -> np.ndarray:
    """
    Z-score a signal.

    Parameters:
        signal (np.ndarray): Input signal to z-score
        fps (int): Frames per second
        window_seconds (float): Size of rolling window in seconds (default: 7.0)

    Returns:
        np.ndarray: Z-scored signal
    """

    # Convert window from seconds to frames (at 10 Hz)
    window = int((fps * window_seconds) / 2)  # e.g., 70 frames for 7 seconds

    rolling_baseline = np.zeros_like(signal)
    rolling_std = np.zeros_like(signal)

    for i in range(len(signal)):
        start_idx = max(0, i - window)
        end_idx = min(len(signal), i + window)
        window_data = signal[start_idx:end_idx]

        rolling_baseline[i] = np.percentile(window_data, 20)
        rolling_std[i] = np.std(window_data)

    # Avoid division by zero
    rolling_std[rolling_std == 0] = np.nanmean(rolling_std[rolling_std != 0])

    # Z-score the signal
    signal_Z = (signal - rolling_baseline) / rolling_std

    return signal_Z


def feature_normalization(
    array: np.ndarray,
) -> np.ndarray:
    """
    Normalize an array to a range of 0 to 1.

    Parameters:
        array (np.ndarray): Input array to normalize. Should be a 2D array, where each row is a sample and each column is a feature. (samples x features)

    Returns:
        np.ndarray: Normalized array. Same shape as input array.
    """
    color_lib = color_dict()

    if array.ndim != 2:
        raise ValueError(
            "Input array must be a 2D array, where each row is a sample and each column is a feature. (samples x features)"
        )

    scaler = MinMaxScaler()
    normalized_array = scaler.fit_transform(array)

    if np.any(np.isnan(normalized_array)):
        rprint(
            f"[bold {color_lib['red']}]Warning:[/bold {color_lib['red']}] NaN values in normalized array. Replacing with 0."
        )
        normalized_array = np.nan_to_num(normalized_array, nan=0.0)

    return normalized_array


def normalize_array_MINMAX(array: np.ndarray, adjustment: float = 0.0) -> np.ndarray:
    """
    Normalize an array to a range of 0 to 1.
    """
    return (array - np.min(array)) / (np.max(array) - np.min(array)) + adjustment

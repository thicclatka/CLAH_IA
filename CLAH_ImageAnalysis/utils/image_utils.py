import numpy as np
import cv2
import os
from skimage.transform import resize
from CLAH_ImageAnalysis.utils import fig_tools
from CLAH_ImageAnalysis.utils import text_dict
from scipy.interpolate import interp1d


def read_image(path_to_image: str) -> np.ndarray:
    """
    Read an image from a file.

    Parameters:
        path_to_image (str): The path to the image file.

    Returns:
        np.ndarray: The image as a numpy array.
    """
    import rasterio
    from rasterio.errors import NotGeoreferencedWarning
    import warnings

    assert isinstance(path_to_image, str), "Path must be a string"
    assert path_to_image.strip(), "Path cannot be empty"
    assert os.path.exists(path_to_image), f"Image file not found: {path_to_image}"

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
        with rasterio.open(path_to_image) as src:
            img = src.read(1)
    return img


def apply_gamma_correction(image: np.ndarray, gamma: float = 0.5) -> np.ndarray:
    """
    Apply gamma correction to enhance brightness in darker areas.

    Parameters:
        image (numpy.ndarray): The input image.
        gamma (float): The gamma value for the correction. Default is 0.5.

    Returns:
        numpy.ndarray: The gamma-corrected image.
    """
    return np.power(image, gamma)


def equalize_histogram(image: np.ndarray) -> np.ndarray:
    """
    Equalize the histogram of the combined image for better contrast.

    Parameters:
        image (numpy.ndarray): The input image to be equalized.

    Returns:
        numpy.ndarray: The equalized image.
    """

    from skimage import exposure

    return exposure.equalize_adapthist(image)


def blur_image(image: np.ndarray, sigma: float = 1) -> np.ndarray:
    """
    Apply Gaussian blur to the input image.

    Parameters:
        image (numpy.ndarray): The input image.
        sigma (float): The standard deviation of the Gaussian kernel. Default is 1.

    Returns:
        numpy.ndarray: The blurred image.
    """
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(image, sigma=sigma)


def normalize_image(image: np.ndarray, output_dtype: type = np.uint8) -> np.ndarray:
    """
    Normalize an image to a specific dtype. Convert to uint8, uint16, or keep
    as float32/float64 with values between 0 and 1.

    Parameters:
        image (np.ndarray): Input image array.
        output_dtype (type): Desired output data type (e.g., np.uint8, np.uint16,
                                np.float32, np.float64).

    Returns:
        np.ndarray: Normalized image.

    Raises:
        ValueError: If output_dtype is not supported.
    """
    allowed_types = [np.uint8, np.uint16, np.float32, np.float64]
    if output_dtype not in allowed_types:
        raise ValueError(
            f"output_dtype must be one of {allowed_types}, got {output_dtype}"
        )

    image_max = image.max()
    if image_max == 0:
        return np.zeros_like(image, dtype=output_dtype)

    normalized_image = image.astype(np.float64) / image_max

    if output_dtype in [np.float32, np.float64]:
        return normalized_image.astype(output_dtype)

    max_val = np.iinfo(output_dtype).max
    scaled_image = (normalized_image * max_val).astype(output_dtype)

    return scaled_image


def create_combined_arr_wOverlap(
    array_list: list[np.ndarray],
    overlap_threshold: float,
    overlap_enhance_factor: float,
) -> np.ndarray:
    """
    Create a combined array with overlap enhancement.

    Parameters:
        array_list (list[np.ndarray]): A list of numpy arrays to combine.
        overlap_threshold (float): The threshold for overlap detection.
        overlap_enhance_factor (float): The factor to enhance overlap.

    Returns:
        np.ndarray: The combined array with overlap enhancement.
    """

    RBG = ([1, 0, 0], [0, 0, 1], [0, 1, 0])
    if isinstance(array_list, list):
        height, width = array_list[0].shape
    elif isinstance(array_list, np.ndarray):
        height, width = array_list.shape

    combined = np.zeros((height, width, 3))
    for i, arr in enumerate(array_list):
        color = fig_tools.hex_to_rgba(RBG[i % 3])[:3]  # ignore alpha
        # extend array to operate over RBG
        colorized_arr = np.stack([arr * color[c] for c in range(3)], axis=-1)
        combined += colorized_arr
    # Detect overlaps, where at least 2 channels have non-zero values
    overlap_mask = combined.sum(axis=-1) - combined.max(axis=-1) > overlap_threshold
    # enhance overlap
    for c in range(3):
        combined[:, :, c][overlap_mask] *= overlap_enhance_factor
    # apply gamma correction
    combined = np.clip(apply_gamma_correction(combined), 0, 1)
    # histogram equalization
    combined = equalize_histogram(combined)
    return combined


def get_DSImage_filename(wCMAP: bool = False, wCH: str | None = None) -> str:
    """
    Get the filename for a downsampled image.

    Returns:
        str: The filename for a downsampled image.
    """

    file_tag = text_dict()["file_tag"]

    determine_wCH = f"_{wCH}" if wCH is not None else ""

    main_tag = file_tag["AVGCA"] + file_tag["TEMPFILT"] + file_tag["DOWNSAMPLE"]
    if wCMAP:
        cmap_tag = file_tag["CMAP"] + file_tag["8BIT"]
    else:
        cmap_tag = ""
    img_tag = file_tag["IMG"]

    return main_tag + cmap_tag + determine_wCH + img_tag


def resize_to_square(image: np.ndarray, round_to: int = None) -> np.ndarray:
    """
    Resize an image to a square shape based on smallest dimension via cv2 interpolation.
    The size is rounded down to the nearest multiple of round_to.

    Parameters:
        image (np.ndarray): Input image array.
        round_to (int): Round to nearest multiple of this number (e.g., 50 or 100).
        Default is None, which does not round.

    Returns:
        np.ndarray: Resized image array.
    """
    # Get the minimum dimension
    size = min(image.shape)
    # Round down to nearest multiple of round_to
    if round_to is not None:
        size = (size // round_to) * round_to

    dims = [size, size]

    # determine downsampling factor
    ds_factor = determine_DS_factor(dims)
    dims = tuple(sz // ds_factor for sz in dims)

    return resize_to_specific_dims(image, dims, interpolation=cv2.INTER_AREA)


def resize_to_specific_dims(
    image: np.ndarray, dims: tuple[int, int], interpolation=cv2.INTER_AREA
) -> np.ndarray:
    """
    Resize an image to specific dimensions using cv2 interpolation.

    Parameters:
        image (np.ndarray): Input image array.
        dims (tuple[int, int]): Desired dimensions (height, width).
        interpolation (cv2.InterpolationFlags): Interpolation method.
            Default is cv2.INTER_AREA.

    Returns:
        np.ndarray: Resized image array.
    """
    return cv2.resize(image, dims, interpolation=interpolation)


def determine_DS_factor(dims: tuple[int, int]) -> int:
    """
    Determine the downsampling factor based on the dimensions.
    Based on the bounds of the dimensions of the image. The bounds are as follows:
    a dimension of 400-650 will be downsampled by a factor of 2, 651-900 by 3,
    901-1200 by 4, and 1201-1500 by 5.

    Parameters:
        dims (tuple[int, int]): Dimensions of the image (height, width).

    Returns:
        int: The downsampling factor.
    """
    bounds = [(400, 650, 2), (651, 900, 3), (901, 1200, 4), (1201, 1500, 5)]

    for min_dim, max_dim, DS_factor in bounds:
        if any(min_dim <= d <= max_dim for d in dims):
            return DS_factor

    # if no bounds are met, return 1
    return 1


def extract_LRTB_from_crop_coords(crop_coords: list) -> tuple[int, int, int, int]:
    """
    Extract the left, right, top, bottom from the crop coordinates.

    Parameters:
        crop_coords (list): The crop coordinates [[x1, y1], [x2, y2]].

    Returns:
        tuple[int, int, int, int]: The left, right, top, bottom.
    """
    x1, y1 = crop_coords[0]
    x2, y2 = crop_coords[1]
    left = min(x1, x2)
    right = max(x1, x2)
    top = min(y1, y2)
    bottom = max(y1, y2)
    return left, top, right, bottom


def find_near_square_dims(length: int) -> tuple[int, int]:
    """
    Find dimensions closest to a square that divide the length exactly

    Parameters:
        length: Length of the 1D array

    Returns:
        tuple[int, int]: (height, width) dimensions that divide length exactly,
                        chosen to be as close to square as possible
    """
    # Get all divisors
    divisors = []
    for i in range(1, int(np.sqrt(length)) + 1):
        if length % i == 0:
            divisors.append(i)
            if i != length // i:  # Avoid duplicating perfect square divisors
                divisors.append(length // i)

    # Sort divisors
    divisors = sorted(divisors)

    # Find the pair of divisors closest to sqrt(length)
    best_h = 1
    best_w = length
    min_diff = length

    for d in divisors:
        w = length // d
        # Calculate how far this pair is from being square
        diff = abs(w - d)
        if diff < min_diff:
            min_diff = diff
            best_h = d
            best_w = w

    return best_h, best_w


def get_frame_timestamp_from_movie(movie_path: str) -> tuple[np.ndarray, float]:
    """
    Get the frame timestamp from a movie file.
    """

    cap = cv2.VideoCapture(movie_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_times = np.zeros(total_frames)
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        frame_times[frame_idx] = timestamp
        frame_idx += 1

    cap.release()
    return frame_times, fps


def sync_N_downsample_timestamps(
    high_freq_timestamps: np.ndarray,
    low_freq_timestamps: np.ndarray,
    sync_pulses: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sync and downsample timestamps from high-frequency to low-frequency.

    Parameters:
        high_freq_timestamps: Array of high frequency timestamps (e.g., 30Hz)
        low_freq_timestamps: Array of low frequency timestamps (e.g., 20Hz)
        sync_pulses: Array of sync pulse timestamps (e.g., 10Hz). Determines method of downsampling. If None, closest match timestamps are used. If provided, timestamps are interpolated between sync pulses.

    Returns:
        tuple[np.ndarray, np.ndarray]: (frame_mapping, time_mapping)
        frame_mapping: Array of frame indices in high_freq_timestamps that correspond to the low_freq_timestamps
        time_mapping: Array of timestamps in high_freq_timestamps that correspond to the low_freq_timestamps
    """
    # Find the closest timestamps in low_freq_timestamps to sync_pulses
    frame_mapping = []
    time_mapping = []

    if sync_pulses is not None:
        for i in range(len(sync_pulses) - 1):
            start_time = sync_pulses[i]
            end_time = sync_pulses[i + 1]

            mask_high_freq = (high_freq_timestamps >= start_time) & (
                high_freq_timestamps <= end_time
            )
            mask_low_freq = (low_freq_timestamps >= start_time) & (
                low_freq_timestamps <= end_time
            )

            high_indices = np.where(mask_high_freq)[0]
            low_indices = np.where(mask_low_freq)[0]

            interp_func = interp1d(
                high_freq_timestamps[high_indices],
                high_indices,
                kind="linear",
                fill_value="extrapolate",
            )
            # Map low frequency timestamps to high frequency indices
            for low_idx, low_time in zip(
                low_indices, low_freq_timestamps[mask_low_freq]
            ):
                # Get the interpolated high frequency index
                high_idx = interp_func(low_time)

                # Store the mapping
                frame_mapping.append(high_idx)
                time_mapping.append(high_freq_timestamps[int(round(high_idx))])
    else:
        # Interpolate with no sync pulses
        frame_mapping, time_mapping = closest_match_timestamps(
            high_freq_timestamps, low_freq_timestamps
        )

    frame_mapping = np.array(frame_mapping)
    time_mapping = np.array(time_mapping)

    return frame_mapping, time_mapping


def closest_match_timestamps(
    high_freq_times: np.ndarray, low_freq_times: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find closest high frequency timestamps for each low frequency timestamp.

    Parameters:
    - high_freq_times: Array of high frequency timestamps (e.g., 30Hz)
    - low_freq_times: Array of low frequency timestamps (e.g., 20Hz)

    Returns:
    - closest_indices: Indices of closest high frequency timestamps
    - closest_times: The actual closest high frequency timestamps
    """
    # Initialize arrays for results
    closest_indices = np.zeros(len(low_freq_times), dtype=int)
    closest_times = np.zeros(len(low_freq_times))

    # For each low frequency timestamp
    for i, low_time in enumerate(low_freq_times):
        # Find absolute differences
        diffs = np.abs(high_freq_times - low_time)
        # Get index of minimum difference
        closest_idx = np.argmin(diffs)
        # Store results
        closest_indices[i] = closest_idx
        closest_times[i] = high_freq_times[closest_idx]

    return closest_indices, closest_times

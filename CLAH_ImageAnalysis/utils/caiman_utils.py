"""
This script serves as a handler for various functions implemented in the CaImAn
(Calcium Imaging Analysis) package. It provides a structured approach to utilize
CaImAn's functionalities, streamlining processes such as motion correction,
image processing using the sliding window technique, and application of
Discrete Fourier Transform (DFT) based shifts.

The CaImAn package, which this script heavily relies on, is an open-source tool
specifically designed for calcium imaging data analysis. By leveraging CaImAn's
capabilities, this script facilitates efficient and robust image analysis workflows.

License Acknowledgment:
This script acknowledges and respects the licenses under which the CaImAn
package has been released. Users of this script are also advised to adhere to
the terms and conditions specified in CaImAn's licenses when using its
functionality either directly or indirectly.

CaImAn GitHub repository: https://github.com/flatironinstitute/CaImAn
"""

import os
import numpy as np
import glob
import caiman as cm
from CLAH_ImageAnalysis.utils import text_dict

text_lib = text_dict()
file_tag = text_lib["file_tag"]

######################################################
# Cluster Management Functions
######################################################
# Functions related to the setup and management of parallel processing clusters.


def start_cluster(
    N_PROC: int | None = None,
) -> tuple[object, object, int]:
    """Starts a local cluster for parallel processing.

    Parameters:
        N_PROC (int, optional): Number of processes to use. Defaults to None, which means all available cores will be used.

    Returns:
        c (object): Caiman cluster object.
        dview (object): Caiman distributed view object.
        n_processes (int): Number of processes used.
    """

    c, dview, n_processes = cm.cluster.setup_cluster(
        backend="local",
        n_processes=N_PROC,
        single_thread=False,
    )
    return c, dview, n_processes


def stop_cluster(dview: object, remove_log: bool = False) -> None:
    """Stops the Caiman cluster.

    Parameters:
        dview (object): Caiman distributed view object.
        remove_log (bool, optional): Whether to remove log files. Defaults to False.
    """
    cm.stop_server(dview=dview)

    # remove log files
    if remove_log:
        log_files = glob.glob("*_LOG_*")
        for log_file in log_files:
            os.remove(log_file)


######################################################
# Memory-Mapped File Functions
######################################################
# Functions for handling memory-mapped files, which are crucial for
# processing large datasets efficiently.


def load_mmap(
    fname_mmap: str | list, reshape_post_moco: bool = False
) -> tuple[np.ndarray, tuple, int]:
    """Loads a memory-mapped file.

    Parameters:
        fname_mmap (str or list): File name(s) of the memory-mapped file(s).
        reshape_post_moco (bool, optional): Whether to reshape the array after motion correction. Defaults to False.

    Returns:
        Yr (numpy.ndarray): Memory-mapped array.
        dims (tuple): Dimensions of the array.
        T (int): Number of frames.
    """

    # fname of mmap is output as a list by cm
    if isinstance(fname_mmap, list):
        fname_mmap = fname_mmap[0]

    Yr, dims, T = cm.load_memmap(fname_mmap)
    if reshape_post_moco:
        dy, dx = dims
        # reshape to frames x Y x X
        array = _reshape_array_post_load(Yr, dims, T)
        return array, dims, dy, dx
    else:
        return Yr, dims, T


def _reshape_array_post_load(
    Yr: np.ndarray, dims: tuple, T: int, order: str = "C"
) -> np.ndarray:
    """Reshapes the array after loading from memory-mapped file.

    Parameters:
        Yr (numpy.ndarray): Memory-mapped array.
        dims (tuple): Dimensions of the array.
        T (int): Number of frames.
        order (str, optional): Order of reshaping. Defaults to "C".

    Returns:
        array (numpy.ndarray): Reshaped array.
    """

    array = np.reshape(Yr.T, [T] + list(dims), order=order)
    return array


def save_mmap(
    fname2save: str,
    base_name: str = file_tag["MMAP_BASE"],
    order: str = "C",
    border_to_0: int = 1,
    chan_num: int = -1,
):
    """Saves an array as a memory-mapped file.

    Parameters:
        fname2save (str): File name to save the memory-mapped file.
        base_name (str, optional): Base name for the memory-mapped file. Defaults to "memmap_".
        order (str, optional): Order of the array. Defaults to "C".
        border_to_0 (int, optional): Border value to set to 0. Defaults to 1.
        chan_num (int, optional): Channel number. Defaults to -1.

    Returns:
        fname_mmap_saved (str): File name of the saved memory-mapped file.
    """

    if chan_num > -1:
        base_name = f"{base_name}_Ch{chan_num + 1}_"
    fname_mmap_saved = cm.save_memmap(
        [fname2save], base_name=base_name, order=order, border_to_0=border_to_0
    )
    return fname_mmap_saved


######################################################
# Motion Correction Utilities
######################################################
# These functions are specific to motion correction, including
# applying shifts using DFT and sliding window operations.


def upsampled_dft(
    data: np.ndarray,
    upsampled_region_size: tuple,
    upsample_factor: int = 1,
    axis_offsets: tuple | None = None,
) -> np.ndarray:
    """Performs upsampled DFT on the data.

    Parameters:
        data (numpy.ndarray): Input data.
        upsampled_region_size (tuple): Size of the upsampled region.
        upsample_factor (int, optional): Upsample factor. Defaults to 1.
        axis_offsets (tuple, optional): Axis offsets. Defaults to None.

    Returns:
        output (numpy.ndarray): Upsampled DFT output.
    """
    output = cm.motion_correction._upsampled_dft(
        data, upsampled_region_size, upsample_factor, axis_offsets
    )
    return output


def sliding_window(
    image: np.ndarray, overlaps: tuple, strides: tuple
) -> tuple[list, list, list, list, list]:
    """Performs sliding window on the image.

    Parameters:
        image (numpy.ndarray): Input image.
        overlaps (tuple): Overlaps between patches.
        strides (tuple): Strides between patches.

    Returns:
        dim_1s (list): List of dimension 1 of the sliding windows.
        dim_2s (list): List of dimension 2 of the sliding windows.
        xs (list): List of X coordinates of the patches.
        ys (list): List of Y coordinates of the patches.
        patches (list): List of patches of the image.
    """

    grid_coords, steps, patches = [], [], []
    for result in cm.motion_correction.sliding_window(image, overlaps, strides):
        dim_1, dim_2, x, y, patch = result
        grid_coords.append((dim_1, dim_2))
        steps.append((x, y))
        # xs.append(x)
        # ys.append(y)
        patches.append(patch)
    return grid_coords, steps, patches


def apply_shifts_dft(
    imgs: list,
    total_shifts: list,
    total_diffs_phase: list,
    is_freq: bool = False,
    border_nan: bool = True,
) -> list:
    """Applies shifts to the images using DFT.

    Parameters:
        imgs (list): List of input images.
        total_shifts (list): List of shifts for each image.
        total_diffs_phase (list): List of phase differences for each image.
        is_freq (bool, optional): Whether the images are in frequency domain. Defaults to False.
        border_nan (bool, optional): Whether to set the border to NaN. Defaults to True.

    Returns:
        output (list): List of shifted images.
    """

    output = [
        cm.motion_correction.apply_shifts_dft(
            im, (sh[0], sh[1]), dffphs, is_freq=is_freq, border_nan=border_nan
        )
        for im, sh, dffphs in zip(imgs, total_shifts, total_diffs_phase)
    ]
    return output


def create_weight_matrix_for_blending(
    img: np.ndarray, overlaps: tuple, strides: tuple
) -> np.ndarray:
    """Creates a weight matrix for blending.

    Parameters:
        img (numpy.ndarray): Input image.
        overlaps (tuple): Overlaps between patches.
        strides (tuple): Strides between patches.

    Returns:
        numpy.ndarray: Weight matrix for blending.
    """

    return cm.motion_correction.create_weight_matrix_for_blending(
        img, overlaps, strides
    )


def apply_high_pass_filter_space(img: np.ndarray, gSig_filt: tuple) -> np.ndarray:
    """Applies high-pass filter to the image.

    Parameters:
        img (numpy.ndarray): Input image.
        gSig_filt (tuple): Filter size for high-pass filter.

    Returns:
        numpy.ndarray: High-pass filtered image.
    """
    return cm.motion_correction.high_pass_filter_space(img, gSig_filt)

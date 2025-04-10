import isx
import numpy as np


def export_gpio_set_to_csv(gpio_file: str, gpio_fname_csv: str) -> None:
    """
    Export the GPIO set to a CSV file.

    Parameters:
        gpio_file (str): The path to the gpio file.
        gpio_fname_csv (str): The path to the csv file to export the gpio set to.
    """
    isx.export_gpio_set_to_csv(gpio_file, gpio_fname_csv)


def get_timestamps_from_isxd(isxd_file: str) -> np.ndarray:
    """
    Get the timestamps from an isxd file.

    Parameters:
        isxd_file (str): The path to the isxd file.

    Returns:
        timestamps (np.ndarray): The timestamps from the isxd file. Array is zeroed and in seconds.
    """
    isxd_data = isx.Movie.read(isxd_file)
    timestamps = [isxd_data.timestamps[i] for i in range(isxd_data.timing.num_samples)]

    # zero the timestamps
    timestamps = timestamps - timestamps[0]
    # convert to seconds
    timestamps = timestamps / 1e6

    timestamps = np.array(timestamps)
    return timestamps

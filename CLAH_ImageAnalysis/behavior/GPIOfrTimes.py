from enum import Enum
from CLAH_ImageAnalysis import utils
import pandas as pd
import numpy as np


class GPIOfrTimes_Txt(Enum):
    """
    Enum class for GPIO frame times constant strings.
    """

    GPIO_1 = "GPIO-1"
    GPIO_2 = "GPIO-2"
    FRAME = "BNC Sync Output"
    EX_LED = "EX-LED"
    OG_LED = "OG-LED"

    TIME_KEY = "Time (s)"
    CHAN_NAME = "Channel Name"
    VAL_KEY = "Value"

    FR_KEY = "FrTimes"
    SYNC_KEY = "syncTimes"
    CUE_KEY = "cueTimes"
    EXLED_KEY = "exledTimes"


def getGPIOfrTimes(folder_path: list | str = [], downsample: int = 1):
    """
    Get the GPIO frame times from the CSV file (_gpio.csv) in the specified folder_path.
    """
    folder_path = utils.path_selector(folder_path)
    gpio_fname, gpio_table = importGPIO(folder_path)
    FRdict = parseGPIO(gpio_table, downsample)

    return FRdict, gpio_fname


def importGPIO(folder_path: list | str = []) -> tuple[str, pd.DataFrame]:
    """
    Import the GPIO CSV file (_gpio.csv) from the specified folder_path.

    Parameters:
        folder_path: The folder_path to the folder containing the GPIO CSV file.

    Returns:
        gpio_fname: The filename of the GPIO CSV file.
        gpio_table: A pandas DataFrame containing the GPIO data.
    """
    utils.folder_tools.chdir_check_folder(folder_path, create=False)
    gpio_ftag_csv = utils.text_dict()["file_tag"]["GPIO_SUFFIX"]
    gpio_ftag = utils.text_dict()["file_tag"]["GPIO"]

    gpio_fname_csv = utils.findLatest(gpio_ftag_csv)

    if not gpio_fname_csv:
        gpio_file = utils.findLatest(gpio_ftag)
        gpio_fname = f"{gpio_file.split(gpio_ftag)[0]}{gpio_ftag_csv}"
        utils.isx_utils.export_gpio_set_to_csv(gpio_file, gpio_fname)
        gpio_fname_csv = gpio_fname

    gpio_table = pd.read_csv(gpio_fname_csv)

    return gpio_fname_csv.split(gpio_ftag)[0], gpio_table


def parseGPIO(gpio_table: pd.DataFrame, downsample: int = 1) -> dict:
    """
    Parse the GPIO data from the GPIO CSV file.
    """

    def _derive_table_from_channel(channel: str) -> pd.DataFrame:
        return gpio_table[gpio_table[GPIOtxt["CHAN_NAME"]] == channel]

    def _get_syncORcueTimes(data_table: pd.DataFrame) -> np.ndarray:
        Sig = np.array(data_table[GPIOtxt["VAL_KEY"]])
        dSig = np.diff(Sig, prepend=0)
        inds = np.where(dSig > np.max(dSig) / 2)[0]
        return np.array(data_table[GPIOtxt["TIME_KEY"]])[inds]

    GPIOtxt = utils.enum_utils.enum2dict(GPIOfrTimes_Txt)
    # remove leading/trailing whitespace from column names
    gpio_table.columns = gpio_table.columns.str.strip()
    # remove leading/trailing whitespace from "Channel Name" column values
    gpio_table[GPIOtxt["CHAN_NAME"]] = gpio_table[GPIOtxt["CHAN_NAME"]].str.strip()

    gpio1_data = _derive_table_from_channel(GPIOtxt["GPIO_1"])
    gpio2_data = _derive_table_from_channel(GPIOtxt["GPIO_2"])
    frame_data = _derive_table_from_channel(GPIOtxt["FRAME"])
    exled_data = _derive_table_from_channel(GPIOtxt["EX_LED"])

    FrTimes = frame_data.loc[
        frame_data[GPIOtxt["VAL_KEY"]] == 1, GPIOtxt["TIME_KEY"]
    ].values[::downsample]

    # FrTimes = np.insert(FrTimes[:-2], 0, 0)
    FrTimes = FrTimes[:-1] - FrTimes[0]

    syncTimes = _get_syncORcueTimes(gpio1_data)
    cueTimes = _get_syncORcueTimes(gpio2_data)
    exledTimes = np.array(exled_data[GPIOtxt["TIME_KEY"]].iloc[1:3])

    FRdict = {
        "TYPE": "GPIO",
        GPIOtxt["FR_KEY"]: FrTimes,
        GPIOtxt["SYNC_KEY"]: syncTimes,
        GPIOtxt["CUE_KEY"]: cueTimes,
        GPIOtxt["EXLED_KEY"]: exledTimes,
    }

    return FRdict

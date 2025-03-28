import time
import datetime
from CLAH_ImageAnalysis.utils import print_wFrame


class TimeKeeper:
    def __init__(self, cst_msg: str = "Process") -> None:
        """
        Initializes a TimeKeeper object.

        Parameters:
            cst_msg (str): Custom message to be displayed when printing the duration.
        """
        # set start
        self.start = time.perf_counter()
        self.msg_start = cst_msg

    def setEndNprintDuration(self, seconds: bool = True, frame_num: int = 0) -> None:
        """
        Sets the end time and prints the duration of the process.

        Parameters:
            seconds (bool): If True, the duration will be displayed in seconds. If False, it will be displayed in minutes.
            frame_num (int): The frame number to be used to be prepended the print statement.
        """
        self.end = time.perf_counter()
        self.duration = self.end - self.start
        if not seconds:
            self.duration = self.duration / 60
            self.units = "minutes"
        else:
            self.units = "seconds"
        print_wFrame(
            f"{self.msg_start} took: {self.duration:.2f} {self.units}",
            frame_num=frame_num,
        )


def get_current_date_string(wSlashes: bool = False) -> str:
    """
    Returns the current date as a string in the format 'YYYYMMDD'.

    Parameters:
        wSlashes (bool, optional): Whether to include slashes in the date format. Defaults to False.

    Returns:
        str: The current date as a string.
    """
    if wSlashes:
        date_format = "%Y/%m/%d"
    else:
        date_format = "%Y%m%d"
    current_date = datetime.date.today()
    date_str = current_date.strftime(date_format)
    return date_str


def get_current_time_string(wColons: bool = False) -> str:
    """
    Returns the current time as a string in the format 'HHMMSS'.

    Parameters:
        wColons (bool): Whether to include colons in the time format.

    Returns:
        str: The current time as a string.
    """
    if wColons:
        time_format = "%H:%M:%S"
    else:
        time_format = "%H%M%S"

    current_time = datetime.datetime.now()
    time_str = current_time.strftime(time_format)
    return time_str

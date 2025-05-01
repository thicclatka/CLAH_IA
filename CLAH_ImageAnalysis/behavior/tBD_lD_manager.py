import numpy as np
from scipy.interpolate import interp1d

from CLAH_ImageAnalysis.behavior import TDML2tBD_enum as TDMLE
from CLAH_ImageAnalysis.behavior import XML2frTimes2P
from CLAH_ImageAnalysis.behavior import TDML2treadBehDict
from CLAH_ImageAnalysis.behavior import behavior_utils
from CLAH_ImageAnalysis.behavior import GPIOfrTimes
from CLAH_ImageAnalysis.core import BaseClass as BC

# TODO: REMOVE THESE GLOBAL VARS AND MOVE TO CLASS & ADJUST IMPORT STATEMENTS


class tBD_lD_manager(BC):
    """
    A utility class for handling treadBehDict (tBD) and lapDict (lD) data.

    Args:
        folder_path (str): The path to the folder containing the data.

    Attributes:
        folder_path (str): The path to the folder containing the data.
        treadBehDict (dict): The dictionary containing treadBehDict data.
        lapDict (dict): The dictionary containing lapDict data.
        total_cue (int): The total number of unique cues.
        cue_arr (list): The list of cue types.

    Methods:
        __init__(self, folder_path): Initializes the tBD_lD_self object.
        _tBD_lD_checker(self): Checks for the latest TREADBEHDICT and LAPDICT files and loads them if found.
        _lapDictPrinter(self): Prints the loaded lapDict data.
        _tBD_lD_creator(self): Creates the treadBehDict and lapDict data.
        _fill_tBDwFRD(self, vel_conversion): Fills the treadBehDict with FRD data.

    """

    def __init__(self, folder_path):
        self.program_name = "tBD"
        self.class_type = "utils"
        BC.__init__(self, self.program_name, mode=self.class_type)

        self.folder_path = folder_path
        self.treadBehDict = {}
        self.lapDict = {}
        self.total_cue = []
        self.cue_arr = None

        self.TDMLkey = self.enum2dict(TDMLE.TXT)
        self.X2Fkey = self.enum2dict(XML2frTimes2P.xml2FRtimes_Txt)
        self.GPIOkey = self.enum2dict(GPIOfrTimes.GPIOfrTimes_Txt)

        # loads tBD & lD if exists (pkl file)
        self._tBD_lD_checker()
        if not self.treadBehDict or not self.lapDict:
            self._tBD_lD_creator()

    def _tBD_lD_checker(self) -> None:
        """
        Check for the latest TREADBEHDICT and LAPDICT files and load them if found.
        If both files are found, the LAPDICT file will be loaded and printed.

        Returns:
            None
        """
        tBD_name = self.findLatest(self.file_tag["TBD"], [self.file_tag["MAT"]])
        lPD_name = self.findLatest(self.file_tag["LAPD"], [self.file_tag["MAT"]])
        fname_arr = [lPD_name, tBD_name]
        if not tBD_name and not lPD_name:
            self.print_wFrm(
                f"No previous {self.dict_name['TREADBEHDICT']} or {self.dict_name['LAPDICT']} found...starting from scratch"
            )
        elif not tBD_name or not lPD_name:
            self.print_wFrm(
                f"Found one {self.dict_name['TREADBEHDICT']} or {self.dict_name['LAPDICT']} but starting from scratch anyway"
            )
        else:
            for fname in fname_arr:
                if self.folder_tools.fileNfolderChecker(fname):
                    name = (
                        self.dict_name["TREADBEHDICT"]
                        if fname == tBD_name
                        else self.dict_name["LAPDICT"]
                    )
                    self.rprint(name)
                    loaded_dict = self.saveNloadUtils.load_file(fname, previous=True)
                    if fname == tBD_name:
                        self.treadBehDict = loaded_dict
                    else:
                        self.lapDict = loaded_dict
                    if fname == lPD_name and self.lapDict:
                        self._lapDictPrinter()

    def _lapDictPrinter(self) -> None:
        """
        Prints lap dictionary results and calculates cue array based on unique cues.

        Returns:
            None
        """
        # find total cues from loaded lapDict
        unique_cues = set()
        for lap_data in self.lapDict.values():
            cue_type = lap_data[self.TDMLkey["CUETYPE"]]
            unique_cues.update(cue_type)

        # find total_cue & create cue_arr accordingly
        self.total_cue = len(unique_cues)
        self.cue_arr = [cue for cue in unique_cues]

        behavior_utils.print_lapDict_results(self.lapDict, unique_cues)

    def _tBD_lD_creator(self):
        """
        This method is responsible for creating the treadBehDict, lapDict, and cue_arr
        by calling the TDML2treadBehDict function with the folder_path as the argument.
        It also retrieves the FRdict and xml_file by calling the get2pFRTimes function
        with the folder_path as the argument. The key_list is set to [ABS_KEY, REL_KEY, FRIDX_KEY].
        Finally, it calls the _fill_tBDwFRD method and saves the treadBehDict to a file.

        Returns:
            None
        """
        self.treadBehDict, self.lapDict, self.cue_arr = TDML2treadBehDict(
            self.folder_path
        )
        if self.findLatest(self.file_tag["XML"]):
            self.FRdict, data_fname = XML2frTimes2P.get2pFRTimes(self.folder_path)
            self.key_list = [
                self.X2Fkey["ABS_KEY"],
                self.X2Fkey["REL_KEY"],
                self.X2Fkey["FRIDX_KEY"],
            ]
            ftag2remove = self.file_tag["XML"]
        elif self.findLatest([self.file_tag["GPIO_SUFFIX"], self.file_tag["CSV"]]):
            self.FRdict, data_fname = GPIOfrTimes.getGPIOfrTimes(
                self.folder_path, downsample=2
            )
            self.key_list = [
                self.GPIOkey["FR_KEY"],
                self.GPIOkey["SYNC_KEY"],
                self.GPIOkey["CUE_KEY"],
                self.GPIOkey["EXLED_KEY"],
            ]
            ftag2remove = []

        # fill tBD with FRD data
        self._fill_tBDwFRD()

        # save treadBehDict after processing done in fill_tBDwFRD
        # using data_fname as basis for mat_fname, so removing
        self.saveNloadUtils.savedict2file(
            dict_to_save=self.treadBehDict,
            dict_name=self.dict_name["TREADBEHDICT"],
            filename=data_fname,
            file_tag_to_remove=ftag2remove,
            file_suffix=self.dict_name["TREADBEHDICT"],
            date=True,
            filetype_to_save=[self.file_tag["MAT"], self.file_tag["PKL"]],
        )

    def _fill_tBDwFRD(self, vel_conversion: float = 30) -> None:
        """
        Fills the treadBehDict dictionary with frame-related data.

        Args:
            vel_conversion (float, optional): Conversion factor for velocity units. Defaults to 30.

        Returns:
            None
        """

        def _add_by_rel_time(FRtype: str) -> np.ndarray:
            syncTime = self.treadBehDict[self.TDMLkey["SYNC"]][self.TDMLkey["ONTIME"]]
            if FRtype == "XML":
                return [
                    rel_time + syncTime
                    for rel_time in self.FRdict[self.X2Fkey["REL_KEY"]]
                ]
            elif FRtype == "GPIO":
                return self.FRdict[self.GPIOkey["FR_KEY"]] + syncTime

        for key in self.key_list:
            self.treadBehDict[key] = self.FRdict[key]

        adjFrTimes = _add_by_rel_time(self.FRdict["TYPE"])

        self.treadBehDict[self.TDMLkey["ADJ_FRAME"]] = adjFrTimes

        y = np.array(
            self.treadBehDict[self.TDMLkey["POSITION"]][self.TDMLkey["POSITION_KEY"]]
        )
        yTimes = np.array(
            self.treadBehDict[self.TDMLkey["POSITION"]][self.TDMLkey["TIME_ADJ"]]
        )
        frEpochInds = np.where((yTimes >= adjFrTimes[0]) & (yTimes <= adjFrTimes[-1]))[
            0
        ]
        y2 = y[frEpochInds]
        yTimes2 = yTimes[frEpochInds]

        #  position resampled to 2p frame times
        resampY_func = interp1d(yTimes2, y2, kind="linear", fill_value="extrapolate")
        resampY = resampY_func(adjFrTimes)
        #  fix resamp at lap boundaries
        resampY_postfix = fix_resampY(resampY)

        #  calculate velocity
        vel = abs(np.diff(resampY))
        vel = np.append(vel, vel[-1])

        #  position from TDML from same epoch as frames
        self.treadBehDict[self.TDMLkey["POSITION"]][self.TDMLkey["POSITION_KEY"]] = y2
        #  times of TDML position from epoch
        self.treadBehDict[self.TDMLkey["POSITION"]][self.TDMLkey["TIME_KEY"]] = yTimes2
        #  NOTE times = adjFrTimes
        self.treadBehDict[self.TDMLkey["POSITION"]][self.TDMLkey["RESAMP"]] = (
            resampY_postfix
        )
        #  units now mm/sec
        self.treadBehDict[self.TDMLkey["POSITION"]][self.TDMLkey["VELOCITY"]] = (
            vel * vel_conversion
        )


def fix_resampY(resampY: np.ndarray, diff_threshold: int = 100) -> np.ndarray:
    """
    Fix the resampled Y values by applying a correction to the elements that have a negative difference greater than the specified threshold.

    Parameters:
        resampY (numpy.ndarray): The resampled Y values.
        diff_threshold (int): The threshold for the negative difference between consecutive elements. Defaults to 100.

    Returns:
        numpy.ndarray: The corrected resampled Y values.
    """
    # Calculate the difference between consecutive elements
    dy = np.diff(resampY, prepend=0)

    # Find indices where the negative difference is greater than 100
    inds = np.where(-dy > diff_threshold)[0] - 1

    resampY2 = np.copy(resampY)

    # Iterate through the indices & apply the correction
    for i in inds:
        if i > 0 and resampY2[i] < resampY2[i - 1]:
            resampY2[i] = resampY2[i - 1] + (resampY2[i - 1] - resampY2[i - 2])

    return resampY2

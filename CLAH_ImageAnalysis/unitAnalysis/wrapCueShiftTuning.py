"""
This script provides functions and classes for performing cue shift tuning analysis
on spike peak data. The primary function, `wrapCueShiftTuning`, orchestrates the
entire process, including loading necessary data, checking maximum firing rates,
separating spike times by lap type, and creating the cue shift structure.

Functions:
    wrapCueShiftTuning(pksDict, path=[], rewOmit=False): Wraps the cue shift tuning analysis.
    maxFR_checker(pksDict, adjFrTimes, maxFR=0, maxFR_multiplier=1.8): Checks if the maximum firing rate threshold is reached.

Classes:
    CreateCueShiftStruc: Class for creating the cueShiftStruc object used in cue shift tuning analysis.

Dependencies:
    - numpy: For numerical operations on arrays.
    - rich: For printing formatted messages.
    - CLAH_ImageAnalysis.utils: Utility functions for image analysis.
    - CLAH_ImageAnalysis.behavior: Behavioral processing functions.
    - CLAH_ImageAnalysis.PlaceFieldLappedAnalysis: Functions for computing place cells.
    - CLAH_ImageAnalysis.unitAnalysis: Enums and functions for unit analysis.
    - CLAH_ImageAnalysis.unitAnalysis.sepCueShiftLapSpkTimes: Function for separating spike times by lap type.

Usage:
    This script is designed to be imported as a module and used within a larger
    analysis pipeline. The main function `wrapCueShiftTuning` should be called
    with appropriate arguments to perform the analysis.

Example:
    from my_analysis_module import wrapCueShiftTuning

    cueShiftStruc, treadBehDict, lapCueDict, loaded_css = wrapCueShiftTuning(pksDict, path='path/to/data', rewOmit=False)
"""

import numpy as np
from rich import print

from CLAH_ImageAnalysis import utils
from CLAH_ImageAnalysis import behavior as beh
from CLAH_ImageAnalysis.PlaceFieldLappedAnalysis import computePlaceCells
from CLAH_ImageAnalysis.unitAnalysis import UA_enum
from CLAH_ImageAnalysis.unitAnalysis import sepCueShiftLapSpkTimes


# TODO: NEED TO EITHER REMOVE REW OMIT OR ADD ABILITY TO OMIT REWARDS


def wrapCueShiftTuning(
    pksDict: dict, path: str | list = [], rewOmit: bool = False
) -> tuple[dict, dict, dict, bool]:
    """
    Wraps the cue shift tuning analysis.

    Parameters:
        pksDict (dict): Dictionary containing spike peak information.
        path (str, optional): Path to the folder. Defaults to an empty list.
        rewOmit (list, optional): List of rewards to omit. Defaults to an empty list.

    Returns:
        tuple: A tuple containing the following:
            - cueShiftStruc (object): The cue shift structure.
            - treadBehDict (dict): Dictionary containing tread behavior information.
            - lapCueDict (dict): Dictionary containing lap and cue information.
            - loaded_css (bool): Indicates whether the cue shift structure was loaded.
    """

    folder_path = utils.path_selector(path)
    print("Creating/Loading treadBehDict & lapDict:")
    tBD_lD_importer = beh.tBD_lD_manager(folder_path)
    treadBehDict = tBD_lD_importer.treadBehDict
    lapDict = tBD_lD_importer.lapDict
    cue_arr = tBD_lD_importer.cue_arr

    ADJ_FR_TIMES = treadBehDict[beh.TDML2tBD_enum.TXT.ADJ_FRAME.value]
    maxFR_Thresh_Reached = maxFR_checker(pksDict, ADJ_FR_TIMES)

    posLapDict, pksByLTDict, lapCueDict = sepCueShiftLapSpkTimes(
        pksDict,
        treadBehDict,
        cue_arr,
        maxFR_Thresh_Reached=maxFR_Thresh_Reached,
        rewOmit=rewOmit,
    )

    print("Creating/loading cueShiftStruc")
    create_cueShiftStruc = CreateCueShiftStruc(
        posLapDict,
        pksByLTDict,
        lapCueDict,
        lapDict,
        ADJ_FR_TIMES,
        computePlaceCells,
    )

    # create spikeDicts for spikes
    # used for shuffling to find Place cells
    # corresponding results fill PCLappedSessDict
    # then fill cueShiftStruc w/:
    #   PCLappedSessDict
    #   lapCueDict
    #   pksByLTDict
    #   posLapDict
    #   h5 filename
    cueShiftStruc, loaded_css = create_cueShiftStruc.init_cSS()

    return cueShiftStruc, treadBehDict, lapCueDict, loaded_css


def maxFR_checker(
    pksDict: dict, adjFrTimes: list, maxFR_multiplier: float = 1.8
) -> bool:
    """
    Checks if the maximum firing rate threshold is reached.

    Parameters:
        pksDict (dict): A dictionary containing peak values for each segment.
        treadBehDict (dict): A dictionary containing behavioral data.
        maxFR_multiplier (float, optional): The multiplier to determine the threshold. Defaults to 1.8.

    Returns:
        bool: True if the maximum firing rate threshold is reached, False otherwise.
    """

    PEAKS = utils.enum_utils.enum2dict(UA_enum.PKS)["PEAKS"]

    maxFR = 0
    for seg in pksDict:
        if pksDict[seg][PEAKS].size > 0:
            maxFR = max(max(pksDict[seg][PEAKS]), maxFR)
    if len(adjFrTimes) > maxFR * maxFR_multiplier:
        maxFR_Thresh_Reached = True
    else:
        maxFR_Thresh_Reached = False

    return maxFR_Thresh_Reached


class CreateCueShiftStruc:
    """
    Class for creating the cueShiftStruc object used in cue shift tuning analysis.

    Attributes:
        posLapDict (dict): Dictionary containing position data for each lap type.
        pksByLTDict (dict): Dictionary containing peak data for each lap type.
        lapCueDict (dict): Dictionary containing cue data for each lap type.
        lapDict (dict): Dictionary containing lap data.
        adjFrTimes (list): List of adjusted frame times.
        computePlaceCells (object): Object for computing place cells.
        spikeDict (dict): Dictionary to store spike data for each lap type.
        PCLappedSessDict (dict): Dictionary to store place cell data for each lap type.
        cueShiftStruc (dict): Dictionary to store the cueShiftStruc object.
        shuffN (int): Number of shuffles.
        loaded_css (bool): Flag indicating if cueShiftStruc was loaded from a file.

    Methods:
        _spikes_by_lapType: Private method to calculate spikes by lap type.
        _PCLappedSess: Private method to calculate place cell data.
        init_cSS: Main function to initialize cueShiftStruc.
        _find_existing_CSS: Private method to load existing cueShiftStruc file.
    """

    def __init__(
        self,
        posLapDict: dict,
        pksByLTDict: dict,
        lapCueDict: dict,
        lapDict: dict,
        adjFrTimes: list,
        computePlaceCells: object,
    ) -> None:
        """
        Initialize the CreateCueShiftStruc object.

        Parameters:
            posLapDict (dict): Dictionary containing position data for each lap type.
            pksByLTDict (dict): Dictionary containing peak data for each lap type.
            lapCueDict (dict): Dictionary containing cue data for each lap type.
            lapDict (dict): Dictionary containing lap data.
            adjFrTimes (list): List of adjusted frame times.
            computePlaceCells (object): Object for computing place cells.
        """

        self.posLapDict = posLapDict
        self.pksByLTDict = pksByLTDict
        self.lapCueDict = lapCueDict
        self.lapDict = lapDict
        self.adjFrTimes = adjFrTimes
        self.computePlaceCells = computePlaceCells
        self.spikeDict = {}
        self.PCLappedSessDict = {}
        self.cueShiftStruc = {}
        self.shuffN = 1000
        self.loaded_css = False
        self.CSSkey = utils.enum_utils.enum2dict(UA_enum.CSS)
        self.text_lib = utils.text_dict()
        self.file_tag = self.text_lib["file_tag"]
        return

    def _spikes_by_lapType(self):
        """
        Calculate spikes by lap type.
        """
        for lapType, segments in self.pksByLTDict.items():
            max_len = len(self.posLapDict[lapType])
            spikes = np.zeros((len(segments), max_len))

            for seg, pks in segments.items():
                seg_index = int(seg.replace("seg", ""))
                for pk in pks:
                    if pk < max_len:  # Check to ensure pk is w/in the bounds
                        spikes[seg_index, int(pk)] = 1

            self.spikeDict[lapType] = spikes

    def _PCLappedSess(self) -> None:
        """
        Calculate place cell data.
        """

        shuffN = self.shuffN

        # TODO: CREATE WRAPPER FOR THIS
        for typeNum, spikes in self.spikeDict.items():
            print(f"Calculating tuning for {typeNum}")
            print(self.text_lib["breaker"]["lean"])
            treadPos = self.posLapDict[typeNum] / np.max(self.posLapDict[typeNum])

            timeSeg = np.array(self.adjFrTimes[: len(self.posLapDict[typeNum])])

            self.computePC_init = self.computePlaceCells(
                spikes, treadPos, timeSeg, shuffN
            )
            self.PCLappedSessDict[typeNum] = self.computePC_init.LappedWEdges()

            print(self.text_lib["breaker"]["lean"])
            print(f"PCLA completed for {typeNum}\n")

    def init_cSS(self) -> tuple[dict, bool]:
        """
        Initialize cueShiftStruc.

        Returns:
            tuple: A tuple containing the cueShiftStruc object and a flag indicating if cueShiftStruc was loaded from a file.
        """

        self._spikes_by_lapType()
        self._find_existing_CSS()
        if not self.loaded_css:
            utils.print_wFrame(
                "No previous cueShiftStruc found...starting from scratch\n"
            )
            print("Starting tuning to create cueShiftStruc:")
            self._PCLappedSess()
            self.cueShiftStruc[self.CSSkey["FNAME"]] = utils.findLatest(
                self.file_tag["H5"], [self.file_tag["SQZ"], self.file_tag["EMC"]]
            )
            self.cueShiftStruc[self.CSSkey["PKS_BY_LT"]] = self.pksByLTDict
            self.cueShiftStruc[self.CSSkey["POS_LAP"]] = self.posLapDict
            self.cueShiftStruc[self.CSSkey["PCLS"]] = self.PCLappedSessDict
            self.cueShiftStruc[self.CSSkey["LAP_CUE"]] = self.lapCueDict
            self.cueShiftStruc[self.CSSkey["LAP_DICT"]] = self.lapDict

        return self.cueShiftStruc, self.loaded_css

    def _find_existing_CSS(self, ftag: str | None = None) -> None:
        """
        Loads the most recent cueShiftStruc file if it exists.

        Parameters:
            ftag (str, optional): File tag. Defaults to file_tag["PKL"].
        """

        ftag = self.file_tag["PKL"] if ftag is None else ftag

        latest_CSS = utils.findLatest([self.file_tag["CSS"], self.file_tag["PKL"]])

        if latest_CSS:
            self.cueShiftStruc = utils.saveNloadUtils.load_file(
                latest_CSS, previous=True
            )
            self.loaded_css = True
        return

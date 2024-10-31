"""
This script provides functions and classes for separating spike times based on cue shifts and lap types.
It includes a main function `sepCueShiftLapSpkTimes` which orchestrates the process, and several helper
classes and functions to handle various aspects of the data processing.

Functions:
    sepCueShiftLapSpkTimes(pksDict, treadBehDict, cue_arr, rewOmit, cueTypes_of_interest=[], maxFR_Thresh_Reached=False):
        Separates the spike times based on cue shifts and lap types.

    fill_lapCueDict(treadBehDict, cueTypes_of_interest, cue_arr, maxFR_Thresh_Reached, LCDkey, TDMLkey, rewOmit, downsample_factor=2):
        Fills the lapCueDict with lap epochs and frame indices based on the provided parameters.

Classes:
    posLapDictNpksByLTDict_funcs:
        A class that represents the functions for creating dictionaries and processing positional lap data.

        Methods:
            __init__(pksDict, numLapTypes, lapTypeArr, lapEpochs, lapFrInds, resampY, PKSkey):
                Initializes the class with provided parameters.

            create_dicts():
                Creates dictionaries for positional lap data and peak data by lap type.

            process_posLapNpksByLt():
                Processes positional lap data and peak data by lap type.

            fill_posLapNpksByLT(lapType, idx, FrIdx):
                Fills the positional lap data and peak data by lap type.

Main Function:
    The `sepCueShiftLapSpkTimes` function serves as the entry point for the script, orchestrating the separation
    of spike times based on cue shifts and lap types. It uses helper functions and classes to process the data
    and generate the necessary dictionaries for further analysis.

Dependencies:
    - numpy: For numerical operations on arrays.
    - rich: For printing formatted messages.
    - CLAH_ImageAnalysis.utils: Utility functions for image analysis.
    - CLAH_ImageAnalysis.behavior.TDML2tBD_enum: Enums for behavioral to 2-photon data processing.
    - CLAH_ImageAnalysis.unitAnalysis.LapFinder4lapCueDict: Class for finding laps and cues in the data.
    - CLAH_ImageAnalysis.unitAnalysis.UA_enum: Enums for unit analysis.

Usage:
    This script is designed to be imported as a module and used within a larger analysis pipeline. The main function
    `sepCueShiftLapSpkTimes` should be called with appropriate arguments to perform the analysis.

Example:
    from my_analysis_module import sepCueShiftLapSpkTimes

    posLapDict, pksByLTDict, lapCueDict = sepCueShiftLapSpkTimes(
        pksDict=my_pksDict,
        treadBehDict=my_treadBehDict,
        cue_arr=my_cue_arr,
        rewOmit=False,
        cueTypes_of_interest=['type1', 'type2'],
        maxFR_Thresh_Reached=False
    )
"""

import numpy as np
from rich import print as rprint

from CLAH_ImageAnalysis import utils
from CLAH_ImageAnalysis.behavior import TDML2tBD_enum as TDMLE
from CLAH_ImageAnalysis.unitAnalysis import LapFinder4lapCueDict
from CLAH_ImageAnalysis.unitAnalysis import UA_enum


######################################################
# main function
######################################################


def sepCueShiftLapSpkTimes(
    pksDict: dict,
    treadBehDict: dict,
    cue_arr: list,
    rewOmit: bool,
    cueTypes_of_interest: list = [],
    maxFR_Thresh_Reached: bool = False,
) -> tuple[dict, dict, dict]:
    """
    Separates the spike times based on cue shifts and lap types.

    Parameters:
        pksDict (dict): A dictionary containing spike time information.
        treadBehDict (dict): A dictionary containing behavioral data.
        cue_arr (list): A list of cue information.
        cueTypes_of_interest (list, optional):  A list of cue types of interest.
                                                Defaults to an empty list.
        maxFR_Thresh_Reached (bool, optional):  A flag indicating if the maximum
                                                firing rate threshold is reached.
                                                Defaults to False.

    Returns:
        tuple: A tuple containing three dictionaries - posLapDict, pksByLTDict, and lapCueDict.
            - posLapDict (dict): A dictionary containing position and lap information.
            - pksByLTDict (dict): A dictionary containing spike times separated by lap types.
            - lapCueDict (dict): A dictionary containing lap and cue information.
    """

    LCDkey = utils.enum_utils.enum2dict(UA_enum.LCD)
    TDMLkey = utils.enum_utils.enum2dict(TDMLE.TXT)
    PKSkey = utils.enum_utils.enum2dict(UA_enum.PKS)
    RESAMPY = treadBehDict[TDMLkey["POSITION"]][TDMLkey["RESAMP"]]
    LAP_KEY = LCDkey["LAP_KEY"]

    rprint("Initiating & filling lapCueDict:")
    lapCueDict = fill_lapCueDict(
        treadBehDict=treadBehDict,
        cueTypes_of_interest=cueTypes_of_interest,
        cue_arr=cue_arr,
        maxFR_Thresh_Reached=maxFR_Thresh_Reached,
        LCDkey=LCDkey,
        TDMLkey=TDMLkey,
        rewOmit=rewOmit,
    )
    utils.print_wFrame(f"{lapCueDict[LCDkey['NUM_LAP_TYPES']]} lapTypes were found")
    utils.print_done_small_proc()

    # initialize class functions that create:
    # posLapDict & pksByLTDict
    rprint("Creating & filling in dicts for pksByLT & posLap:")
    pLD_pBLD_funcs = posLapDictNpksByLTDict_funcs(
        pksDict=pksDict,
        numLapTypes=lapCueDict[LCDkey["NUM_LAP_TYPES"]],
        lapTypeArr=lapCueDict[LAP_KEY][LCDkey["LAP_TYPEARR"]],
        lapEpochs=lapCueDict[LAP_KEY][LCDkey["LAP_EPOCH"]],
        lapFrInds=lapCueDict[LAP_KEY][LCDkey["LAP_FRIDX"]],
        resampY=RESAMPY,
        PKSkey=PKSkey,
    )

    posLapDict, pksByLTDict = pLD_pBLD_funcs.process_posLapNpksByLt()
    utils.print_done_small_proc()

    return posLapDict, pksByLTDict, lapCueDict


######################################################
# class util functions
######################################################


class posLapDictNpksByLTDict_funcs:
    """
    A class that represents the functions for creating dictionaries and processing
    positional lap data.

    Methods:
        create_dicts(): Creates dictionaries for positional lap data and peak data by lap type.
        process_posLapNpksByLt(): Processes positional lap data and peak data by lap type.
        fill_posLapNpksByLT(): Fills the positional lap data and peak data by lap type.
    """

    def __init__(
        self,
        pksDict: dict,
        numLapTypes: int,
        lapTypeArr: list,
        lapEpochs: list,
        lapFrInds: list,
        resampY: list,
        PKSkey: dict,
    ) -> None:
        """
        Initializes the class with provided parameters.

        Parameters:
            pksDict (dict): A dictionary containing peak data.
            numLapTypes (int): The number of lap types.
            lapTypeArr (list): A list of lap types.
            lapEpochs (list): A list of lap epochs.
            lapFrInds (list): A list of lap frame indices.
            resampY (list): A list of resampled Y values.
            PKSkey (dict): A dictionary containing peak strings.
        """

        self.pksDict = pksDict
        self.numLapTypes = numLapTypes
        self.lapTypeArr = lapTypeArr
        self.lapEpochs = lapEpochs
        self.lapFrInds = lapFrInds
        self.resampY = resampY
        self.PKSkey = PKSkey
        self.posLapDict = {}
        self.pksByLTDict = {}

    def create_dicts(self) -> None:
        """
        Creates dictionaries for positional lap data and peak data by lap type.
        """

        # later posLapDict will be converted into array after being filled
        self.posLapDict = {
            f"{self.PKSkey['LAP_TYPE']}{i+1}": [] for i in range(self.numLapTypes)
        }
        # just create lists w/in entries for now
        for i in range(self.numLapTypes):
            self.pksByLTDict[f"{self.PKSkey['LAP_TYPE']}{i+1}"] = {
                key: [] for key in self.pksDict
            }

        return

    def process_posLapNpksByLt(self) -> tuple[dict, dict]:
        """
        Processes positional lap data and peak data by lap type.

        Returns:
            posLapDict (dict): A dictionary containing positional lap data.
            pksByLTDict (dict): A dictionary containing peak data by lap type.
        """

        self.create_dicts()

        for idx, FrIdx in enumerate(self.lapFrInds):
            lapType = self.lapTypeArr[idx]
            self.fill_posLapNpksByLT(lapType, idx, FrIdx)

        # turn entries in posLapDict into arrays
        for key in self.posLapDict:
            self.posLapDict[key] = np.array(self.posLapDict[key])

        return self.posLapDict, self.pksByLTDict

    def fill_posLapNpksByLT(self, lapType: int, idx: int, FrIdx: int) -> None:
        """
        Fills the positional lap data and peak data by lap type.

        Parameters:
            lapType (int): The lap type.
            idx (int): The index.
            FrIdx (int): The frame index.
        """

        if lapType != 0:
            lapTypeKey = f"{self.PKSkey['LAP_TYPE']}{lapType}"
            start_idx = 0 if idx == 0 else self.lapFrInds[idx - 1]
            end_idx = self.lapFrInds[idx]

            # Offset for peak indices
            current_posLap = len(self.posLapDict[lapTypeKey])

            for seg_num, seg_key in enumerate(self.pksDict.keys()):
                unitPks = np.unique(self.pksDict[seg_key][self.PKSkey["PEAKS"]])
                lapPks = [pk for pk in unitPks if start_idx <= pk and pk <= end_idx]

                if lapPks:
                    if idx == 0:
                        adjusted_pks = lapPks
                    else:
                        adjusted_pks = [
                            pk - start_idx + current_posLap for pk in lapPks
                        ]
                    self.pksByLTDict[lapTypeKey][seg_key].extend(adjusted_pks)

            self.posLapDict[lapTypeKey].extend(self.resampY[start_idx : end_idx + 1])


######################################################
# non-class util functions
######################################################


def fill_lapCueDict(
    treadBehDict: dict,
    cueTypes_of_interest: list,
    cue_arr: list,
    maxFR_Thresh_Reached: bool,
    LCDkey: dict,
    TDMLkey: dict,
    rewOmit: bool,
    downsample_factor: int = 2,
) -> dict:
    """
    Fills the lapCueDict with lap epochs and frame indices based on the provided parameters.

    Parameters:
        treadBehDict (dict): Dictionary containing behavioral data.
        cueTypes_of_interest (list): List of cue types of interest.
        cue_arr (ndarray): Array of cue values.
        maxFR_Thresh_Reached (bool): Indicates if the maximum frame threshold is reached.
        downsample_factor (int, optional): Downsample factor for resampled position. Defaults to 2.

    Returns:
        dict: Dictionary containing lap epochs and frame indices.
    """

    lapCueDict = UA_enum.create_lapCueDict()

    resampY = treadBehDict[TDMLkey["POSITION"]][TDMLkey["RESAMP"]]
    adjFrTimes = treadBehDict[TDMLkey["ADJ_FRAME"]]

    if maxFR_Thresh_Reached:
        utils.print_wFrame(
            (
                "max frame threshold reached => "
                "downsampling resampled position by a factor of {downsample_factor}"
            )
        )
        resampY = resampY[::downsample_factor]

    # appendFunc = Appender4lapCueDict(lapCueDict=lapCueDict)
    finderFunc = LapFinder4lapCueDict(
        resampY=resampY,
        adjFrTimes=adjFrTimes,
        lapCueDict=lapCueDict,
        treadBehDict=treadBehDict,
        LCDkey=LCDkey,
        TDMLkey=TDMLkey,
        cue_arr=cue_arr,
        rewOmit=rewOmit,
    )

    # findLapDict = finderFunc.LapEpochsNInds()  # find lapEpochs & FrInds
    # lapCueDict = appendFunc.add_cue(findLapDict)  # add to lapCueDict
    lapCueDict[LCDkey["LAP_KEY"]] = (
        finderFunc.LapEpochsNInds()
    )  # find lapEpochs & FrInds
    # line below does a lot need to add more comments
    # TODO: add more comments
    lapCueDict = finderFunc.cueLapsProcessor(cueTypes4LCD=cueTypes_of_interest)

    return lapCueDict

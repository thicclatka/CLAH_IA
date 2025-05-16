from collections import OrderedDict

import numpy as np

from CLAH_ImageAnalysis.behavior import TDML2tBD_enum as TDMLE
from CLAH_ImageAnalysis.core import BaseClass as BC
from CLAH_ImageAnalysis.PlaceFieldLappedAnalysis import PCLA_enum
from CLAH_ImageAnalysis.unitAnalysis import CCF_Dep, UA_enum


class CCF_Utils(BC):
    def __init__(self, cueShiftStruc, treadBehDict) -> None:
        self.program_name = "CCF"
        self.class_type = "utils"
        BC.__init__(self, self.program_name, mode=self.class_type)

        self.LCDkey = self.enum2dict(UA_enum.LCD)
        self.TDMLkey = self.enum2dict(TDMLE.TXT)
        self.PCLAkey = self.enum2dict(PCLA_enum.TXT)
        self.CSSkey = self.enum2dict(UA_enum.CSS)
        self.CCFkey = self.enum2dict(UA_enum.CCF)
        self._unpack_CCFkey()

        # store lapCueDict & related variables into self
        self.lapCueDict = cueShiftStruc[self.CSSkey["LAP_CUE"]]
        self.lapEpochs = self.lapCueDict[self.LCDkey["LAP_KEY"]][
            self.LCDkey["LAP_EPOCH"]
        ]
        self.lapFrInds = self.lapCueDict[self.LCDkey["LAP_KEY"]][
            self.LCDkey["LAP_FRIDX"]
        ]

        self.lapTypeArr = self.lapCueDict[self.LCDkey["LAP_KEY"]][
            self.LCDkey["LAP_TYPEARR"]
        ].copy()

        # store treadBehDict & related variables into self
        self.treadBehDict = treadBehDict
        self.resampY = treadBehDict[self.TDMLkey["POSITION"]][self.TDMLkey["RESAMP"]]
        self.frTimes = treadBehDict[self.TDMLkey["ADJ_FRAME"]]

        # PCLS
        self.PCLS = cueShiftStruc[self.CSSkey["PCLS"]]

        # set of cues
        self.cues = [
            self.LCDkey["CUE1"],
            self.LCDkey["CUE2"],
            self.LCDkey["LED"],
            self.LCDkey["TACT"],
            self.LCDkey["TONE"],
            self.LCDkey["OPTO"],
        ]

        # for opto organization
        self.optoCheck = False

        self._find_binnedCuePos()
        self._find_cueLapBins()
        self._find_omitNswitchLaps()
        self._find_omitCueTimes()

    def _unpack_CCFkey(self) -> None:
        """
        Unpacks the CCF key dictionary and assigns the values to instance variables.

        This method extracts the values from the CCF key dictionary and assigns them to
        instance variables for easier access and readability.

        """
        self.CUE1 = self.CCFkey["C1"]
        self.CUE2 = self.CCFkey["C2"]
        self.BOTH = self.CCFkey["B"]
        self.CUE1_L1 = self.CCFkey["C1L1"]
        self.CUE1_L2 = self.CCFkey["C1L2"]
        self.TONE = self.CCFkey["T"]
        self.LED = self.CCFkey["L"]
        self.OPTO = self.CCFkey["OP"]
        self.ALL = self.CCFkey["A"]
        self.SWITCH = self.CCFkey["SW"]
        self.SHIFT = self.CCFkey["SH"]
        self.OMIT = self.CCFkey["OM"]
        self.OBOTH = f"{self.OMIT}{self.BOTH}"
        self.OALL = f"{self.OMIT}{self.ALL}"
        self.OCUE1 = f"{self.OMIT}{self.CUE1}"
        self.OCUE2 = f"{self.OMIT}{self.CUE2}"
        self.OC1L1 = f"{self.OMIT}{self.CUE1_L1}"
        self.OC1L2 = f"{self.OMIT}{self.CUE1_L2}"
        self.OC1SW = f"{self.OMIT}{self.CUE1}_{self.SWITCH}"
        self.C1SW = f"{self.CUE1}_{self.SWITCH}"

        # extract ind for stat testing & other later analyses
        self.ind = self.CCFkey["IND"]
        self.ind_size = np.abs(self.ind[0]) + np.abs(self.ind[-1])

    def _find_binnedCuePos(self) -> None:
        """
        Finds the binned cue positions for each cue in the `cues` list.

        The binned cue positions are stored in the `binnedCuePos` dictionary, where the keys are the cues and the values
        are the corresponding binned cue positions.

        If the cue is not the LED cue and the length of the binned cue positions is not equal to 2, the binned cue positions
        are directly assigned to the `binnedCuePos` dictionary.

        If the cue is not the LED cue and the length of the binned cue positions is equal to 2, the binned cue positions
        are also assigned to the `binnedCuePos` dictionary.

        If the cue is the LED cue, a fluke position seen with LED during 3 cue for 1 session is handled by assigning only
        the first element of the binned cue positions to the `binnedCuePos` dictionary.

        The set of total locations based on the binned cue positions is stored in the `total_loc_set` attribute.

        Returns:
            None
        """
        self.binnedCuePos = {}
        for cue in self.cues:
            curr_binnedCuePos = self.lapCueDict[cue][self.LCDkey["BINNEDPOS"]]
            if len(curr_binnedCuePos) > 0:
                if cue != self.LCDkey["LED"] and len(curr_binnedCuePos) != 2:
                    self.binnedCuePos[cue] = curr_binnedCuePos
                elif cue != self.LCDkey["LED"] and len(curr_binnedCuePos) == 2:
                    self.binnedCuePos[cue] = curr_binnedCuePos
                else:
                    # this appears for fluke position seen with LED during 3 cue for 1 session
                    self.binnedCuePos[cue] = [curr_binnedCuePos[0]]

        # get set of total locations based on binnedCuePos
        self.total_loc_set = set()
        for cue in self.binnedCuePos:
            self.total_loc_set.update(self.binnedCuePos[cue])

    def _find_cueLapBins(self) -> None:
        """
        Finds the cue lap bins for each cue in the `cues` list.

        Returns:
            None

        Modifies:
            - `self.cueLapBin`: A dictionary containing the cue lap bins for each cue.
            - `self.cues`: The `cues` list with cues that have empty lap bins removed.
        """
        self.cueLapBin = {}
        cues2remove = set()
        for cue in self.cues:
            self.cueLapBin[cue] = np.array(
                self.lapCueDict[cue][self.LCDkey["LAP_BIN_KEY"]]
            )
            if not self.cueLapBin[cue].size:
                cues2remove.add(cue)
        self.cues = sorted(list(set(self.cues) - cues2remove))

        if "OPTO" in self.cues:
            self.optoCheck = True

    def _find_omitNswitchLaps(self) -> None:
        """
        Finds the omitted laps and switch laps based on the cue locations.

        Returns:
            tuple: A tuple containing two arrays:
                - omitLaps: A dictionary where the keys represent different conditions and the values are arrays of omitted laps.
                - switchLaps: An array of switch laps.
        """
        # for 1+ cue(s) but only 1 location per cue == 1
        # for 2 cues w/switch == 2
        self.max_num_loc = max(np.max(self.cueLapBin[cue]) for cue in self.cues)
        # Initialize omitLaps dictionary
        key_suffixes = [
            self.CUE1,
            self.CUE2,
            self.BOTH,
            self.CUE1_L1,
            self.CUE1_L2,
            self.TONE,
            self.LED,
            self.OPTO,
            self.ALL,
        ]

        omitLaps = {f"{self.OMIT}{suffix}": [] for suffix in key_suffixes}
        switchLaps = []

        # create condition mappings based on max_num_loc
        # keys of this dictionary are tuples of form (max_num_loc, len(cues))
        # in key entries, tuples are organized of form (cue1_loc, cue2_loc, ..., cueType (string))
        key4cond = (self.max_num_loc, len(self.cues))
        cond_sfx = {
            (2, 2): [
                (0, 2, self.CUE1),
                (1, 0, self.CUE2),
                (0, 0, self.BOTH),
                (2, 1, self.SWITCH),
            ],
            (1, 2): [
                (0, 1, self.cues[0]),
                (1, 0, self.cues[-1]),
                (0, 0, self.BOTH),
            ],
            (1, 1): [(0, self.CUE1)],
            (2, 1): [(2, self.CUE1_L1), (1, self.CUE1_L2), (0, self.BOTH)],
            (1, 3): [
                (0, 0, 0, self.ALL),
                (1, 1, 0, self.TONE),
                (1, 0, 1, self.LED),
                (0, 1, 1, self.CUE1),
            ],
        }
        condition_dict = {
            tuple(val[:-1]): (
                f"{self.OMIT}{val[-1]}" if val[-1] != self.SWITCH else val[-1]
            )
            for val in cond_sfx.get(key4cond, [])
        }

        for lap_idx, lapType in enumerate(self.lapTypeArr):
            clb = {cue: self.cueLapBin[cue][lap_idx] for cue in self.cues}
            condition = tuple(clb.get(cue, []) for cue in self.cues)

            if condition in condition_dict:
                key = condition_dict[condition]
                if key != self.SWITCH:
                    omitLaps[key].append(lap_idx)
                else:
                    switchLaps.append(lap_idx)

        # convert lists to arrays
        for key in omitLaps:
            omitLaps[key] = np.array(omitLaps[key])
        switchLaps = np.array(switchLaps)

        # store omitLaps and switchLaps into self
        self.omitLaps = omitLaps
        self.switchLaps = switchLaps

    def _find_omitCueTimes(self) -> None:
        """
        Finds the omit cue times based on the given lap epochs and omit laps.

        Returns:
            dict:   A dictionary containing the omit cue times for each omit key.
                    The keys of the dictionary are the omit keys, and the values are
                    numpy arrays representing the omit cue times.
        """
        lapEpochs = self.lapEpochs
        max_epoch_length = 0
        for epoch in lapEpochs:
            epoch_diff = np.max(epoch[1] - epoch[0])
            if epoch_diff > max_epoch_length:
                max_epoch_length = epoch_diff

        OL_keys_to_use = [key for key, value in self.omitLaps.items() if len(value) > 0]
        # init omitCueTimes dict
        # self.total_loc_set is used for # of rows:
        #   - if 1 cue/1loc => 1 row
        #   - if 3 cues/1loc per cue => 3 rows
        #   - if 1 cue/2loc => 2 rows
        #   - if 2 cue/2loc => 2 rows
        self.omitCueTimes = {
            key: np.full(
                (
                    (len(self.total_loc_set), len(self.omitLaps[key]))
                    if key == self.OALL or key == self.OBOTH
                    else len(self.omitLaps[key])
                ),
                np.nan,
            )
            for key in OL_keys_to_use
        }

        for omit_key in OL_keys_to_use:
            for omitIdx, omitLap in enumerate(self.omitLaps[omit_key]):
                epochSlice = slice(lapEpochs[omitLap][0], lapEpochs[omitLap][1] + 1)
                epochPos = self.resampY[epochSlice]
                epochTimes = self.frTimes[epochSlice]

                if omit_key not in [self.OBOTH, self.OALL]:
                    self._process_non_both_omit(omit_key, omitIdx, epochPos, epochTimes)
                elif omit_key == self.OBOTH:
                    self._process_both_omit(omitIdx, epochPos, epochTimes)
                elif omit_key == self.OALL:
                    self._process_all_omit(omitIdx, epochPos, epochTimes)

    def _assign_omit_times(
        self, bCP_key, omit_key, cueIdx, omitIdx, epochPos, epochTimes
    ) -> None:
        """
        Assigns omit times based on the given parameters.

        Parameters:
            bCP_key (str): The key for accessing binnedCuePos dictionary.
            omit_key (str): The key for accessing omitCueTimes dictionary.
            cueIdx (int): The index of the cue.
            omitIdx (int): The index of the omit.
            epochPos (numpy.ndarray): Array of epoch positions.
            epochTimes (numpy.ndarray): Array of epoch times.

        Returns:
            None
        """
        cuePosInd = np.argwhere(epochPos > self.binnedCuePos[bCP_key][cueIdx])
        if cuePosInd.size > 0 and cuePosInd[0][0] <= len(epochTimes):
            cuePosInd = cuePosInd[0][0]
            if not np.isnan(epochTimes[cuePosInd]):
                if self.omitCueTimes[omit_key].ndim == 1:
                    self.omitCueTimes[omit_key][omitIdx] = epochTimes[cuePosInd]
                else:
                    self.omitCueTimes[omit_key][cueIdx, omitIdx] = epochTimes[cuePosInd]

    def _process_non_both_omit(self, omit_key, omitIdx, epochPos, epochTimes):
        """
        Process non-both omit.

        This method is used to process non-both omit cues. It assigns epoch times based on the given omit key and other parameters.

        Parameters:
        - omit_key (str): The omit key to process.
        - omitIdx (int): The omit index.
        - epochPos (list): The epoch positions.
        - epochTimes (list): The epoch times.

        Returns:
        None
        """
        for idx, cue in enumerate(self.cues):
            if self.max_num_loc == 1:
                idx_binPos = 0
            else:
                idx_binPos = idx
            # for shortcuts
            OC = f"{self.OMIT}{cue}"
            OCL2 = f"{self.OMIT}{cue}_L2"
            OCL2F = f"{self.OMIT}{cue}_L{idx_binPos + 1}"
            if omit_key == OCL2:
                idx_binPos = 1
            if omit_key == OC or omit_key == OCL2F:
                self._assign_epoch_times(
                    omitIdx, idx_binPos, epochPos, epochTimes, omit_key
                )

    def _process_all_omit(self, omitIdx, epochPos, epochTimes) -> None:
        """
        Process laps where all cues are omitted. This is generally for the 3 cue experiments.

        Args:
            omitIdx (int): The index of the omitted cue.
            epochPos (list): List of epoch positions.
            epochTimes (list): List of epoch times.

        Returns:
            None
        """
        for bCP_key in self.binnedCuePos:
            cueIdx = 0
            # if bCP_key == CCFK["C1"]:
            #     cueIdx = 0
            # elif bCP_key == "LED":
            #     cueIdx = 1
            # elif bCP_key == "TONE":
            #     cueIdx = 2
            self._assign_omit_times(
                bCP_key,
                self.OALL,
                cueIdx,
                omitIdx,
                epochPos,
                epochTimes,
            )

    def _process_both_omit(self, omitIdx, epochPos, epochTimes) -> None:
        """
        Process laps where both cues are omitted. This is generally used for 2 cue experiments.

        Args:
            omitIdx (int): The index of the omit condition.
            epochPos (list): List of epoch positions.
            epochTimes (list): List of epoch times.

        Returns:
            None
        """
        first_key = next(iter(self.binnedCuePos))
        for cp_idx, cuePos in enumerate(self.binnedCuePos[first_key]):
            self._assign_epoch_times(omitIdx, cp_idx, epochPos, epochTimes, self.OBOTH)

    def _assign_epoch_times(
        self, omitIdx, cueIdx, epochPos, epochTimes, omit_key
    ) -> None:
        """
        Assigns epoch times based on the given parameters.

        Parameters:
        - omitIdx (int): The index of the omitted epoch.
        - cueIdx (int): The index of the cue epoch.
        - epochPos (list): The positions of the epochs.
        - epochTimes (list): The times of the epochs.
        - omit_key (str): The key indicating the type of omission.

        Returns:
        None
        """
        if omit_key != self.OBOTH:
            bCP_key = omit_key.split(self.OMIT)[1]
            if "_L" in bCP_key:
                bCP_key = bCP_key.split("_L")[0]
        else:
            bCP_key = next(iter(self.binnedCuePos))
            # bCP_key = self.CUE1
        self._assign_omit_times(
            bCP_key, omit_key, cueIdx, omitIdx, epochPos, epochTimes
        )

    def _setup_refLapType_vars(self, refLapType, lapTypeNameArr) -> None:
        """
        Sets up the reference lap type variables.

        Args:
            refLapType (int): The reference lap type.
            lapTypeNameArr (list): The array of lap type names.

        Returns:
            None
        """
        self.lapTypeNameArr = lapTypeNameArr
        self.refLapType = refLapType
        self.refLapType_key = f"lapType{refLapType + 1}"
        self.refLapType_name = lapTypeNameArr[refLapType]

    def _setup_numOdor_numLoc_vars(self) -> None:
        """
        Sets up variables to determine the number of odors and locations in the experiment.

        This method checks the lapTypeNameArr attribute to determine the experimental conditions.
        If the experiment involves two odors in one location, the attribute twoOdor_oneLoc is set to True.
        If the experiment involves one odor in two locations, the attribute oneOdor_twoLoc is set to True.
        """
        # for 2 odor in one location experiment
        self.twoOdor_oneLoc = False
        exp_2odor_vals = {self.CUE1, self.CUE2, self.OBOTH}
        if exp_2odor_vals.issubset(set(self.lapTypeNameArr)):
            self.twoOdor_oneLoc = True

        self.oneOdor_twoLoc = False
        exp_1odor_vals = {self.CUE1, self.SHIFT, self.OBOTH}
        if exp_1odor_vals.issubset(set(self.lapTypeNameArr)):
            self.oneOdor_twoLoc = True

    def _init_posRates_forRef_n_nonRef(self) -> None:
        """
        Initializes the positive rates for the reference and non-reference lap types.

        This method calculates and stores the positive rates for the reference lap type,
        as well as for any non-reference lap types specified in the `omitLaps` and `switchLaps` dictionaries.

        Returns:
            None
        """
        isPC = self.PCLS[self.refLapType_key][self.PCLAkey["SHUFF"]][
            self.PCLAkey["ISPC"]
        ]
        self.PC = np.where(isPC == 1)[0]

        # pos rates for ref lap type
        self.posRatesRef = self.PCLS[self.refLapType_key][self.PCLAkey["POSRATE"]]
        # TODO: CHECK THIS maxpRR_ind
        self.maxpRR_ind, self.sortInd = self.dep.find_maxIndNsortmaxInd(
            self.posRatesRef
        )
        self.pcPkPos = self.maxpRR_ind[self.PC]

        # fill in posRates dict of nonReference
        self.posRatesNonRef = {}
        for omit_keys in self.omitLaps.keys():
            if self.omitLaps[omit_keys].any():
                self._fill_in_posRatesNonRef(self.omitLaps[omit_keys], omit_keys)
        if self.switchLaps.any():
            self._fill_in_posRatesNonRef(self.switchLaps, self.SWITCH)
        if self.optoCheck:
            self._fill_in_posRatesNonRef(
                self.lapTypeArr == self.lapTypeNameArr.index("CUEwOPTO") + 1,
                "CUEwOPTO",
            )

        if self.twoOdor_oneLoc:
            # need to sort dict so that CUE1 shows up before CUE2
            sorted_keys = sorted(self.posRatesNonRef.keys())
            sorted_dict = OrderedDict((k, self.posRatesNonRef[k]) for k in sorted_keys)
            self.posRatesNonRef = sorted_dict

    def _fill_in_posRatesNonRef(self, arr_to_use, key_to_use) -> None:
        """
        Fills in the `posRatesNonRef` dictionary with the positional rates for a given key.

        Parameters:
        - arr_to_use: The array to use for determining the lap type.
        - key_to_use: The key to use for filling in the `posRatesNonRef` dictionary.

        Returns:
        None
        """
        lapType_of_int = int(np.unique(self.lapTypeArr[arr_to_use])[0])
        lt_key = f"lapType{lapType_of_int}"

        # lap name correction for 2odor exp
        # exp are different here because lapTypeNameArr has already been modified in QT_utils
        if self.twoOdor_oneLoc:
            key_to_use = (
                self.CUE1
                if key_to_use == self.OCUE2
                else self.CUE2
                if key_to_use == self.OCUE1
                else key_to_use
            )
        if self.oneOdor_twoLoc:
            key_to_use = self.SHIFT if key_to_use == self.OC1L2 else key_to_use

        if lt_key != self.refLapType_key:
            self.posRatesNonRef[key_to_use] = self.PCLS[lt_key][self.PCLAkey["POSRATE"]]

    def _init_CueCellInd_for_Start_n_Mid(self) -> None:
        """
        Initializes the CueCellInd dictionary for the "START" and "MID" cues.

        The "START" cue includes PC cells with peak positions greater than or equal to 90 or less than or equal to 10.
        The "MID" cue includes PC cells with peak positions between 20 and 80 (inclusive).
        """
        startRange = (self.pcPkPos >= 90) | (self.pcPkPos <= 10)
        pcRange = (self.pcPkPos > 10) & (self.pcPkPos < 90)

        self.CueCellInd["START"] = self.PC[startRange]
        self.CueCellInd["PC"] = self.PC[pcRange]

        if not self.optoCheck and len(self.cues) == 1 and len(self.total_loc_set) == 1:
            midRange = (self.pcPkPos >= 40) & (self.pcPkPos <= 65)
        elif (
            not self.optoCheck and len(self.cues) == 1 and len(self.total_loc_set) == 2
        ):
            midRange = (self.pcPkPos >= 20) & (self.pcPkPos <= 70)
        elif self.optoCheck:
            midRange = (self.pcPkPos >= 40) & (self.pcPkPos <= 70)
        else:
            midRange = pcRange

        self.CueCellInd["MID"] = self.PC[midRange]

    def _find_evTimeByCue(self, bcp_edge=6) -> None:
        """
        Finds event times by cue type and fills in the `evTimes` dictionary.

        Args:
            bcp_edge (int): The number of bins to consider as the edge of the binned cue position.

        Returns:
            None
        """
        self.evTimes = {}
        # self.evPos = {}
        self.bcp_edge = bcp_edge

        # Iterate over each key in treadBehDict[CUE_EVENTS]
        for key in self.treadBehDict[self.TDMLkey["CUE_EVENTS"]]:
            self.key_upper = key.upper()
            self.times_arr = np.array(
                self.treadBehDict[self.TDMLkey["CUE_EVENTS"]][key][
                    self.TDMLkey["START"]
                ][self.TDMLkey["TIME_KEY"]]
            ).copy()
            self.pos_arr = np.array(
                self.treadBehDict[self.TDMLkey["CUE_EVENTS"]][key][
                    self.TDMLkey["START"]
                ][self.TDMLkey["POSITION_KEY"]]
            ).copy()
            self.curr_binnedCuePos = self.binnedCuePos[self.key_upper]
            # fills in evTimesDict according to CueType
            self._fill_evTimesDict()
            # handles 2 locations w/switch laps
            if len(self.curr_binnedCuePos) > 1 and not self.optoCheck:
                self._fill_evTimesDict(switch=True)

        if self.optoCheck:
            self._fill_evTimesDict_OptoFix()
        else:
            for omit_key in self.omitCueTimes:
                curr_omitCueTime = self.omitCueTimes[omit_key]
                if (
                    omit_key not in [self.OBOTH, self.OALL, self.OC1L1, self.OC1L2]
                    and not self.twoOdor_oneLoc
                ):
                    self.evTimes[omit_key] = np.array(curr_omitCueTime.copy())
                elif omit_key == self.OBOTH and self.twoOdor_oneLoc:
                    self.evTimes[omit_key] = np.array(curr_omitCueTime.copy())
                    # remove NaN row that was added when init omitCueTimes
                    # sometimes total_loc_set gives 2 locs instead of 1 (ie 903, 904), so this adjusts for that
                    if self.evTimes[omit_key].shape[0] > 1:
                        self.evTimes[omit_key] = np.delete(
                            self.evTimes[omit_key], 1, axis=0
                        )
                    self.evTimes[omit_key] = self.evTimes[omit_key].T
                elif omit_key == self.OC1L1:
                    self.evTimes[self.OC1SW] = np.array(curr_omitCueTime).copy()
                elif omit_key == self.OC1L2:
                    self.evTimes[self.OCUE1] = np.array(curr_omitCueTime).copy()
            # if self.twoOdor_oneLoc:
            #     self.evTimes["OMITCUE1_SWITCH"] = self.evTimes["OMITCUE1_L1"].copy()

            for omit_key in self.omitCueTimes:
                if omit_key == self.OBOTH and not self.twoOdor_oneLoc:
                    if (
                        not self.oneOdor_twoLoc
                        and "OMITOPTO" not in self.omitCueTimes.keys()
                    ):
                        self.cue_list = [self.CUE1, self.CUE2]
                    elif "OMITOPTO" in self.omitCueTimes.keys():
                        self.cue_list = [self.CUE1, self.OPTO]
                    else:
                        self.cue_list = [self.C1SW, self.CUE1]

                    for c_idx, cue in enumerate(self.cue_list):
                        omitCueTime = self.omitCueTimes[self.OBOTH][c_idx, :]
                        omitCueTime = omitCueTime[~np.isnan(omitCueTime)].T
                        self.evTimes[f"{self.OMIT}{cue}"] = np.concatenate(
                            [self.evTimes[f"{self.OMIT}{cue}"], omitCueTime]
                        )
                if omit_key == self.OALL:
                    for c_idx, cue in enumerate([self.CUE1, self.LED, self.TONE]):
                        omitCueTime = self.omitCueTimes[self.OALL][c_idx, :]
                        omitCueTime = omitCueTime[~np.isnan(omitCueTime)].T
                        self.evTimes[f"{self.OMIT}{cue}"] = np.concatenate(
                            [self.evTimes[f"{self.OMIT}{cue}"], omitCueTime]
                        )

        for key in self.evTimes:
            self.evTimes[key] = np.array(self.evTimes[key])

    def _fill_evTimesDict(self, switch=False) -> None:
        """
        Fills the evTimes dictionary with event times based on the current key and switch status.

        Parameters:
            switch (bool): Indicates whether the switch is enabled or not. Default is False.

        Returns:
            None
        """
        if self.key_upper != self.CUE2 and not self.oneOdor_twoLoc:
            index = 0 if not switch else -1
        elif self.key_upper == self.CUE2 or (
            self.key_upper == self.CUE1 and self.oneOdor_twoLoc
        ):
            index = -1 if not switch else 0
            if self.twoOdor_oneLoc:
                # if 2odor/1loc for Cue2 revert index to 0
                index = 0
        EVT_name = f"{self.key_upper}_{self.SWITCH}" if switch else self.key_upper
        if (
            self.key_upper == self.CUE1
            and self.optoCheck
            and len(self.curr_binnedCuePos) == 3
        ):
            self.bcp_edge = 31
        self.evTimes[EVT_name] = self.times_arr[
            (self.pos_arr >= self.curr_binnedCuePos[index] - self.bcp_edge)
            & (self.pos_arr <= self.curr_binnedCuePos[index] + self.bcp_edge)
        ]

    def _fill_evTimesDict_OptoFix(self) -> None:
        # Assuming self.evTimes[self.CUE1] and self.evTimes[self.OPTO] are 2D numpy arrays
        cuekey2use = self.CUE1 if self.CUE1 in self.cues else self.CUE2
        cue = np.column_stack(
            [
                self.evTimes[cuekey2use],
                np.full(self.evTimes[cuekey2use].shape[0], fill_value=0),
            ]
        )
        opto = np.column_stack(
            [
                self.evTimes[self.OPTO],
                np.full(self.evTimes[self.OPTO].shape[0], fill_value=1),
            ]
        )

        # Vertically concatenate the modified arrays
        opto_arr = np.concatenate([cue, opto], axis=0)
        opto_arr = opto_arr[np.argsort(opto_arr[:, 0])]

        cueWopto = []
        cueOnly = []
        for idx, (time, ctype) in enumerate(opto_arr):
            ctype_next = opto_arr[idx + 1, 1] if idx + 1 < opto_arr.shape[0] else -1
            if ctype == 0 and ctype_next == 1:
                cueOnly.append(time)
            if ctype == 0 and ctype_next == 0:
                ctype_prev = opto_arr[idx - 1, 1] if idx - 1 >= 0 else -1
                time_prev = opto_arr[idx - 1, 0] if idx - 1 >= 0 else None
                if ctype_prev == 1:
                    time_diff = round(time - time_prev)
                    if time_diff == 1:
                        # This is a cue that was immediately preceded by opto, so will skip
                        continue
                    else:
                        cueOnly.append(time)
                else:
                    cueOnly.append(time)
            if ctype == 1 and ctype_next == 0:
                time_next = opto_arr[idx + 1, 0]
                time_diff = round(time_next - time)
                if time_diff == 1:
                    cueWopto.append(time_next)
            if ctype_next == -1:
                if ctype == 0:
                    cueOnly.append(time)

        self.evTimes["CUEwOPTO"] = np.array(cueWopto)
        self.Cuekey2use_BU = self.evTimes[cuekey2use].copy()
        self.evTimes[cuekey2use] = np.array(cueOnly)
        self.evTimes["OPTO"] = self.omitCueTimes[f"OMIT{cuekey2use}"].copy()
        self.evTimes["OMITBOTH"] = self.omitCueTimes["OMITBOTH"][0, :].copy()

    def _find_evTrigSig(
        self, Ca_arr: np.ndarray, evKey: str, ds_factor: int = 1
    ) -> np.ndarray:
        """
        Finds the event-triggered signal for a given event key.

        Args:
            Ca_arr (numpy.ndarray): Array of calcium values.
            evKey (str): Key of the event.
            ds_factor (int): Downsample factor. Default is 1.

        Returns:
            numpy.ndarray: Event-triggered signal array.

        """
        sigTime = np.array(self.frTimes[::ds_factor])
        evTrigSig = np.full((self.ind_size, len(self.evTimes[evKey])), np.nan)
        for idx, evTime in enumerate(self.evTimes[evKey]):
            start_idx, end_idx, slice_length = CCF_Dep.calcInd4evTrigSig(
                sigTime, evTime, self.ind, len(Ca_arr)
            )
            evTrigSig[:slice_length, idx] = Ca_arr[start_idx:end_idx]
        return evTrigSig

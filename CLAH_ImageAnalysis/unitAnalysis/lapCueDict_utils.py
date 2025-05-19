import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

######################################################
# finder func
######################################################


class LapFinder4lapCueDict:
    def __init__(
        self,
        resampY: np.ndarray,
        adjFrTimes: np.ndarray,
        lapCueDict: dict,
        treadBehDict: dict,
        LCDkey: dict,
        TDMLkey: dict,
        cue_arr: list,
        rewOmit: bool = False,
    ) -> None:
        self.resampY = resampY
        self.adjFrTimes = adjFrTimes
        self.LCDkey = LCDkey
        self.TDMLkey = TDMLkey
        self.appender = Appender4lapCueDict(lapCueDict, LCDkey, TDMLkey)
        self.lapCueDict = lapCueDict
        self.treadBehDict = treadBehDict
        self.cue_arr = cue_arr
        self.lapEpochs = None
        self.lapFrInds = None
        self.numLapTypes = None
        # default: cue1, cue2, rew(lCD)/rewZone(tdml), tone, tact, led, opto
        # this var needs to be initialized before self.cue_dict
        self.cueTypes_keylist = [
            *[LCDkey[f"{cue.upper()}"] for cue in self.cue_arr],
            LCDkey["REW"],
        ]
        # by default via automation: cue 1, cue 2, opto, tact, tone, led
        # cueTypes w/ a lap
        self.cueTypes4LCD = None
        self.edges4cueTypeLaps = None
        # Note: ends of lap via RFID not included, so add 1 to cover
        self.numLaps_RFID = (
            len(self.treadBehDict[TDMLkey["LAP_KEY"]][TDMLkey["TIME_KEY"]]) + 1
        )
        self.numLaps_Epochs = None  # this is fill in when lapEpochs are found
        # Default order for lapCode:
        # cue1  = 10^0
        # cue2  = 10^1
        # led   = 10^2
        # tone  = 10^3
        # tact  = 10^4
        # opto  = 10^5
        self.cueTypes4lapCode = [
            *[LCDkey[f"{cue.upper()}"] for cue in self.cue_arr],
        ]
        self.cueTypes4lapCode.sort()

        self.power_of_ten_dict = {}
        self.lapTypeArr = None
        self.rewOmit = rewOmit
        self.cue_dict = self.create_cue_dict()

    def create_cue_dict(self) -> dict:
        """Create a dictionary of specific cues from treadBehDict.

        Returns:
            dict: A dictionary containing specific cues from treadBehDict.
        """
        # List of cue types, interest defined by containing key "Lap"
        cue_types = self.cueTypes_keylist

        # Creating cue_dict dynamically
        self.cue_dict = {}
        for cue_type in cue_types:
            if cue_type != self.LCDkey["REW"]:
                # Handle CUE1 and CUE2, which are under the 'CUE_EVENTS' key
                self.cue_dict[cue_type] = self.treadBehDict[self.TDMLkey["CUE_EVENTS"]][
                    self.TDMLkey[cue_type]
                ]
            elif cue_type == self.LCDkey["REW"]:
                self.cue_dict[cue_type] = self.treadBehDict[self.TDMLkey["REW_ZONE"]]
            # else:
            #     # For other cue types
            #     self.cue_dict[cue_type] = self.treadBehDict[cue_type]

        return self.cue_dict

    def cueLapsProcessor(self, cueTypes4LCD: list = []) -> dict:
        """
        Main processor function for cueLaps.

        This function performs the following tasks:
        1) Determines the maximum cue based on the lap number.
        2) Finds the coarse numLapType number and binned position for the maximum cue.
        3) Processes cue events.

        Parameters:
            cueTypes4LCD (list): List of cue types for processing. By default, it is an empty list.

        Returns:
            dict: The lapCueDict dictionary containing the processed data.

        Note:
            - The cueTypes4LCD list can be changed by providing a user list input.
            - The lapCueDict dictionary is updated with the processed data.

        """
        # by default will be empty, but can change with user list input
        self.cueTypes4LCD = cueTypes4LCD
        # find cue type with max trials & fill corresponding max_cue key entry of lCD
        # fill in self.cueTypes4LCD if not set before self._fill_cueTypeData_in2LCD
        self.lapCueDict = self._fill_cueTypeData_in2LCD(base_zero_shift=True)

        # find numLapTypes & BinnedPos for max cue
        # (
        #     self.numLapTypes,
        #     self.lapCueDict[LCDkey["MAX_CUE"]][LCDkey["BINNEDPOS"]],
        # ) = self.LapType_NumNBinnedPos(
        #     self.lapCueDict[LCDkey["MAX_CUE"]][LCDkey["POSITION_KEY"]][
        #         LCDkey["START"]
        #     ]
        # )

        for cueType in self.cueTypes4LCD:
            self.lapCueDict[cueType][self.LCDkey["BINNEDPOS"]] = (
                self.LapType_NumNBinnedPos(
                    self.lapCueDict[cueType][self.LCDkey["POSITION_KEY"]][
                        self.LCDkey["START"]
                    ]
                )
            )

        self.lapCueDict = self.cLP_cueEvents()

        return self.lapCueDict

    def cLP_cueEvents(self) -> dict:
        """
        Find cueLap derived from lapEpochs.
        Find lapBins by cueType.
        Create cueLapCode.
        Adjust numLapType based on cueLapCode output.

        Returns:
            dict: The updated lapCueDict containing cueLap, lapBins, cueLapCode, and adjusted numLapTypes.
        """
        # sometimes cant rely on cueLap data bc RFID may have missed lap
        # so reconstruct from times
        # involves comparison w/in func & use method which leads to higher lap count
        # will print note about what cueLap will be
        self.lapCueDict = self.cueLap_via_lapEpoch()
        # create lapBins based on cueType
        self.lapCueDict = self.cueLapBins_via_cueTypeTimes()
        # create code based on cueLapBins
        # originally lapTypeArr in matlab
        # also integrates rew vs nonrew trial laptypearr tag
        self.lapCueDict = self.cueLapCode()
        # adjust numLapTypes based on cueLapCode() output
        # finds max numLapTypes
        # > 2 zeros => another numLapType is created
        # -- 0 => max numLapType + 1
        self.lapCueDict = self.PostCueLapCode_numLapType()

        return self.lapCueDict

    def cueORrewLap(self, cueTime: list) -> list:
        """
        Finds the cue or reward lap using the specified cueTime.

        Parameters:
            cueTime (list): A list of cue times.

        Returns:
            lap_to_find (list): A list of lap indices corresponding to the cue times.

        This method uses the `adjFrTimes` and `lapEpochs` as a frame of reference to determine the lap associated with each cue time.
        It iterates through each cue time and checks if it falls within the time range of each lap.
        If a cue time falls within a lap, the corresponding lap index is stored in the `lap_to_find` list.
        If a cue time does not fall within any lap, the corresponding lap index is set to `None`.

        Note: The `adjFrTimes` and `lapEpochs` should be initialized before calling this method.
        """

        lap_to_find = [None] * len(cueTime)
        numLaps = self.numLaps_Epochs
        for i in range(len(cueTime)):
            for j in range(numLaps):
                start_idx = self.lapEpochs[j][0]
                end_idx = self.lapEpochs[j][1]
                if start_idx < len(self.adjFrTimes) and end_idx < len(self.adjFrTimes):
                    if (
                        cueTime[i] >= self.adjFrTimes[start_idx]
                        and cueTime[i] <= self.adjFrTimes[end_idx]
                    ):
                        lap_to_find[i] = j

        return lap_to_find

    def cueLap_via_lapEpoch(self) -> dict:
        """
        Finds the number of cue laps via cueORrewLap.
        Compares cue laps to the maximum cue lap number derived via RFID.
        If the number of cue laps is larger than the RFID method, it cues laps via epoch.

        Returns:
            dict: The updated lapCueDict with the maximum cue lap number updated if necessary.
        """

        cueTime = self.lapCueDict[self.max_cue_type][self.LCDkey["TIME_KEY"]]
        cueLap = self.cueORrewLap(cueTime=cueTime)
        cueLap_arr_viaMaxCue = self.lapCueDict[self.max_cue_type][
            self.LCDkey["LAP_NUM_KEY"]
        ]
        cue_arrays_equal = np.array_equal(
            cueLap_arr_viaMaxCue,
            cueLap,
        )

        if not cue_arrays_equal and len(cueLap) > len(cueLap_arr_viaMaxCue):
            print("|-- Using cueLap derived from lapEpochs")
            self.lapCueDict[self.LCDkey["MAX_CUE"]][self.LCDkey["LAP_NUM_KEY"]] = cueLap
        else:
            print("|-- Laps derived from RFID & lapEpochs are equivalent")

        return self.lapCueDict

    def cueLapBins_via_cueTypeTimes(
        self, edge_bounds: list = [0, 2001], bin_size: int = 100
    ) -> dict:
        """
        Determine cueLapBins using cueTimes based on cueType.

        Parameters:
            edge_bounds (list, optional): The bounds of the edges for the cueType laps. Defaults to [0, 2001].
            bin_size (int, optional): The size of each bin. Defaults to 100.

        Returns:
            dict: A dictionary containing the cueType lap bins.
        """
        # see __init__ to see default cuetypes_of_interest
        cuetypes_of_interest = self.cueTypes4LCD
        self.edges4cueTypeLaps = np.arange(edge_bounds[0], edge_bounds[-1], bin_size)

        for cue in cuetypes_of_interest:
            self.lapCueDict = self.cueTypeLapBins_by_type(cueType=cue)

        return self.lapCueDict

    def cueTypeLapBins_by_type(self, cueType: str) -> dict:
        """Finds cueLap Binned pos by type.

        Parameters:
            cueType (str): The type of cue.

        Returns:
            dict: The updated lapCueDict.

        """

        cueTimes = self.cue_dict[cueType][self.LCDkey["START"]][self.LCDkey["TIME_KEY"]]
        cuePos = self.cue_dict[cueType][self.LCDkey["START"]][
            self.LCDkey["POSITION_KEY"]
        ]
        lapEpochs = self.lapEpochs
        adjFrTimes = self.adjFrTimes
        edges = self.edges4cueTypeLaps
        numLaps = self.numLaps_Epochs
        if cueTimes.any():
            BinnedLap = np.full(numLaps, np.nan)
            # Assign cue positions to laps
            for i, time in enumerate(cueTimes):
                for j, (start, end) in enumerate(lapEpochs):
                    if adjFrTimes[start] <= time <= adjFrTimes[end]:
                        BinnedLap[j] = cuePos[i]

            # Histogram binning
            # N, bin_edges = np.histogram(cuePosLap, bins=edges)
            bin_indices = np.digitize(BinnedLap, edges)
            bin_indices[np.isnan(BinnedLap)] = 0
            locs = np.unique(bin_indices)
            locs = locs[locs != 0]

            # Remap bins
            for i, loc in enumerate(locs, start=1):
                bin_indices[bin_indices == loc] = i

            self.lapCueDict[cueType][self.LCDkey["LAP_BIN_KEY"]] = bin_indices
            self.lapCueDict[cueType][self.LCDkey["LAP_LOC_KEY"]] = locs

        return self.lapCueDict

    def cueLapCode(self) -> dict:
        """
        from lap binned pos, create a code for each lap given cues that occur during given lap

        Returns:
            dict: The updated lapCueDict with lapTypeArr added.
        """
        cue_types = self.cueTypes4lapCode  # set this order here for it is imp!!!
        cueByTen = {
            "CUE1": 10**0,
            "CUE2": 10**1,
            "LED": 10**2,
            "TONE": 10**3,
            "TACT": 10**4,
            "OPTO": 10**5,
        }
        # cueLapCode org:
        # cue1  = 10^0
        # cue2  = 10^1
        # led   = 10^2
        # tone  = 10^3
        # tact  = 10^4
        # opto  = 10^5
        self.power_of_ten_dict = {cue: cueByTen[cue] for cue in cue_types}

        # Initialize cueLapCode
        cueLapCode = []

        # Find the longest array among the cue types
        max_length = max(
            len(self.lapCueDict[cue][self.LCDkey["LAP_BIN_KEY"]]) for cue in cue_types
        )

        # Calculate cueLapCode
        for i in range(max_length):
            code = 0
            for cue in cue_types:
                lapBin = self.lapCueDict[cue][self.LCDkey["LAP_BIN_KEY"]]
                # Check if i is within the bounds of the array and the array is not empty
                if i < len(lapBin) and lapBin.size > 0:
                    code += lapBin[i] * self.power_of_ten_dict[cue]
            cueLapCode.append(code)

        # Calculate lapTypeCodes and lapTypeArr
        cueLapCode = np.array(cueLapCode)
        lapTypeCodes = np.unique(cueLapCode)
        lapTypeCodes = lapTypeCodes[lapTypeCodes != 0]
        # Remap lapTypeCodes to a continuous range
        lapTypeArr = np.zeros_like(cueLapCode)
        for i, code in enumerate(lapTypeCodes, start=1):
            lapTypeArr[cueLapCode == code] = i

        # find no rewards trials
        lapsNoRew = self.rewLapType()

        # tag no reward trials, leave reward trials unchanged
        # if rewTimes are found
        if self.rewOmit:
            if lapsNoRew:
                for lap in lapsNoRew:
                    lapTypeArr[lap - 1] = self.numLapTypes + 1

        # fill in lCD & lapTypeArr (latter needed for easier code use later)
        self.lapTypeArr = lapTypeArr
        self.lapCueDict[self.LCDkey["LAP_KEY"]][self.LCDkey["LAP_TYPEARR"]] = lapTypeArr

        return self.lapCueDict

    def LapType_NumNBinnedPos(
        self, cuePosStart: np.ndarray, diff_thresh: int = 20, laptype_thresh: int = 15
    ) -> list:
        """
        Coarse measure to find the number of lap types and find binned positions for the maximum cue type.

        Parameters:
            cuePosStart (array-like): The starting positions of cues.
            diff_thresh (int, optional): The threshold for the maximum difference between cue positions. Defaults to 20.
            laptype_thresh (int, optional): The threshold for the number of lap types. Defaults to 15.

        Returns:
            lapTypeBinnedPos (list): The binned positions for the maximum cue type.

        """
        lapTypeBinnedPos = []
        numLaps = self.numLaps_RFID
        if max(np.diff(cuePosStart)) > diff_thresh:
            N, edges = np.histogram(cuePosStart, bins="auto")
            numLapTypes = len(N[N > numLaps / laptype_thresh])

            N_nLT, edges_nLT = np.histogram(cuePosStart, bins=numLapTypes)
            bin_indices = np.digitize(cuePosStart, edges_nLT) - 1
            highest_valid_index = len(edges_nLT) - 2
            bin_indices[bin_indices == (len(edges_nLT) - 1)] = highest_valid_index

            for bi in np.unique(bin_indices):
                bin_cuePos = cuePosStart[bin_indices == bi]
                if bin_cuePos.size > 0:
                    lapTypeBinnedPos.append(round(np.mean(bin_cuePos)))
                else:
                    lapTypeBinnedPos.append(np.nan)
        else:
            lapTypeBinnedPos = [round(np.mean(cuePosStart))]

        return lapTypeBinnedPos

    def LapEpochsNInds(
        self, height: float = 0.5, distance: int = 50, toPlot: bool = False
    ) -> dict:
        """
        Find lapEpochs and FrInds.

        Parameters:
            height (float): The minimum peak height for finding peaks in the signal. Defaults to 0.5.
            distance (int): The minimum peak distance for finding peaks in the signal. Defaults to 50.
            toPlot (bool): Whether to plot the lap epochs. Defaults to False.

        Returns:
            dict: A dictionary containing the lapFrInds and lapEpochs.
        """

        norm_resampY = self.resampY / max(self.resampY)
        diff_nrY = np.diff(norm_resampY, append=0)
        inverted_diff_nrY = -diff_nrY

        # Use find_peaks with minimum peak height and minimum peak distance
        pks_indices, _ = find_peaks(inverted_diff_nrY, height=height, distance=distance)
        # for last lap (partial lap)
        pks_indices = np.append(pks_indices, len(self.resampY))

        self.lapFrInds = pks_indices
        # find lapEpochs & also fill numLaps_Epochs variable
        self.lapEpochs = self.lapEpochProcessor(pks_indices, len(self.resampY))

        if toPlot:
            self.plot_laps(norm_resampY, pks_indices)

        return {
            self.LCDkey["LAP_FRIDX"]: self.lapFrInds,
            self.LCDkey["LAP_EPOCH"]: self.lapEpochs,
        }

    def lapEpochProcessor(self, pks_indices: list, y_length: int) -> list:
        """
        Process the peak indices and y_length to create lap epochs.

        Parameters:
            pks_indices (list): A list of peak indices.
            y_length (int): The length of the y data.

        Returns:
            list: A list of lap epochs, where each lap epoch is represented as a list [start_index, end_index].
        """

        lapEpochs = []
        lapEpochs.append([0, pks_indices[0]])

        # Iterating over the peak indices to create epochs
        for i in range(1, len(pks_indices)):
            lapEpochs.append([pks_indices[i - 1] + 1, pks_indices[i]])

        # Adjusting the last epoch
        # Set the last start index to the frame following the last peak
        last_start_index = pks_indices[-1] + 1

        # Ensure the last epoch covers frames up to the end of y_length
        if last_start_index < y_length:
            lapEpochs.append([last_start_index, y_length - 1])
        else:
            # Adjust the end of the second-to-last epoch if the last peak is at the end of y_length
            lapEpochs[-1][1] = y_length - 1

        # fill in self.numLaps_Epoch
        self.numLaps_Epochs = len(lapEpochs)

        return lapEpochs

    def _fill_cueTypeData_in2LCD(self, base_zero_shift: bool = True) -> dict:
        """
        Start filling in cueType data into lapCueDict.

        Parameters:
            base_zero_shift (bool, optional): Whether to shift laps to start from 1 instead of 0. Defaults to True.

        Returns:
            dict: The updated lapCueDict containing cueType data.
        """
        # if cueTypes4LCD is not set, only use cue types which contain "Lap"
        if not self.cueTypes4LCD:
            self.cueTypes4LCD = [
                key
                for key in self.cue_dict
                if self.TDMLkey["LAP_NUM_KEY"] in self.cue_dict[key]
            ]
        self.cuedict_CT_only = {key: self.cue_dict[key] for key in self.cueTypes4LCD}
        self.max_cue_type = max(
            self.cuedict_CT_only,
            key=lambda cue: len(
                self.cuedict_CT_only[cue][self.TDMLkey["START"]][
                    self.TDMLkey["TIME_KEY"]
                ]
            ),
        )
        self.lapCueDict = self.appender.add_cue(self.cuedict_CT_only, cue_type=True)
        if base_zero_shift:
            # Shift laps to start from 1 instead of 0
            for cueType in self.cueTypes4LCD:
                self.lapCueDict[cueType][self.LCDkey["LAP_NUM_KEY"]] += 1

        return self.lapCueDict

    def rewLapType(self) -> list:
        """
        Finds non-reward trials to mark lapTypeArr with numLapTypes + 1
        Reward trials are left unchanged

        Returns:
            list: A list of lap numbers where rewards did not occur
        """
        lapsNoRew = []
        rewTime = self.cue_dict[self.LCDkey["REW"]][self.LCDkey["START"]][
            self.LCDkey["TIME_KEY"]
        ]
        if rewTime.any():
            numLaps = self.numLaps_Epochs

            rewLaps = self.cueORrewLap(rewTime)
            lapsTemp = list(range(1, numLaps))

            rewLapArr = [1] * numLaps

            # Marking the laps where rewards occurred
            for lap in rewLaps:
                if lap is not None and isinstance(lap, int):
                    rewLapArr[lap] = 0

            # Filtering out the laps where rewards did not occur
            lapsNoRew = [lap for lap, val in zip(lapsTemp, rewLapArr) if val != 0]

        return lapsNoRew

    def PostCueLapCode_numLapType(self, numOmit_threshold: int = 2) -> dict:
        """
        After cueLap code is created, find max_numLapTypes & create an additional LapType
        (max_num + 1) for any 0's that still exist (if # > 0)

        Parameters:
            numOmit_threshold (int): The threshold value for the number of 0's in lapTypeArr. If the number of 0's exceeds this threshold, a new LapType will be created.

        Returns:
            dict: The updated lapCueDict dictionary with the maximum number of LapTypes and the modified lapTypeArr.
        """

        max_numLapTypes = max(np.unique(self.lapTypeArr))

        numOmit = numOmit = np.sum(self.lapTypeArr == 0)
        if numOmit > numOmit_threshold:
            max_numLapTypes += 1
            self.lapTypeArr[self.lapTypeArr == 0] = max_numLapTypes

        self.lapCueDict[self.LCDkey["NUM_LAP_TYPES"]] = max_numLapTypes
        self.lapCueDict[self.LCDkey["LAP_KEY"]][self.LCDkey["LAP_TYPEARR"]] = (
            self.lapTypeArr
        )

        return self.lapCueDict

    def plot_laps(self, y: list, lapFrInds: list) -> None:
        """
        Plot lapFrInds to check if done properly

        Parameters:
            y (list): The normalized resampled Y values.
            lapFrInds (list): The indices of the lap peaks.
        """

        plt.figure()
        plt.plot(y, label="Normalized Resampled Y")
        plt.plot(lapFrInds, [y[ind] for ind in lapFrInds], "r*", label="Lap Peaks")
        plt.xlabel("Frames")
        plt.ylabel("Normalized Value")
        plt.title("Lap Peaks in Data")
        plt.legend()
        plt.show()


######################################################
#  appender funcs
######################################################


class Appender4lapCueDict:
    """
    A class for appending cue data to lapCueDict.

    Attributes:
        lapCueDict (dict): The lapCueDict to which the cue data will be appended.
        cue_dict (dict): The cue data to be appended.

    Methods:
        add_cue(cue_dict, cue_type=False): Appends the cue data to lapCueDict.
        add_cue_info(): Fills in lapCueDict["max_cue"] with the cue information.
    """

    def __init__(self, lapCueDict: dict, LCDkey: dict, TDMLkey: dict) -> None:
        self.lapCueDict = lapCueDict
        self.LCDkey = LCDkey
        self.TDMLkey = TDMLkey
        self.cue_dict = None

    def add_cue(self, cue_dict: dict, cue_type: bool = False) -> dict:
        """
        Appends the cue data to lapCueDict.

        Parameters:
            cue_dict (dict): The cue data to be appended.
            cue_type (bool, optional): Indicates if the cue type should be adjusted for max_cue_type. Defaults to False.

        Returns:
            dict: The updated lapCueDict.
        """
        self.cue_dict = cue_dict
        if cue_type:
            self.add_cue_info()
        else:
            for key in cue_dict:
                self.lapCueDict[self.LCDkey["LAP_KEY"]][key] = cue_dict[key]
        return self.lapCueDict

    def add_cue_info(self) -> dict:
        """
        Fills in lapCueDict["max_cue"] with the cue information.

        Returns:
            dict: The updated lapCueDict.
        """
        for cueType in self.cue_dict:
            for key in self.lapCueDict[cueType]:
                if key == self.LCDkey["POSITION_KEY"]:
                    for subkey in self.lapCueDict[cueType][key]:
                        self.lapCueDict[cueType][key][subkey] = np.array(
                            self.cue_dict[cueType][subkey][self.TDMLkey["POSITION_KEY"]]
                        )
                elif key == self.LCDkey["TIME_KEY"]:
                    self.lapCueDict[cueType][key] = np.array(
                        self.cue_dict[cueType][self.TDMLkey["START"]][
                            self.TDMLkey["TIME_KEY"]
                        ]
                    )
                elif key == self.LCDkey["LAP_NUM_KEY"]:
                    self.lapCueDict[cueType][key] = np.array(
                        self.cue_dict[cueType][self.TDMLkey["LAP_NUM_KEY"]]
                    )
        return self.lapCueDict

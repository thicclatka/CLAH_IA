import copy
import time

import numpy as np
from scipy.ndimage import convolve

try:
    from scipy.signal import gaussian
except:
    from scipy.signal.windows import gaussian

from CLAH_ImageAnalysis.PlaceFieldLappedAnalysis import PCLA_dependencies
from CLAH_ImageAnalysis.PlaceFieldLappedAnalysis import PCLA_enum
from CLAH_ImageAnalysis.utils.enum_utils import enum2dict

pcl_str = enum2dict(PCLA_enum.TXT)


class PCLA_utils:
    """
    Utility class for Place Field Lapped Analysis (PCLA).

    Parameters:
        treadPos (numpy.ndarray): Array of tread positions.
        timeSeg (numpy.ndarray): Array of time segments.

    Attributes:
        treadPos (numpy.ndarray): Array of tread positions.
        timeSeg (numpy.ndarray): Array of time segments.
        lapInts (numpy.ndarray): Array of lap intervals.
        lapVec (numpy.ndarray): Array of lap vectors.
        gauss_kernel (numpy.ndarray): Gaussian kernel for smoothing.
        LSV (numpy.ndarray): Linearly Spaced Vector.
        dependencies (PCLA_dependencies): Instance of PCLA_dependencies class.
        dTSeg (float): Median of time segment differences.
        TransVectorLapOutput (dict): Dictionary to store intermediate results.
        RatePerc (float): Rate percentage.

    Methods:
        add_TVLO_to_utils(TransVectorLapOutput): Adds TransVectorLapOutput to utils.
        _init_TVLO4shuffle(): Initializes output structure for shuffle position vector vars.
        return_filled_TransVectorLapOutput(): Returns the filled TransVectorLapOutput.
        find_lapVecNInts(d1_thresh, bins): Finds lap vectors and intervals.
        find_Epochs(minVel, maxVel, minEpochDur, Moving): Finds epochs for moving or still.
        find_runTimes(minVel, minRunTime, trimRunStarts, trimRunEnds): Finds run times.
        runTimes_adjuster(runTimes, minRunTime, trimRunStarts, trimRunEnds, histo_correction): Adjusts run times.
        fill_gkernel_N_LSV(kernel_length, sigma, LSV_start, LSV_stop, LSV_num, LSV_end): Fills in gaussian kernel and LSV.
        find_nanedNvalidPos(spikes): Finds naned and valid positions.
        find_kRun(excludeVec): Finds kRun.
        find_rawOccupancy(): Finds raw occupancy.
        find_rawSums_N_PosRates(spikes): Finds raw sums and position rates.
        smooth_Occupancy_N_posRates(): Smooths occupancy and position rates.
    """

    def __init__(self, treadPos: np.ndarray, timeSeg: np.ndarray) -> None:
        self.treadPos = treadPos.copy()
        self.timeSeg = timeSeg.copy()
        self.lapInts = []
        self.lapVec = np.full(len(self.treadPos), np.nan)
        self.gauss_kernel = []
        self.LSV = []
        self.dependencies = PCLA_dependencies()
        self.dTSeg = np.median(np.diff(self.timeSeg))
        np.random.seed(int(time.time()))

    def add_TVLO_to_utils(self, TransVectorLapOutput: dict) -> None:
        """
        Adds TransVectorLapOutput to utils.

        Parameters:
            TransVectorLapOutput (dict): TransVectorLapOutput dictionary.

        Returns:
            None
        """
        # need to ensure TVLO in utils is empty dict before using further utils
        self.TransVectorLapOutput = {}
        self.TransVectorLapOutput = TransVectorLapOutput
        self.RatePerc = self.TransVectorLapOutput["PARAMS"][pcl_str["RATEPERC"]]

    def _init_TVLO4shuffle(self) -> None:
        """
        Initializes output structure based on shuffle position vector vars.

        Returns:
            None
        """
        # Initialize output structure based on shuffle position vector vars
        shape_to_use = self.ratesR.shape[0]
        init_zero = np.zeros(shape_to_use, dtype=int)
        init_nan = np.full(shape_to_use, np.nan)
        init_empty = [[] for _ in range(shape_to_use)]

        self.TransVectorLapOutput[pcl_str["SHUFF"]] = {}
        for key in [pcl_str["POS_PFIN"], pcl_str["POS_PFIN_ALL"]]:
            self.TransVectorLapOutput[pcl_str["SHUFF"]][key] = init_empty.copy()
        for key in [pcl_str["POS_PFPK"], pcl_str["PKRATE"]]:
            self.TransVectorLapOutput[pcl_str["SHUFF"]][key] = init_nan.copy()

        self.TransVectorLapOutput[pcl_str["SHUFF"]][pcl_str["ISPC"]] = init_zero.copy()
        self.TransVectorLapOutput[pcl_str["SHUFF"]][pcl_str["RATEPERC"]] = self.RatePerc
        self.TransVectorLapOutput[pcl_str["SHUFF"]][pcl_str["WHICHLAP"]] = (
            self.whichLap.copy()
        )

    def return_filled_TransVectorLapOutput(self) -> dict:
        """
        Returns the filled TransVectorLapOutput.

        Returns:
            dict: Filled TransVectorLapOutput.
        """
        return self.TransVectorLapOutput

    def find_lapVecNInts(
        self, d1_thresh: float = -0.5, bins: np.ndarray = np.arange(0, 1.01, 0.01)
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Finds lap vectors and intervals.

        Parameters:
            d1_thresh (float): Threshold for differences in treadPos.
            bins (numpy.ndarray): Array of bin edges for histogram.

        Returns:
            tuple: Tuple containing lap vectors and intervals.
        """
        # Calculate differences in treadPos
        d1 = np.diff(self.treadPos)
        d1 = np.insert(d1, 0, 0)
        d1 = np.append(d1, 0)

        # Find indices where treadPos changes sharply
        ints1, _ = self.dependencies.suprathresh(np.diff(self.treadPos), d1_thresh)
        ints1[ints1 > len(self.treadPos) - 1] = len(self.treadPos) - 1

        start_indices = ints1[:, 0].tolist()
        end_indices = ints1[:, 1].tolist()

        # Filter intervals based on time differences
        tInt1 = [
            (self.timeSeg[start], self.timeSeg[end])
            for start, end in zip(start_indices, end_indices)
        ]

        tInt1 = np.array(tInt1)

        valid = np.diff(tInt1, axis=1).flatten() > 0
        ints1 = ints1[valid]
        tInt1 = tInt1[valid]
        ints1[:, 1] = ints1[:, 1] + 1

        # Process intervals to determine laps
        c = 1
        for i in range(len(ints1)):
            h, _ = np.histogram(self.treadPos[ints1[i, 0] : ints1[i, 1] + 1], bins=bins)
            if np.mean(h > 0) >= 0.5:
                self.lapInts.append(tInt1[i])
                self.lapVec[ints1[i, 0] : ints1[i, 1] + 1] = [c] * (
                    ints1[i, 1] + 1 - ints1[i, 0]
                )
                c += 1

        self.lapInts = np.array(self.lapInts)
        self.lapVec = np.array(self.lapVec)

        return self.lapVec, self.lapInts

    def find_Epochs(
        self,
        minVel: float | None = None,
        maxVel: float = 3,
        minEpochDur: float = 3,
        Moving: bool = True,
    ) -> np.ndarray:
        """
        Finds epochs for either moving or still.

        Parameters:
            minVel (float | None, optional): Minimum velocity threshold.
            maxVel (float, optional): Maximum velocity threshold.
            minEpochDur (float, optional): Minimum epoch duration.
            Moving (bool, optional): Flag indicating whether to find moving or still epochs.

        Returns:
            numpy.ndarray: Array of epochs.
        """
        """Find epochs for either moving or still"""
        Epochs = []
        treadPos = self.treadPos.copy()
        treadPos = treadPos * 2
        treadPos = (0.5 - treadPos) * np.pi * 2
        timeSeg = self.timeSeg

        if Moving:
            timeSeg = self.dependencies.transposer(timeSeg, greater=True).copy()

        fs = 1 / np.mean(np.diff(timeSeg))
        # TODO create gaussian func in filter_utils.py in dependencies
        sigma = int(round(fs / 2))
        kernel_length = sigma * 3
        gauss_kernel = gaussian(kernel_length, sigma)
        gauss_kernel /= gauss_kernel.sum()

        d1 = self.dependencies.angleDiff(treadPos[:-1], treadPos[0:])
        d1 = 200 * abs(d1 / (2 * np.pi))

        d1 = self.dependencies.convolveWithTrim(d1.T, gauss_kernel)

        velocity = abs(d1) * fs
        velocity.append(0)

        if Moving:
            seg, segLength = self.dependencies.suprathresh(velocity, minVel)
        else:
            seg, segLength = self.dependencies.suprathresh(-1 * velocity, -1 * maxVel)

        # Extract start and end times from T based on indices in s
        Epochs = np.column_stack((timeSeg[seg[:, 0]], timeSeg[seg[:, 1]]))

        # Filter intervals by duration
        Epochs = Epochs[np.diff(Epochs, axis=1).flatten() > minEpochDur]

        return Epochs

    def find_runTimes(
        self, minVel: float, minRunTime: float, trimRunStarts: float, trimRunEnds: float
    ) -> None:
        """
        Finds run times.

        Parameters:
            minVel (float): Minimum velocity threshold.
            minRunTime (float): Minimum run time.
            trimRunStarts (float): Trim run starts.
            trimRunEnds (float): Trim run ends.

        Returns:
            None
        """
        if minVel == 0:
            min_timeSeg = min(self.timeSeg)
            max_timeSeg = max(self.timeSeg)
            adjuster = 1e-10
            self.runTimes = np.array(
                [
                    [min_timeSeg - adjuster, min_timeSeg],
                    [max_timeSeg, max_timeSeg + adjuster],
                ]
            )
            # self.runTimes = np.array([[min(self.timeSeg), max(self.timeSeg)]])
        else:
            if minVel > 0:
                # find moving epochs
                self.runTimes = self.find_Epochs(minVel=minVel, Moving=True)
            elif minVel < 0:
                # find still epochs
                self.runTimes = self.find_Epochs(
                    maxVel=-1 * minVel, minEpochDur=minRunTime, Moving=False
                )
            # adjust runTimes for histogram & w/trimRunStart & Ends
            self.runTimes = self.runTimes_adjuster(
                self.runTimes, minRunTime, trimRunStarts, trimRunEnds
            )

        self.runTimes = np.array(self.runTimes)

        self.TransVectorLapOutput[pcl_str["RUNTIME"]] = self.runTimes

    def runTimes_adjuster(
        self,
        runTimes: np.ndarray,
        minRunTime: float,
        trimRunStarts: float,
        trimRunEnds: float,
        histo_correction: float = 0.00001,
    ) -> np.ndarray:
        """
        Adjusts run times.

        Parameters:
            runTimes (numpy.ndarray): Array of run times.
            minRunTime (float): Minimum run time.
            trimRunStarts (float): Trim run starts.
            trimRunEnds (float): Trim run ends.
            histo_correction (float): Histogram correction.

        Returns:
            numpy.ndarray: Adjusted run times.
        """
        runTimes[:, 1] += histo_correction  # for histogram
        runTimes[:, 0] += trimRunStarts
        runTimes[:, 1] -= trimRunEnds
        # Filter intervals based on duration
        runTimes = runTimes[np.diff(runTimes, axis=1).flatten() >= minRunTime]

        return runTimes

    def fill_gkernel_N_LSV(
        self,
        kernel_length: int,
        sigma: float,
        LSV_start: float,
        LSV_stop: float,
        LSV_num: int,
        LSV_end: float,
    ) -> None:
        """
        Fills in gaussian kernel and LSV.

        Parameters:
            kernel_length (int): Length of the kernel.
            sigma (float): Standard deviation of the kernel.
            LSV_start (float): Start value of the Linearly Spaced Vector.
            LSV_stop (float): Stop value of the Linearly Spaced Vector.
            LSV_num (int): Number of points in the Linearly Spaced Vector.
            LSV_end (float): End value of the Linearly Spaced Vector.

        Returns:
            None
        """
        # fill in gaussian kernel
        self.gauss_kernel = gaussian(kernel_length, sigma)
        self.gauss_kernel /= self.gauss_kernel.sum()

        # Linearly Spaced Vector
        self.LSV = self.dependencies.create_linearly_spaced_vector(
            start=LSV_start, stop=LSV_stop, num=LSV_num, LSV_end=LSV_end
        )
        return

    def find_nanedNvalidPos(self, spikes: np.ndarray) -> np.ndarray:
        """
        Finds naned and valid positions.

        Parameters:
            spikes (numpy.ndarray): Array of spikes.

        Returns:
            numpy.ndarray: Array of spikes.
        """
        pos = self.treadPos.copy()
        pos[self.kRun <= 0] = np.nan
        actNaN = np.mean(np.isnan(spikes), axis=0) == 1
        pos[actNaN] = np.nan

        self.whichLap = self.lapVec[~np.isnan(pos)]
        valid_pos = ~np.isnan(pos)
        spikes = spikes[:, valid_pos]
        pos_valid = pos[valid_pos]

        self.TransVectorLapOutput[pcl_str["NA_POS"]] = pos
        self.TransVectorLapOutput["valid"] = pos_valid
        self.treadPos_valid = pos_valid

        return spikes

    def find_kRun(self, excludeVec: np.ndarray | None = None) -> None:
        """
        Finds kRun.

        Parameters:
            excludeVec (numpy.ndarray): Array of exclude vectors.

        Returns:
            None
        """
        self.kRun, _ = self.dependencies.process_intervalsNcounts(
            self.runTimes, self.timeSeg
        )
        self.kRun[self.lapVec <= 0] = 0
        if excludeVec is not None:
            self.kRun[excludeVec > 0] = 0

    def find_rawOccupancy(self) -> None:
        """
        Finds raw occupancy.

        Returns:
            None
        """
        rawOccupancy, edges = np.histogram(self.treadPos_valid, bins=self.LSV)
        self.whichPlace = np.digitize(self.treadPos_valid, edges, right=True) - 1
        rawOccupancy = rawOccupancy.astype(float)
        rawOccupancy[rawOccupancy == 0] = np.nan

        rawOccupancy = rawOccupancy * self.dTSeg

        self.rawOccupancy = rawOccupancy
        self.TransVectorLapOutput[pcl_str["RAW_OCCU"]] = rawOccupancy

    def find_rawSums_N_PosRates(self, spikes: np.ndarray) -> None:
        """
        Finds raw sums and position rates.

        Parameters:
            spikes (numpy.ndarray): Array of spikes.

        Returns:
            None
        """
        rawSums = np.zeros((spikes.shape[0], 100))
        for i in range(100):
            k = self.whichPlace == i
            if np.sum(k) > 0:
                rawSums[:, i] = np.nansum(spikes[:, k], axis=1)

        rawPosRates = rawSums / self.rawOccupancy
        rawPosRates[np.isnan(rawPosRates)] = 0
        # fill TVLP w/ rawPosRates w/NaNs
        self.TransVectorLapOutput[pcl_str["POSSUM"]] = rawSums
        self.rawSums = rawSums

        # fill TVLP w/ rawPosRates w/NaNs
        self.TransVectorLapOutput[pcl_str["POSRATERAW"]] = rawPosRates
        # rawPosRates has no NaNs
        rawPosRates[np.isnan(rawPosRates)] = 0
        self.rawPosRates = rawPosRates

    def smooth_Occupancy_N_posRates(self) -> None:
        """
        Smooths occupancy and position rates.

        Returns:
            None
        """
        self.TransVectorLapOutput[pcl_str["OCCU"]] = convolve(
            self.rawOccupancy, self.gauss_kernel, mode="wrap"
        )
        self.TransVectorLapOutput[pcl_str["POSRATE"]] = self.dependencies.convolve2D(
            self.rawPosRates, self.gauss_kernel[np.newaxis, :], shape="wrap"
        )

    def shuffle2findPlaceCells(self, spikes: np.ndarray, shuffN: int) -> None:
        """
        Shuffles positions to find significant threshold for identifying place cells.

        Parameters:
            spikes (list): List of spike data.
            shuffN (int): Number of shuffles to perform.

        Returns:
            None
        """
        self.shuffN = shuffN
        # Concatenate rates and positions for threshold analysis
        posRate = copy.deepcopy(self.TransVectorLapOutput[pcl_str["POSRATE"]])
        self.ratesR = np.concatenate(
            [posRate, posRate],
            axis=1,
        )
        self.concatenated_pos = np.concatenate([np.arange(0, 100), np.arange(0, 100)])

        # initalize the Shuff key of TVLO w/some arrays
        self._init_TVLO4shuffle()
        self.uLaps = np.unique(self.whichLap[~np.isnan(self.whichLap)])

        # shuffle pos to find sig threshold to find cells
        self.shufflePos_for_threshold(spikes)

        # iterates over found cells to determine if they are place cells
        self.PCA_loop()

    def shufflePos_for_threshold(self, spikes):
        """
        Shuffles the position vector and performs histogram counting and other operations to calculate thresholds and detect significant place fields.

        Parameters:
            spikes (numpy.ndarray): The spike data.

        Returns:
            None
        """
        init_zero = np.zeros((spikes.shape[0], 100, self.shuffN))
        self.TransVectorLapOutput["spikes"] = spikes
        ratesAllShuff = init_zero.copy()
        rawRatesAllShuff = init_zero.copy()
        rawSumsAll = init_zero.copy()
        circShuffLapN = np.zeros((len(self.uLaps), self.shuffN), dtype=int)

        for i, ulap in enumerate(self.uLaps):
            count = np.sum(self.whichLap == ulap)

            if count > 0:
                circShuffLapN[i, :] = np.random.randint(count, size=self.shuffN)
            else:
                circShuffLapN[i, :] = 0

        for sh in range(self.shuffN):
            treadPos_toShuffle = self.treadPos_valid.copy()
            # Shuffle the position vector
            for i, ulap in enumerate(self.uLaps):
                lap_indices = self.whichLap == ulap
                treadPos_toShuffle[lap_indices] = np.roll(
                    treadPos_toShuffle[lap_indices], circShuffLapN[i, sh]
                )

            # Perform histogram counting and other operations
            whichPlace = np.digitize(treadPos_toShuffle, self.LSV) - 1
            rawSums = np.zeros((spikes.shape[0], 100))

            for i in range(100):
                k = whichPlace == i
                if np.sum(k) > 0:
                    rawSums[:, i] = np.nansum(spikes[:, k], axis=1)

            rawSumsAll[:, :, sh] = rawSums

            rawPosRates = rawSums / self.TransVectorLapOutput[pcl_str["RAW_OCCU"]]

            # rawPosRates = rawSums / np.tile(
            #     self.TransVectorLapOutput[pcl_str["RAW_OCCU"]], (rawSums.shape[0], 1)
            # )
            rawRatesAllShuff[:, :, sh] = rawPosRates

            rawPosRates[np.isnan(rawPosRates)] = 0
            posRates = self.dependencies.convolve2D(
                rawPosRates, self.gauss_kernel[np.newaxis, :], "wrap"
            )
            ratesAllShuff[:, :, sh] = posRates

        # Fill TVLO with [Shuff][ratesAllShuff] = ratesAllShuff
        self.TransVectorLapOutput[pcl_str["SHUFF"]][pcl_str["RATEALLSHUFF"]] = (
            ratesAllShuff
        )

        # Calculate threshold and detect significant place fields
        percRate = np.percentile(
            ratesAllShuff,
            self.TransVectorLapOutput[pcl_str["SHUFF"]][pcl_str["RATEPERC"]],
            axis=2,
        )
        self.TransVectorLapOutput[pcl_str["SHUFF"]][pcl_str["SHUFF_MNRATE"]] = (
            np.nanmean(ratesAllShuff, axis=2)
        )
        self.TransVectorLapOutput[pcl_str["SHUFF"]][pcl_str["THRESHRATE"]] = percRate
        percRate = np.concatenate([percRate, percRate], axis=1)
        self.sigRate = (self.ratesR > percRate).astype(float)
        self.TransVectorLapOutput[pcl_str["SHUFF"]][pcl_str["SIGRATE"]] = self.sigRate
        self.TransVectorLapOutput[pcl_str["SHUFF"]][pcl_str["PERCRATE"]] = percRate
        self.pC = np.where(
            np.sum(self.sigRate, axis=1)
            >= self.TransVectorLapOutput["PARAMS"][pcl_str["MINPFBINS"]]
        )[0]

        self.TransVectorLapOutput[pcl_str["SHUFF"]][pcl_str["CIRC_SHUFFN"]] = (
            circShuffLapN
        )

    def PCA_loop(self) -> None:
        """
        Place cell analysis loop:
        - Uses sigRate from shufflePos_for_threshold (which relies on ratesR)
        - Iterates over cells (pC) to determine if they are indeed place cells:
            - Finds within set all bins that are above threshold & last for at least 5 bins (minPFBins)
        """
        for i in self.pC:
            sigBins, sigBinsL = self.dependencies.suprathresh(self.sigRate[i, :], 0.5)
            sigBins = sigBins[
                sigBinsL >= self.TransVectorLapOutput["PARAMS"][pcl_str["MINPFBINS"]], :
            ]
            sigBinsAll = []

            if sigBins.size > 0:
                # Concatenate significant bins
                for bin_range in sigBins:
                    sigBinsAll.extend(range(bin_range[0], bin_range[1] + 1))

                # Determine significant bins based on threshold
                sBins = (
                    self.ratesR[i, :]
                    > np.nanmean(self.ratesR[i, :])
                    * self.TransVectorLapOutput["PARAMS"][pcl_str["EDGE_R_MULT"]]
                )
                sBins = np.logical_or(sBins, self.sigRate[i, :])
                pfS, pfL = self.dependencies.suprathresh(sBins.astype(float), 0.5)
                pfS = pfS[
                    pfL >= self.TransVectorLapOutput["PARAMS"][pcl_str["MINPFBINS"]], :
                ]
                pfL = pfL[
                    pfL >= self.TransVectorLapOutput["PARAMS"][pcl_str["MINPFBINS"]]
                ]

                k = []
                k.extend(
                    [
                        (
                            1
                            if sum(np.isin(sigBinsAll, range(start, end + 1)))
                            >= self.TransVectorLapOutput["PARAMS"][pcl_str["MINPFBINS"]]
                            else 0
                        )
                        for start, end in pfS
                    ]
                )

                pfS = pfS[np.array(k) > 0, :]
                pfL = pfL[np.array(k) > 0]

                k = np.sum(pfS > 100, axis=1) < 2
                pfS = pfS[k, :]
                pfL = pfL[k]

                # Sort place fields by length and keep the largest
                s_idx = np.argsort(pfL)[::-1]
                pfS = pfS[s_idx, :]

                inPFAll = np.zeros(100, dtype=int)
                maxRate = []
                # Determine the position of place fields and their peak rates
                for start, end in pfS:
                    inPFIn = np.zeros(100, dtype=int)
                    if end > 100:
                        inPFIn[: end % 100] = 1
                        inPFIn[start:] = 1
                    else:
                        inPFIn[start : end + 1] = 1

                    if np.sum(inPFAll[inPFIn > 0]) == 0:
                        inPFAll[inPFIn > 0] = 1
                        pf_indices = np.where(inPFIn > 0)[0]
                        pkr_pos_arr = pf_indices
                        maxRate.append(np.max(self.ratesR[i, pf_indices]))

                if maxRate:
                    peak_rate_idx = np.argmax(maxRate)
                    specific_pkr_pos_idx = pkr_pos_arr[peak_rate_idx].copy()

                    posIn = self.concatenated_pos[specific_pkr_pos_idx]
                    ratesIn = self.ratesR[i, specific_pkr_pos_idx]
                    _, maxBin = np.max(ratesIn), np.argmax(ratesIn)
                    if not isinstance(posIn, np.ndarray):
                        maxPosIn = posIn
                    else:
                        maxPosIn = posIn[maxBin]

                    # Fill in TVLO dict with values
                    self.TransVectorLapOutput[pcl_str["SHUFF"]][pcl_str["POS_PFPK"]][
                        i
                    ] = maxPosIn
                    self.TransVectorLapOutput[pcl_str["SHUFF"]][pcl_str["PKRATE"]][
                        i
                    ] = maxRate[peak_rate_idx]
                    self.TransVectorLapOutput[pcl_str["SHUFF"]][pcl_str["ISPC"]][i] = 1
                    self.TransVectorLapOutput[pcl_str["SHUFF"]][pcl_str["POS_PFIN"]][
                        i
                    ] = specific_pkr_pos_idx
                    self.TransVectorLapOutput[pcl_str["SHUFF"]][
                        pcl_str["POS_PFIN_ALL"]
                    ][i] = pf_indices

    @staticmethod
    def find_InfoPerSpk_N_Sec(
        binned_rate: np.ndarray, binned_occu: np.ndarray
    ) -> tuple[float, float]:
        """
        Calculate information per spike and per second.

        Parameters:
            binned_rate (np.ndarray): binned rate of spikes.
            binned_occu (np.ndarray): binned occupancy.

        Returns:
        tuple: Tuple containing:
            - info_per_spike (float): information per spike.
            - info_per_sec (float): information per second.
        """

        # Normalizing the binned rate
        l1 = binned_rate / np.mean(binned_rate)
        kL = np.where(l1 > 0)[0]  # Indices where l1 is greater than 0

        # Normalized occupancy
        norm_occ_l = binned_occu / np.sum(binned_occu)

        # Information per spike
        info_per_spike = np.sum(l1[kL] * np.log2(l1[kL]) * norm_occ_l[kL])

        # Information per second
        info_per_sec = np.sum(binned_rate[kL] * np.log2(l1[kL]) * norm_occ_l[kL])

        return info_per_spike, info_per_sec

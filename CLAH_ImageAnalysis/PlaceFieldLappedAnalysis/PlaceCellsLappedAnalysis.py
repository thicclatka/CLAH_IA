"""
This module contains the `computePlaceCells` class, which is responsible for computing place cells based on spike data and treadmill position.

The `computePlaceCells` class has the following attributes:
- `spikes`: The spike data.
- `treadPos`: The treadmill position data.
- `timeSeg`: The time segments for analysis.
- `shuffN`: The number of shuffles for statistical analysis.

The `computePlaceCells` class has the following methods:
- `_find_mLapRatePFind_via_lap_rateNpf_indices`: Private method to find the mean lap rate per field index.
- `_postShuffle_PCSimple_init`: Private method to initialize the `PCSimple` dictionary for post-shuffle analysis.
- `_init_PCLSDict`: Private method to initialize the `PCLapSess` dictionary for place cell analysis.
- `_proc_PCLapSess`: Private method to process the `PCLapSess` dictionary for place cell analysis.
- `_proc_InfoPerSpk_N_Sec`: Private method to calculate information per spike and per second.
- `_proc_infoMetrics_ZnP`: Private method to calculate Z-scores and p-values for information metrics.
- `_calc_pfs_by_lap`: Private method to calculate place field strengths by lap.
- `_Lap_N_Bin_processor`: Private method to process lap and bin data.

Note: This code is part of a larger module for place field lapped analysis in CLAH Image Analysis.
"""

import copy
import time

import numpy as np
from CLAH_ImageAnalysis.core import BaseClass as BC
from CLAH_ImageAnalysis.PlaceFieldLappedAnalysis import PCLA_dependencies
from CLAH_ImageAnalysis.PlaceFieldLappedAnalysis import PCLA_enum
from CLAH_ImageAnalysis.PlaceFieldLappedAnalysis import PCLA_utils


class computePlaceCells(BC):
    """
    A class that represents the computation of place cells.

    Parameters:
        spikes (numpy.ndarray): Spike data for each segment and cell.
        treadPos (numpy.ndarray): Treadmill position data for each segment.
        timeSeg (numpy.ndarray): Time segments for each segment.
        shuffN (int): Number of shuffles to perform.

    Attributes:
        spikes (numpy.ndarray): Spike data for each segment and cell.
        numSeg (int): Number of segments.
        treadPos (numpy.ndarray): Treadmill position data for each segment.
        timeSeg (numpy.ndarray): Time segments for each segment.
        shuffN (int): Number of shuffles to perform.
        utils (PCLA_utils): Utility object for place cell analysis.
        dependencies (PCLA_dependencies): Dependency object for place cell analysis.
        dTSeg (float): Median time difference between segments.
        LSV (list): List of least squares vectors.
        gauss_kernel (list): List of Gaussian kernels.
        lapInts (list): List of lap intervals.
        lapVec (numpy.ndarray): Lap vector.
        TransVectorLapOutput (dict): Dictionary containing the output of TransVectorLapCircShuff_wEdges.
        minRunTime (list): List of minimum run times.
        runTimes (list): List of run times.
        kRun (list): List of run indices.
        whichLap (list): List of lap indices.
        treadPos_valid (list): List of valid treadmill positions.
        whichPlace (list): List of place indices.
        PCSimple (dict): Dictionary containing place cell analysis results.
        PCLapSess (dict): Dictionary containing lap-specific place cell analysis results.
        PC_ValidTimePoints (dict): Dictionary containing valid time points for place cell analysis.
        isPC (list): List of place cell indices.

    Methods:
        _find_mLapRatePFind_via_lap_rateNpf_indices(c_idx): Helper method to find mLapRatePFind via lap rate and pf indices.
        _postShuffle_PCSimple_init(): Helper method to initialize PCSimple with NaNs.
        _init_PCLSDict(): Helper method to initialize PCLapSess dictionary.
        _proc_PCLapSess(): Helper method to process PCLapSess.
        _proc_InfoPerSpk_N_Sec(): Helper method to calculate information per spike and per second.
        _proc_infoMetrics_ZnP(metric): Helper method to calculate Z-scores and P-values for a given metric.
        _calc_pfs_by_lap(): Helper method to calculate place fields by lap.
        _Lap_N_Bin_processor(): Helper method to process lap and bin data.
    """

    def __init__(
        self, spikes: np.ndarray, treadPos: np.ndarray, timeSeg: np.ndarray, shuffN: int
    ) -> None:
        """
        Initializes an instance of the PlaceCellsLappedAnalysis class.

        Parameters:
            spikes (numpy.ndarray): Array containing spike data.
            treadPos (numpy.ndarray): Array containing tread position data.
            timeSeg (numpy.ndarray): Array containing time segment data.
            shuffN (int): Number of shuffles.

        """
        # set random seed
        np.random.seed(int(time.time()))
        self.program_name = "PlaceCellsLappedAnalysis"
        self.class_type = "utils"
        BC.__init__(self, self.program_name, mode=self.class_type)

        # init self vars with passed in values
        self._initFilledVars(spikes, treadPos, timeSeg, shuffN)

        # init self vars to be filled
        self._initEmptyVars()

    def _initFilledVars(
        self, spikes: np.ndarray, treadPos: np.ndarray, timeSeg: np.ndarray, shuffN: int
    ) -> None:
        self.spikes = spikes
        self.numSeg = self.spikes.shape[0]
        self.treadPos = treadPos
        self.timeSeg = timeSeg
        self.shuffN = shuffN
        self.utils = PCLA_utils(treadPos=treadPos, timeSeg=timeSeg)
        self.dependencies = PCLA_dependencies()
        self.PCLAkey = self.enum2dict(PCLA_enum.TXT)
        self.dTSeg = np.median(np.diff(self.timeSeg))
        self.lapVec = np.full(len(self.treadPos), np.nan)

    def _initEmptyVars(self) -> None:
        # init self vars to be filled
        self.LSV = []
        self.gauss_kernel = []
        self.lapInts = []
        self.TransVectorLapOutput = {}
        self.minRunTime = []
        self.runTimes = []
        self.kRun = []
        self.whichLap = []
        self.treadPos_valid = []
        self.whichPlace = []
        self.PCSimple = {}
        self.PCLapSess = {}
        self.PC_ValidTimePoints = {}
        self.isPC = []

    def _find_mLapRatePFind_via_lap_rateNpf_indices(self, c_idx: int) -> np.ndarray:
        """
        Calculate the mean lap rate for place cells based on lap rate and place field indices.

        Parameters:
            c_idx (int): Index of the place cell.

        Returns:
            numpy.ndarray: Array containing the mean lap rate for each lap.

        """
        lap_rate = self.PCSimple[self.PCLAkey["BYLAP"]][self.PCLAkey["POSRATE"]][
            c_idx, :, :
        ].T
        lap_rate = lap_rate / np.nanmean(lap_rate, axis=1, keepdims=True)

        pf_indices = self.PCSimple[self.PCLAkey["SHUFF"]][self.PCLAkey["POS_PFIN_ALL"]][
            c_idx
        ]
        mLapRatePFind = np.nanmean(lap_rate[:, pf_indices], axis=1)
        return mLapRatePFind

    def _postShuffle_PCSimple_init(self) -> None:
        """
        Initializes the `_postShuffle_PCSimple` object.

        This method initializes the Info arrays with NaNs and initializes the InfoPerSpk and InfoPerSec structures.

        Parameters:
            self: The instance of the class.
        """
        # Initialize Info arrays with NaNs
        full_NaN_shuff = np.full((self.nCells, self.shuffN_postShuffle), np.nan)
        full_NaN = np.full(self.nCells, np.nan)

        # Initialize the InfoPerSpk and InfoPerSec structures
        info_struct = {
            self.PCLAkey["VAL"]: full_NaN_shuff.copy(),
            self.PCLAkey["ZSCORE"]: full_NaN.copy(),
            self.PCLAkey["PVAL"]: full_NaN.copy(),
        }

        self.PCSimple[self.PCLAkey["INFOSPK"]] = full_NaN.copy()
        self.PCSimple[self.PCLAkey["INFOSEC"]] = full_NaN.copy()
        self.PCSimple[self.PCLAkey["SHUFF"]][self.PCLAkey["INFOSPK"]] = (
            info_struct.copy()
        )
        self.PCSimple[self.PCLAkey["SHUFF"]][self.PCLAkey["INFOSEC"]] = (
            info_struct.copy()
        )

    def _init_PCLSDict(self) -> None:
        """
        Initializes the PCLapSess dictionary.

        This method makes a copy of the PCSimple dictionary and initializes
        the necessary fields with appropriate values.

        Returns:
            None
        """
        # Make a copy of PCSimple
        self.PCLapSess = copy.deepcopy(self.PCSimple)
        shape_to_use = self.PCSimple[self.PCLAkey["POSRATE"]].shape[0]
        init_zero = np.zeros(shape_to_use, dtype=int)
        init_nan = np.full(shape_to_use, np.nan)
        init_empty = [[] for _ in range(shape_to_use)]

        nLap = self.PCSimple[self.PCLAkey["BYLAP"]][self.PCLAkey["OCCU"]].shape[1]
        minL = round(nLap * self.lapPerc)
        self.minL = max(minL, self.lapMin)
        self.PCLapSess[self.PCLAkey["MINLAPSN"]] = self.minL

        for key in [
            self.PCLAkey["POS_PFPK"],
            self.PCLAkey["BESTFIELD"],
            self.PCLAkey["PKRATE"],
        ]:
            self.PCLapSess[self.PCLAkey["SHUFF"]][key] = init_nan.copy()

        self.PCLapSess[self.PCLAkey["SHUFF"]][self.PCLAkey["ISPC"]] = init_zero.copy()
        self.PCLapSess[self.PCLAkey["SHUFF"]][self.PCLAkey["LAPREL"]] = (
            init_empty.copy()
        )

    def _proc_PCLapSess(self) -> None:
        """
        Process the Place Cell Lapped Session.

        This method initializes the Place Cell Lapped Session dictionary, calculates various metrics for each place cell,
        and stores the results in the dictionary.

        Returns:
            None
        """
        self._init_PCLSDict()
        pos = np.arange(0, 100)  # Assuming 100 positions

        for idx, c_idx in enumerate(self.isPC):
            kFields = []
            mLapRatePFind = self._find_mLapRatePFind_via_lap_rateNpf_indices(c_idx)
            if np.sum(mLapRatePFind > 1) >= self.minL:
                kFields.append(1)
            else:
                kFields.append(0)

            if np.sum(kFields) > 0:
                self.PCLapSess[self.PCLAkey["SHUFF"]][self.PCLAkey["ISPC"]][c_idx] = 1
                valid_fields = [
                    self.PCSimple[self.PCLAkey["SHUFF"]][self.PCLAkey["POS_PFIN_ALL"]][
                        c_idx
                    ][j]
                    for j in range(len(kFields))
                    if kFields[j]
                ]
                self.PCLapSess[self.PCLAkey["SHUFF"]][self.PCLAkey["POS_PFIN_ALL"]][
                    c_idx
                ] = valid_fields
                valid_means = [
                    mLapRatePFind[j] for j in range(len(kFields)) if kFields[j]
                ]
                self.PCLapSess[self.PCLAkey["SHUFF"]][self.PCLAkey["LAPREL"]][c_idx] = (
                    valid_means
                )

                fieldRates, peakPosAll = [], []
                posIn = pos[valid_fields]
                ratesIn = self.PCLapSess[self.PCLAkey["POSRATE"]][c_idx, valid_fields]
                peakRate, peakPos = np.max(ratesIn), np.argmax(ratesIn)
                peakPos = posIn[peakPos]
                fieldRates.append(peakRate)
                peakPosAll.append(peakPos)

                bestField = np.argmax(fieldRates)
                self.PCLapSess[self.PCLAkey["SHUFF"]][self.PCLAkey["BESTFIELD"]][
                    c_idx
                ] = bestField
                self.PCLapSess[self.PCLAkey["SHUFF"]][self.PCLAkey["PKRATE"]][c_idx] = (
                    fieldRates[bestField]
                )
                self.PCLapSess[self.PCLAkey["SHUFF"]][self.PCLAkey["POS_PFPK"]][
                    c_idx
                ] = peakPosAll[bestField]
                self.PCLapSess[self.PCLAkey["SHUFF"]][self.PCLAkey["POS_PFIN"]][
                    c_idx
                ] = valid_fields[bestField]

    def _proc_InfoPerSpk_N_Sec(self) -> None:
        """
        Calculate information per spike and per second for each cell.

        This method calculates the information per spike and per second for each cell in the analysis.
        It iterates over the cells and calculates the information using the `find_InfoPerSpk_N_Sec` method
        from the `utils` module. The calculated information is then stored in the `PCSimple` data structure.

        Returns:
            None
        """
        for i in range(self.nCells):
            # Actual data
            infoSp, infoSec = self.utils.find_InfoPerSpk_N_Sec(
                self.PCSimple[self.PCLAkey["POSRATE"]][
                    i, self.PCSimple[self.PCLAkey["G_BINS"]]
                ],
                self.PCSimple[self.PCLAkey["OCCU"]][
                    self.PCSimple[self.PCLAkey["G_BINS"]]
                ],
            )
            self.PCSimple[self.PCLAkey["INFOSPK"]][i] = infoSp
            self.PCSimple[self.PCLAkey["INFOSEC"]][i] = infoSec

            # Shuffled data
            for sh in range(self.shuffN_postShuffle):
                infoSp, infoSec = self.utils.find_InfoPerSpk_N_Sec(
                    self.PCSimple[self.PCLAkey["SHUFF"]][self.PCLAkey["RATEALLSHUFF"]][
                        i, self.PCSimple[self.PCLAkey["G_BINS"]], sh
                    ],
                    self.PCSimple[self.PCLAkey["OCCU"]][
                        self.PCSimple[self.PCLAkey["G_BINS"]]
                    ],
                )
                self.PCSimple[self.PCLAkey["SHUFF"]][self.PCLAkey["INFOSPK"]][
                    self.PCLAkey["VAL"]
                ][i, sh] = infoSp
                self.PCSimple[self.PCLAkey["SHUFF"]][self.PCLAkey["INFOSEC"]][
                    self.PCLAkey["VAL"]
                ][i, sh] = infoSec

            # find Z & pValues & fill in PCSimple
            metric = [self.PCLAkey["INFOSPK"], self.PCLAkey["INFOSEC"]]
            for key in metric:
                self._proc_infoMetrics_ZnP(metric=key)

    def _proc_infoMetrics_ZnP(self, metric: str) -> None:
        """
        Calculates Z-scores and fills in P-values for the given metric
        ('InfoPerSpk' or 'InfoPerSec') within the Shuff key of PCSimple.

        Args:
            metric (str): The metric for which to calculate Z-scores and P-values.
                Can be either 'InfoPerSpk' or 'InfoPerSec'.

        Returns:
            None
        """
        for i in range(self.nCells):
            shuff_values = self.PCSimple[self.PCLAkey["SHUFF"]][metric][
                self.PCLAkey["VAL"]
            ][i, :]
            if shuff_values.size > 0 and not np.all(np.isnan(shuff_values)):
                # Calculate Z-score
                mean_shuff = np.nanmean(
                    self.PCSimple[self.PCLAkey["SHUFF"]][metric][self.PCLAkey["VAL"]][
                        i, :
                    ]
                )
                std_shuff = np.nanstd(
                    self.PCSimple[self.PCLAkey["SHUFF"]][metric][self.PCLAkey["VAL"]][
                        i, :
                    ]
                )
                self.PCSimple[self.PCLAkey["SHUFF"]][metric][self.PCLAkey["ZSCORE"]][
                    i
                ] = (self.PCSimple[metric][i] - mean_shuff) / std_shuff

                # Calculate P-value
                self.PCSimple[self.PCLAkey["SHUFF"]][metric][self.PCLAkey["PVAL"]][
                    i
                ] = np.mean(
                    self.PCSimple[self.PCLAkey["SHUFF"]][metric][self.PCLAkey["VAL"]][
                        i, :
                    ]
                    >= self.PCSimple[metric][i]
                )

    def _calc_pfs_by_lap(self) -> None:
        """
        Calculate place field strengths by lap.

        This method calculates the place field strengths for each place cell by lap.
        It stores the mean place field strengths and statistics in the `PCSimple` attribute.

        Returns:
            None
        """
        init_nan = np.full((len(self.isPC), 2), np.nan)
        init_empty = [[] for _ in range(len(self.isPC))]
        pfs_by_lap_means = init_empty.copy()
        pfs_by_lap_stats = init_nan.copy()
        for idx, c_idx in enumerate(self.isPC):
            mLapRatePFind = self._find_mLapRatePFind_via_lap_rateNpf_indices(c_idx)

            # Store mean place field strengths & statistics
            pfs_by_lap_means[idx] = mLapRatePFind
            pfs_by_lap_stats[idx, :] = [
                np.mean(mLapRatePFind > 1),
                np.sum(mLapRatePFind > 1),
            ]

        self.PCSimple[self.PCLAkey["BYLAP"]][self.PCLAkey["PFS_MN"]] = pfs_by_lap_means
        self.PCSimple[self.PCLAkey["BYLAP"]][self.PCLAkey["PFS_ST"]] = pfs_by_lap_stats

    def _Lap_N_Bin_processor(self) -> np.ndarray:
        """
        Process lap and bin data for place cell analysis.

        This method calculates various metrics related to lap and bin data for place cell analysis.
        It performs lap-specific calculations, identifies bins with insufficient occupancy across laps,
        assigns lap-specific metrics to PCSimple, and identifies good and bad bins.

        Returns:
            badPosition (ndarray): Boolean array indicating bad positions based on bad bins.
        """
        nanPos = copy.deepcopy(self.PC_ValidTimePoints[self.PCLAkey["NA_POS"]])
        self.lapVec[np.isnan(nanPos)] = np.nan
        self.lapU = np.unique(self.lapVec[~np.isnan(self.lapVec)])

        posRateByLap = np.zeros((self.numSeg, 100, len(self.lapU)))
        posRateRawByLap = np.zeros_like(posRateByLap)
        rawOccuByLap = np.full((100, len(self.lapU)), np.nan)
        OccuByLap = np.full((100, len(self.lapU)), np.nan)

        for lap_idx, lap_num in enumerate(self.lapU):
            # Exclude the current lap
            notLap = self.lapVec != lap_num
            pc = {}
            pc = self.TransVectorLapCircShuff_wEdges(
                shuff_override=0, excludeVec=notLap
            )

            # Store the lap-specific calculations
            posRateByLap[:, :, lap_idx] = pc[self.PCLAkey["POSRATE"]]
            posRateRawByLap[:, :, lap_idx] = pc[self.PCLAkey["POSRATERAW"]]
            rawOccuByLap[:, lap_idx] = pc[self.PCLAkey["RAW_OCCU"]]
            OccuByLap[:, lap_idx] = pc[self.PCLAkey["OCCU"]]

        # Calculate the minimum lap number
        nLap = OccuByLap.shape[1]
        minL = round(nLap * self.lapPerc)
        minL = max(minL, self.lapMin)

        # Find bins with insufficient occupancy across laps
        badBins = np.where(np.sum(rawOccuByLap > 0, axis=0) < minL)[0]

        # Create position bins and determine which bin each position belongs to
        posBins = np.linspace(0, 1, 101)
        posBins[-1] += 0.01
        whichBin = np.digitize(self.treadPos, posBins) - 1

        # Identify bad positions based on bad bins
        badPosition = np.isin(whichBin, badBins)

        self.print_wFrm(
            f"Number of bins w/ insufficient occupancy across laps: {sum(badPosition==1)}",
            frame_num=1,
        )
        # Assign lap-specific metrics to PCSimple
        self.PCSimple[self.PCLAkey["BYLAP"]] = {
            self.PCLAkey["POSRATE"]: posRateByLap,
            self.PCLAkey["POSRATERAW"]: posRateRawByLap,
            self.PCLAkey["RAW_OCCU"]: rawOccuByLap,
            self.PCLAkey["OCCU"]: OccuByLap,
        }

        # Assign badBins to PCSimple
        self.PCSimple[self.PCLAkey["B_BINS"]] = badBins.tolist()

        # Identify good bins (those not in badBins)
        bins = np.arange(1, 101)
        self.PCSimple[self.PCLAkey["G_BINS"]] = bins[~np.isin(bins, badBins)] - 1

        return badPosition

    def LappedWEdges(self, lapPerc: float = 0.20, lapMin: int = 2) -> dict:
        """
        Perform lapped analysis with edges.

        Args:
            lapPerc (float): The percentage of laps to consider as valid.
            lapMin (int): The minimum number of laps required for a position bin to be considered valid.

        Returns:
            dict: A dictionary containing the results of the lapped analysis.
        """
        self.rprint("Finding:")
        self.print_wFrm("lap vectors & intervals")
        self.lapVec, self.lapInts = self.utils.find_lapVecNInts()
        self.lapPerc = lapPerc
        self.lapMin = lapMin

        # no shuffle, just procuring valid time points
        self.print_wFrm("valid time points")
        self.PC_ValidTimePoints = self.TransVectorLapCircShuff_wEdges(shuff_override=0)

        # fills in respective entries for PCSimple dict
        # provides excludeVec for following shuffle procedure
        self.print_wFrm("bad position bins based on occupancy & min lap length")
        badPosition = self._Lap_N_Bin_processor()

        # run shuffle but exclude bad bins (where occupancy is not > 0 for at least minL)
        self.rprint("Shuffling data to find Place Cells:")
        self.print_wFrm(f"shuffling position data {self.shuffN} times")
        self.PCSimple.update(
            self.TransVectorLapCircShuff_wEdges(excludeVec=badPosition)
        )
        PC_count = sum(self.PCSimple[self.PCLAkey["SHUFF"]][self.PCLAkey["ISPC"]])
        self.print_wFrm(
            f"Post shuffle: Found {PC_count} potential place cells", frame_num=1
        )

        # Calculate mean of 'isPC' and sum of 'Occupancy'
        # self.mean_isPC = np.mean(self.PCSimple[self.PCLAkey["SHUFF"]][self.PCLAkey["ISPC"]])
        # self.sumOccupancy = np.nansum(self.PCSimple[self.PCLAkey["OCCU"]])

        # Info calculations
        self.print_wFrm("Calculating PC info after shuffle")
        self.nCells = self.PCSimple[self.PCLAkey["POSRATE"]].shape[0]
        self.shuffN_postShuffle = self.PCSimple[self.PCLAkey["SHUFF"]][
            self.PCLAkey["RATEALLSHUFF"]
        ].shape[2]

        self._postShuffle_PCSimple_init()

        # Calc/find/process information per spike and per second
        self._proc_InfoPerSpk_N_Sec()

        # Find Place Cells & store in self variable
        self.isPC = np.where(
            self.PCSimple[self.PCLAkey["SHUFF"]][self.PCLAkey["ISPC"]]
        )[0]

        self.print_wFrm("Calculating Place Field Strength")
        # Calculate place field strength
        self._calc_pfs_by_lap()

        self.print_wFrm("Refining shuffle results wrt place cell ID")
        # create & fill in PCLapSess for LappedWEdges output
        self._proc_PCLapSess()

        PC_count_postproc = sum(
            self.PCLapSess[self.PCLAkey["SHUFF"]][self.PCLAkey["ISPC"]]
        )
        self.print_wFrm(
            f"Post refining results: Found {PC_count_postproc} place cells", frame_num=1
        )

        return self.PCLapSess

    def TransVectorLapCircShuff_wEdges(
        self,
        shuff_override: int | None = None,
        excludeVec: np.ndarray | None = None,
        RatePerc: int = 99,
        edgeRateMultiple: int = 2,
        trimRunStarts: float = 0.25,
        trimRunEnds: float = 0.25,
        minRunTime: int = 2,
        minPFBins: int = 5,
        minVel: float = 0,
    ) -> dict:
        """
        Calculates the TransVectorLapOutput for circularly shuffled spike data with edges.

        Parameters:
            shuff_override (int): Overrides the default number of shuffles.
            excludeVec (numpy.ndarray): Array of indices to exclude from the analysis.
            RatePerc (int): Percentage of maximum firing rate to consider as a place cell.
            edgeRateMultiple (int): Multiple of the maximum firing rate to consider as an edge cell.
            trimRunStarts (float): Percentage of run time to trim from the start.
            trimRunEnds (float): Percentage of run time to trim from the end.
            minRunTime (int): Minimum duration of a valid run.
            minPFBins (int): Minimum number of bins with firing rate above RatePerc to consider as a place field.
            minVel (int): Minimum velocity threshold for valid runs.

        Returns:
            dict: TransVectorLapOutput containing the calculated parameters and results.
        """
        shuffN = self.shuffN if shuff_override is None else shuff_override
        # populate self.gauss_kernel & self.LSV
        self.utils.fill_gkernel_N_LSV(
            kernel_length=10,
            sigma=5,
            LSV_start=0,
            LSV_stop=1,
            LSV_num=101,
            LSV_end=1.0001,
        )

        # initalize activity & copy to avoid changes to self.spikes
        activity = []
        activity = self.spikes.copy()

        self.TransVectorLapOutput = {}
        self.TransVectorLapOutput["PARAMS"] = {
            self.PCLAkey["RATEPERC"]: RatePerc,
            self.PCLAkey["EDGE_R_MULT"]: edgeRateMultiple,
            self.PCLAkey["TRIMSTART"]: trimRunStarts,
            self.PCLAkey["TRIMEND"]: trimRunEnds,
            self.PCLAkey["MINRUN"]: minRunTime,
            self.PCLAkey["MINPFBINS"]: minPFBins,
            self.PCLAkey["MINVEL"]: minVel,
            self.PCLAkey["SHUFFN"]: shuffN,
        }

        # add TVLP to utils class funcs
        self.utils.add_TVLO_to_utils(self.TransVectorLapOutput)

        # find runTimes w/minVel & minRunTime
        self.utils.find_runTimes(
            minVel=self.TransVectorLapOutput["PARAMS"][self.PCLAkey["MINVEL"]],
            minRunTime=self.TransVectorLapOutput["PARAMS"][self.PCLAkey["MINRUN"]],
            trimRunStarts=self.TransVectorLapOutput["PARAMS"][
                self.PCLAkey["TRIMSTART"]
            ],
            trimRunEnds=self.TransVectorLapOutput["PARAMS"][self.PCLAkey["TRIMEND"]],
        )

        # procure kRun value
        self.utils.find_kRun(excludeVec)

        # fills in respective self values for utils class funcs
        # activity here will only be reflecting where pos is ~NaN
        activity = self.utils.find_nanedNvalidPos(activity)

        # fills rawOccupancy values within utils func
        self.utils.find_rawOccupancy()

        # fills rawSums & rawPosRates
        self.utils.find_rawSums_N_PosRates(activity)

        # smooths Occupancy and PosRates within TVLO
        self.utils.smooth_Occupancy_N_posRates()

        if shuffN > 1:
            self.utils.shuffle2findPlaceCells(spikes=activity, shuffN=shuffN)

        self.TransVectorLapOutput["treadPos"] = self.treadPos
        self.TransVectorLapOutput["timeSeg"] = self.timeSeg

        self.TransVectorLapOutput = self.utils.return_filled_TransVectorLapOutput()

        return self.TransVectorLapOutput

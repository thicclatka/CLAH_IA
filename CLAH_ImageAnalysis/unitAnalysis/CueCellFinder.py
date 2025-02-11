"""
CueCellFinder Script

This script is designed for identifying, analyzing, and visualizing cue cells within a given dataset. The main class `CueCellFinder` leverages several utility and plotting classes to process neural data, perform statistical analyses, and generate visual representations of the results.

Usage:
1. Initialize the CueCellFinder class with the required data structures.
2. Use the provided methods to perform analysis and generate plots for cue cells.
3. Export the results for further analysis or visualization.

Example:
    from CLAH_ImageAnalysis.unitAnalysis import CueCellFinder

    cue_finder = CueCellFinder(cueShiftStruc, treadBehDict, C_Temporal, A_Spatial, PKSUtils)
    cue_finder.plot_startNmidCueCellTuning(refLapType, lapTypeNameArr)
    cue_finder.plot_midCueCells_via_2xMethod()
    cue_finder.plot_midCueCells_viaStatTest()
    cue_finder.createNexport_CCFdict()

Note: Ensure that the necessary input data structures (`cueShiftStruc`, `treadBehDict`, etc.) are correctly formatted and available before initializing the `CueCellFinder` class.

"""

import numpy as np
from CLAH_ImageAnalysis.unitAnalysis import CCF_Dep
from CLAH_ImageAnalysis.unitAnalysis import CCF_Plotter
from CLAH_ImageAnalysis.unitAnalysis import CCF_StatTesting
from CLAH_ImageAnalysis.unitAnalysis import CCF_Utils

######################################################
#  main class func
######################################################


class CueCellFinder(CCF_Utils):
    def __init__(
        self,
        cueShiftStruc: dict,
        treadBehDict: dict,
        C_Temporal: np.ndarray,
        A_Spatial: np.ndarray,
        PKSUtils: object,
        pksDict: dict,
        forPres: bool,
    ) -> None:
        """
        Initializes the CueCellFinder class.

        Parameters:
            cueShiftStruc (dict): The cue shift structure.
            treadBehDict (dict): The tread behavior dictionary.
            C_Temporal (np.ndarray): The C_Temporal value.
            A_Spatial (np.ndarray): The A_Spatial value.
            PKSUtils (object): The PKSUtils class.
            pksDict (dict): The pks dictionary.
            forPres (bool): Whether to plot for presentation.
        """
        # initiate CueCellInd dict
        self.CueCellInd = {}

        # store C_Temporal & A_Spatial
        self.C_Temporal = C_Temporal
        self.A_Spatial = A_Spatial

        # store initiated PKSUtils class into self
        self.PKSUtils = PKSUtils

        # store pksDict
        self.pksDict = pksDict

        # for presentation purposes
        self.forPres = forPres

        # this creates binnCuePos, cueLapBin, omitLaps, switchLaps, omitCueTimes, ind
        CCF_Utils.__init__(
            self,
            cueShiftStruc=cueShiftStruc,
            treadBehDict=treadBehDict,
        )

        # DS image name
        self.DS_image = self.image_utils.get_DSImage_filename()
        # self.DS_image = (
        #     self.file_tag["AVGCA"]
        #     + self.file_tag["TEMPFILT"]
        #     + self.file_tag["DOWNSAMPLE"]
        #     + self.file_tag["IMG"]
        # )

    def plot_startNmidCueCellTuning(
        self, refLapType: int, lapTypeNameArr: list
    ) -> None:
        """
        Plots the start and mid cue cell tuning curves.

        Args:
            refLapType (int): The reference lap type.
            lapTypeNameArr (list): The array of lap type names.
        """

        # need to Initialize CCF_plotter class
        # - CCF init to produce omitLaps for QT_utils
        # - QT_utils finds lapTypeNameArr
        # - need to introduce lapTypeNameArr to init CCF_plotter
        self.Plotter = CCF_Plotter(refLapType, lapTypeNameArr, forPres=self.forPres)

        # stores lapTypeNameARr & various refLapType vars into self
        self._setup_refLapType_vars(refLapType, lapTypeNameArr)

        # sets up twoOdor_oneLoc, oneOdor_twoLoc vars needed for various conditions in later processed
        self._setup_numOdor_numLoc_vars()

        # initiate posRates for reference & nonreference laptypes
        self._init_posRates_forRef_n_nonRef()

        # create CueCellInd
        self._init_CueCellInd_for_Start_n_Mid()

        for cc_types in self.CueCellInd:
            if self.CueCellInd[cc_types].size > 0:
                self.Plotter.plotUnitByTuning(
                    posRatesRef=self.posRatesRef,
                    posRatesNonRef=self.posRatesNonRef,
                    IndToPlot=self.CueCellInd[cc_types],
                    CC_Type=cc_types,
                    fig_name=f"{cc_types}CueCells",
                )

    def _find_midFR_2xMethod(self, edge: int = 5, isInfRatio: int = 25) -> None:
        """
        Find the mid field firing rate using the 2x method.

        Args:
            edge (int): The number of indices to include on each side of the peak.
            isInfRatio (int): The value to replace infinite ratios with.
        """

        self.midFieldRate = {self.refLapType_name: []}
        self.midFieldRate.update({key: [] for key in self.posRatesNonRef})

        for idx, mc in enumerate(self.CueCellInd["MID"]):
            # find mean firing rate for area around peak (+/- 5)
            slice_to_use = slice(
                self.maxpRR_ind[mc] - edge, self.maxpRR_ind[mc] + edge + 1
            )
            self.midFieldRate[self.refLapType_name].append(
                np.mean(self.posRatesRef[mc, slice_to_use])
            )
            for key in self.posRatesNonRef:
                self.midFieldRate[key].append(
                    np.mean(self.posRatesNonRef[key][mc, slice_to_use])
                )

        for key in self.midFieldRate:
            self.midFieldRate[key] = np.array(self.midFieldRate[key])

        # TODO: CHECK THIS TO SEE IF GETTING APPROPRIATE KEY OMIT FOR OPTO
        if self.refLapType_name == "BOTHCUES":
            self.key_omit = "OMITBOTH"
        elif self.refLapType_name == "CUE/LED/TONE":
            self.key_omit = "OMITALL"
        elif self.refLapType_name == "CUE1" and "OMITCUE1" in self.lapTypeNameArr:
            self.key_omit = "OMITCUE1"
        elif self.refLapType_name == "CUE1" and "OMITCUE1" not in self.lapTypeArr:
            self.key_omit = "OMITBOTH"
        elif self.refLapType_name == "CUE":
            self.key_omit = "OMITBOTH"

        self.omitRatio = (
            self.midFieldRate[self.refLapType_name] / self.midFieldRate[self.key_omit]
        )
        self.CueCellInd["MID_2X"] = self.CueCellInd["MID"][
            np.argwhere(self.omitRatio > 2)
        ]
        self.CueCellInd["MID_2X"] = self.CueCellInd["MID_2X"].flatten()

        self.omitRatio = np.where(np.isinf(self.omitRatio), isInfRatio, self.omitRatio)

        self.CueCellInd["NOT_CC"] = np.setdiff1d(
            self.CueCellInd["MID"], self.CueCellInd["MID_2X"]
        )

    def plot_midCueCells_via_2xMethod(self) -> None:
        """
        Plots the mid cue cells using the 2x method.

        This method finds the mid field rate using the 2x method and plots the mid cue cells
        based on the calculated field rates. It also plots the non-cue cells if there are any.
        """

        self._find_midFR_2xMethod()
        if self.CueCellInd["MID"].size == 0:
            self.Plotter.plot_MFR_SP(
                maxInd=self.maxpRR_ind,
                CellCueInd=self.CueCellInd["MID"],
                omitRatio=self.omitRatio,
                midFieldRateRef=self.midFieldRate[self.refLapType_name],
                midFieldRateOmit=self.midFieldRate[self.key_omit],
            )
        if self.CueCellInd["MID_2X"].size > 0:
            self.Plotter.plotUnitByTuning(
                posRatesRef=self.posRatesRef,
                posRatesNonRef=self.posRatesNonRef,
                IndToPlot=self.CueCellInd["MID_2X"],
                CC_Type="MID_2X",
                fig_name="MidCueCellS_2x_PFR",
            )
        if len(self.CueCellInd["NOT_CC"]) > 0:
            self.Plotter.plotUnitByTuning(
                posRatesRef=self.posRatesRef,
                posRatesNonRef=self.posRatesNonRef,
                IndToPlot=self.CueCellInd["NOT_CC"],
                CC_Type="NON",
                fig_name="NonCueCells",
            )

    def plot_midCueCells_viaStatTest(self) -> None:
        """
        Plots the mid-cue cells using statistical tests.
        """

        # func from CCF_Utils
        self._find_evTimeByCue()
        self._find_cueTrigSig()

        # use statistical testing on cueAmp
        # which is based on cueTrigSig
        self.cueTrigSig_StatTest()

        # plot cueTrigSig accordingly based on statistical testing to
        # determine cue cells
        self.plot_cTS()

        # plot cueAmp if cueAmp data is not empty
        self._plot_cueAmp()
        self._MapStatTestResults()

        # plot related Opto results if optoCheck is True
        if self.optoCheck:
            self._findNplot_OptoDiff()
            self._findNplot_OptoTrigSig()

    def cueTrigSig_StatTest(self) -> None:
        """
        Performs statistical testing on the cue-triggered signal.

        This method calculates the statistical tests (RankSum and T-Test) on the cue-triggered signal
        using the provided event times, cues, and indices.
        """

        self.CCF_Stats = CCF_StatTesting(
            self.cueTrigSig, self.evTimes, self.cues, self.ind
        )
        self.CCF_Stats._apply_StatTest_on_cueTrigSig()

        # extract the RankSumDict, TTestDict, and cueAmp from CCF_Stats
        self.RankSumDict, self.TTestDict, self.cueAmp = (
            self.CCF_Stats._return_statsNcueAmp()
        )

    def _plot_cueAmp(self) -> None:
        """
        Plot cue amplitude boxplots and export stats results to csv.

        If `cueAmp` data is available, this method plots cue amplitude boxplots using the `Plotter` class and exports the statistical testing results to a CSV file. If no `cueAmp` data is available, it prints a message indicating that there is no data.
        """

        if self.cueAmp:
            # plot cueAmp boxplots
            self.print_wFrm(
                "cueAmp (1 Fig; suplots by cell with cueAmp Violin Plots by cueType)"
            )
            self.Plotter.plot_cueTrigSig_OR_cueAmp(
                self.cueAmp, "cueAmp", ind=self.ind, VIO=True
            )
            # export stats results to csv
            self.print_wFrm(
                "Exporting Stats testing results using cueAmp comparisons to csv",
                frame_num=1,
            )
        else:
            self.print_wFrm("No cueAmp data... skipping")

    def _MapStatTestResults(self) -> None:
        """
        Maps the statistical test results to create the `sigASpat` dictionary.

        This method initializes a dictionary `sigASpat` that holds the A Spatial values.
        It iterates over the `RankSumDict` dictionary to populate the `sigASpat` dictionary.
        For each main cue and comparison cue, it converts the `sig` column of the `RankSumDict` entry into an array.
        It creates an index for significant cells and initializes the `sigASpat` entry for the comparison cue.
        Then, it fills the `sigASpat` entry by iterating over the significant cell indices.
        Finally, it prints a message and calls the `plot_sigASpat_overDSImage` method to plot the A_Spatial Cue Comparison Profile.
        """

        # need to init a dict that holds the A Spatial values
        self.sigASpat = {
            key: {key2: [] for key2 in self.RankSumDict[key]}
            for key in self.RankSumDict
        }
        for main_cue in self.RankSumDict:
            for comp_cue in self.RankSumDict[main_cue]:
                if self.RankSumDict[main_cue][comp_cue].size > 0:
                    plottable = True
                    # turn sig column of RankSumDict entry into array
                    sigArr = np.array(
                        self.RankSumDict[main_cue][comp_cue][:, self.CCF_Stats.sig_idx]
                    )
                    # create index for sig cells
                    # note this will always be bounded by CueCellInd["MID"]
                    SigCellIdxArr = np.where(sigArr == 1)[0]
                    # init sigASpat entry for comp_cue to enable filling via for loop
                    self.sigASpat[main_cue][comp_cue] = np.empty_like(
                        self.A_Spatial[:, range(len(SigCellIdxArr))]
                    )
                    for a_idx, c_idx in enumerate(SigCellIdxArr):
                        # find cell # from cell list by splitting from Cell_ struct
                        cell_idx = int(self.cell_list[c_idx].split("_")[1])
                        self.sigASpat[main_cue][comp_cue][:, a_idx] = self.A_Spatial[
                            :, cell_idx
                        ]
                else:
                    plottable = False
        if plottable:
            self.print_wFrm("A_Spatial Cue Comparison Profile (1 Fig per comparison)")
            self.Plotter.plot_sigASpat_overDSImage(
                self.sigASpat,
                self.RankSumDict,
                self.DS_image,
                self.CCF_Stats.sig_idx,
                self.CCF_Stats.dir_idx,
            )
        else:
            self.print_wFrm("No cells are present to test significant... skipping")

    def _find_cueTrigSig(self) -> None:
        """
        Finds the cue-triggered signals for each cell.
        """

        num_cells = self.C_Temporal.shape[0]

        # init empty cueTrigSig based on keys for evTimes
        self.cueTrigSig = {
            f"Cell_{cell}": {key: [] for key in self.evTimes}
            for cell in self.CueCellInd["MID"]
        }

        # create cell_list for latex indexing
        self.cell_list = list(self.cueTrigSig.keys())

        # init empty allTrigSig based on keys for evTimes
        self.allTrigSig = {
            f"Cell_{cell}": {key: [] for key in self.evTimes}
            for cell in range(num_cells)
        }

        # iterate over cells to find allTrigSig
        # filter for MID cells to find cueTrigSig
        for cell_num in range(num_cells):
            cell = f"Cell_{cell_num}"
            Ca_arr = self.C_Temporal[cell_num, :].copy()
            Ca_arr = self.PKSUtils.zScoreCa(Ca_arr)

            curr_pks = self.pksDict[f"seg{cell_num}"]["pks"]

            self.Plotter.plot_Ca_Trace_wLaps(
                Ca_arr, self.resampY.copy(), self.lapFrInds, curr_pks, cell_num
            )
            for key in self.evTimes:
                self.allTrigSig[cell][key] = self._find_evTrigSig(Ca_arr, key)
                if cell_num in self.CueCellInd["MID"]:
                    self.cueTrigSig[cell][key] = self.allTrigSig[cell][key].copy()

    def plot_cTS(self) -> None:
        """
        Plots the cue-triggered signals.
        """

        cue2use = self.cues[0]

        if len(self.cues) == 1:
            omit2use = "OMITCUE1"
        elif len(self.cues) == 2:
            omit2use = "OMITBOTH"
        elif len(self.cues) == 3:
            omit2use = "OMITALL"
        # omit2use = "OMITCUE1"

        if omit2use not in self.RankSumDict[cue2use].keys():
            omit2use = "OMITCUE1"

        sigArr = np.array(
            self.RankSumDict[cue2use][omit2use][:, self.CCF_Stats.sig_idx]
        )
        dirArr = np.array(
            self.RankSumDict[cue2use][omit2use][:, self.CCF_Stats.dir_idx]
        )
        MidSigInd = self.CueCellInd["MID"][np.where((sigArr == 1) & (dirArr == 1))[0]]

        # find index for Mossy Cells if txt file exists
        MossyCellList = []
        MossyFile = self.utils.findLatest(
            [self.file_tag["MOSSY"], self.file_tag["TXT"]]
        )
        if MossyFile:
            with open(MossyFile, "r") as f:
                MossyCellList = [int(line.strip()) for line in f]

        # Initialize counts and index lists
        cellCategories = ["CUE", "PLACE", "START", "NON"]
        if MossyFile:
            # Add MOSSY category if MossyCellList is not empty
            cellCategories.append("MOSSY")

        self.CueCellTable = {cat: 0 for cat in cellCategories}
        self.CueCellTable.update({f"{cat}_IDX": [] for cat in cellCategories})
        self.CueCellTable["TOTAL"] = len(self.allTrigSig.keys())

        # Define conditions and corresponding categories
        cond4cellClassification = [
            (
                lambda cell_num: cell_num in self.CueCellInd["PC"]
                and cell_num not in MidSigInd
                and cell_num not in MossyCellList,
                "PLACE",
            ),
            (
                lambda cell_num: cell_num in MidSigInd
                and cell_num not in MossyCellList,
                "CUE",
            ),
            (
                lambda cell_num: cell_num in self.CueCellInd["START"]
                and cell_num not in MossyCellList,
                "START",
            ),
            (lambda cell_num: cell_num in MossyCellList, "MOSSY"),
        ]

        for cell in self.allTrigSig.keys():
            cell_num = int(cell.split("_")[-1])
            category = "NON"  # Default category
            for condition, cat in cond4cellClassification:
                if condition(cell_num):
                    category = cat
                    break
            self.CueCellTable[category] += 1
            self.CueCellTable[f"{category}_IDX"].append(cell_num)

        for cellType in cellCategories:
            self.CueCellTable[f"{cellType}_prop"] = (
                self.CueCellTable[cellType] / self.CueCellTable["TOTAL"]
            )

        print()
        print("Cue Cell Classification Results:")
        self.print_wFrm("Total Cells: " + str(self.CueCellTable["TOTAL"]))
        for cellType in cellCategories:
            self.print_wFrm(
                f"{cellType} Cells: {self.CueCellTable[cellType]} ({self.CueCellTable[f'{cellType}_prop']:.2%})"
            )
        print()

        #! SUPER IMPORTANT
        # add CueCellTable to Plotter
        # allows for better labeling in plots
        self.Plotter.plot_CueCellTable(self.CueCellTable)

        self.rprint("Plotting:")
        if self.cueTrigSig:
            self.print_wFrm("cueTrigSig (Figs by CueType; subplots by cell)")
            self.Plotter.plot_cueTrigSig_OR_cueAmp(
                self.cueTrigSig.copy(), "cueTrigSig", ind=self.ind
            )
            self.print_wFrm(
                "Mean cueTrigSig (1 Fig; subplots by cell with avgCTS by cueType)"
            )
            self.Plotter.plot_cueTrigSig_OR_cueAmp(
                self.cueTrigSig.copy(), "cueTrigSig", ind=self.ind, SEM=True
            )
        else:
            self.print_wFrm("No cueTrigSig to plot...skipping")

    def _findNplot_OptoDiff(self) -> None:
        """
        Finds and plots the opto difference.
        """

        def nansem(data: np.ndarray) -> np.ndarray:
            """
            Calculates the standard error of the mean (SEM) for non-nan values in the data.

            Parameters:
                data (np.ndarray): The data to calculate the SEM for.

            Returns:
                np.ndarray: The SEM of the data.
            """
            n = np.sum(~np.isnan(data))
            return np.nanstd(data, axis=0) / np.sqrt(n)

        self.cues4opto = [self.cues[0], "CUEwOPTO"]
        self.optoType = ["OPTOplus", "OPTOminus"]

        self.OptoDiff = {key: [] for key in self.allTrigSig.keys()}
        self.OptoAmp = {ot: {ct: [] for ct in self.cues4opto} for ot in self.optoType}
        self.OptoStd = {ot: {ct: [] for ct in self.cues4opto} for ot in self.optoType}
        self.OptoIdx = {ot: [] for ot in self.optoType}
        self.OptoPosVal = []
        self.OptoNegVal = []
        # amplitudes = {key: [] for key in cues4opto}
        for cell in self.allTrigSig.keys():
            amplitudes = {key: [] for key in self.cues4opto}
            for c in self.cues4opto:
                for cue_idx in range(self.allTrigSig[cell][c].shape[-1]):
                    amplitudes[c].append(
                        CCF_Dep.findMax_fromTrigSig(
                            TrigSig=self.allTrigSig[cell][c][:, cue_idx],
                            ind=self.ind,
                            baseline=False,
                        )
                    )

            cue = np.array(amplitudes[self.cues[0]])
            cuewOpto = np.array(amplitudes["CUEwOPTO"])

            meanCue = np.nanmean(cue)
            meanCwO = np.nanmean(cuewOpto)
            meanDiff = meanCwO - meanCue
            # min_val = min(meanCue, meanCwO)
            # max_val = max(meanCue, meanCwO)
            # normalizedDiff = (
            #     (meanDiff) / (max_val - min_val) if max_val != min_val else 0
            # )

            stdDiff = CCF_Dep.bootstrap2findSTD(
                cue, cuewOpto, n_samples=max(len(cue), len(cuewOpto))
            )

            self.OptoDiff[cell] = (meanDiff, stdDiff)

            # Determine the appropriate key based on the value of meanDiff
            key = "OPTOplus" if meanDiff > 0 else "OPTOminus"

            # Append the mean values to the appropriate lists
            self.OptoAmp[key][self.cues4opto[0]].append((cell, meanCue))
            self.OptoAmp[key][self.cues4opto[1]].append((cell, meanCwO))

            # std of the amps
            self.OptoStd[key][self.cues4opto[0]].append((cell, nansem(cue)))
            self.OptoStd[key][self.cues4opto[1]].append((cell, nansem(cuewOpto)))

            self.OptoIdx[key].append(
                (cell, CCF_Dep.classify_cellType(cell, self.CueCellTable))
            )

            # Append meanDiff to the appropriate list
            if meanDiff > 0:
                self.OptoPosVal.append(meanDiff)
            else:
                self.OptoNegVal.append(meanDiff)

        # plot results
        self.Plotter.plot_OptoDiff(
            self.OptoDiff,
            diff_val_tuple=(self.OptoPosVal, self.OptoNegVal),
        )
        # plot OptoDiff over ASpat iamge
        self.Plotter.plot_OptoDiff_overDSImage(
            DS_image=self.DS_image,
            OptoDiff=self.OptoDiff,
            ASpat=self.A_Spatial,
        )

    def _findNplot_OptoTrigSig(self, threshold: int = 10**2) -> None:
        """
        Finds and plots the opto-triggered signals.

        Parameters:
            threshold (int, optional): The threshold for outliers. Defaults to 10**2.
        """

        def _find_meanWfilter(val: np.ndarray) -> np.ndarray:
            """
            Finds the mean of the values in the array that are less than the threshold.

            Parameters:
                val (np.ndarray): The array to find the mean of.

            Returns:
                np.ndarray: The mean of the values in the array that are less than the threshold.
            """
            return np.nanmean(val[np.abs(val) < threshold])

        cueTypes = self.allTrigSig[next(iter(self.allTrigSig))].keys()

        self.OptoTrigSig = {ot: {} for ot in self.optoType}
        self.meanOTS = {ot: {ct: [] for ct in cueTypes} for ot in self.optoType}
        self.selectOptoAmp = {
            ot: {ct: [] for ct in self.cues4opto} for ot in self.optoType
        }
        self.selectOptoStd = {
            ot: {ct: [] for ct in self.cues4opto} for ot in self.optoType
        }

        self.selectOptoIdx = {ot: [] for ot in self.optoType}

        mean_plus = _find_meanWfilter(np.array(self.OptoPosVal))
        mean_minus = _find_meanWfilter(np.array(self.OptoNegVal))

        for cell, opto_vals in self.OptoDiff.items():
            diff = opto_vals[0]
            if diff > mean_plus:
                key = "OPTOplus"
            elif diff < mean_minus:
                key = "OPTOminus"
            else:
                continue

            # need to get correct idx for cell based on whether opto is plus or minus
            for idx, (cell_idx, _) in enumerate(self.OptoAmp[key][self.cues4opto[0]]):
                if cell == cell_idx:
                    idx2use = idx
                    break

            self.OptoTrigSig[key][cell] = self.allTrigSig[cell].copy()

            self.selectOptoIdx[key].append(
                (cell, CCF_Dep.classify_cellType(cell, self.CueCellTable))
            )
            for oc in self.cues4opto:
                self.selectOptoAmp[key][oc].append(self.OptoAmp[key][oc][idx2use][-1])
                self.selectOptoStd[key][oc].append(self.OptoStd[key][oc][idx2use][-1])

        for ot, TrigSigsByCell in self.OptoTrigSig.items():
            for cell, TrigSig in TrigSigsByCell.items():
                for cueType, ts in TrigSig.items():
                    mean = np.nanmean(ts, axis=1)
                    # if mean.shape[0] == sum(np.abs(self.ind)):
                    self.meanOTS[ot][cueType].append(mean)
            for cueType, meanTS in self.meanOTS[ot].items():
                self.meanOTS[ot][cueType] = np.array(meanTS).T

        # plot by cell
        for optokey, TrigSigsByCell in self.OptoTrigSig.items():
            self.Plotter.plot_cueTrigSig_OR_cueAmp(
                dict_to_plot=TrigSigsByCell,
                fname="Opto",
                ind=self.ind,
                SEM=True,
                OPTO=optokey,
            )

        # plot mean by opto Type
        self.Plotter.plot_meanOptoTrigSig(meanOTS=self.meanOTS, ind=self.ind)
        # plot cue vs cue + opto
        self.Plotter.plot_cueVScueOpto(
            optoAmps=self.selectOptoAmp,
            OptoStd=self.selectOptoStd,
            OptoIdx=self.selectOptoIdx,
            cues4opto=self.cues4opto,
        )

    def createNexport_CCFdict(self):
        """
        Creates a dictionary containing various data related to cue cell finding.

        Returns:
            dict: A dictionary containing the following keys:
                - "ATRIGSIG": All trigger signals
                - "CTRIGSIG": Cue trigger signals
                - "RANKSUM": Rank sum dictionary
                - "TTEST": T-test dictionary
                - "AMP": Cue amplitude
                - "EVTIME": Event times
                - "CELLIND": Cue cell indices
                - "CELLTABLE": Cue cell table
                - "OPTODIFF": Opto difference
                - "OPTOTRIGSIG": Opto-triggered signals
                - "OPTOAMP": Opto amplitudes
                - "OPTOSTD": Opto standard deviations
                - "OPTOIDX": Opto indices
                - "S_OPTOAMP": Selected opto amplitudes
                - "S_OPTOSTD": Selected opto standard deviations
                - "S_OPTOIDX": Selected opto indices
        """
        CCFdict = {
            self.CCFkey["ATRIGSIG"]: self.allTrigSig,
            self.CCFkey["CTRIGSIG"]: self.cueTrigSig,
            self.CCFkey["RANKSUM"]: self.RankSumDict,
            self.CCFkey["TTEST"]: self.TTestDict,
            self.CCFkey["AMP"]: self.cueAmp,
            self.CCFkey["EVTIME"]: self.evTimes,
            self.CCFkey["CELLIND"]: self.CueCellInd,
            self.CCFkey["CELLTABLE"]: self.CueCellTable,
            self.CCFkey["OPTODIFF"]: self.OptoDiff if self.optoCheck else None,
            self.CCFkey["OPTOTRIGSIG"]: self.OptoTrigSig if self.optoCheck else None,
            self.CCFkey["OPTOAMP"]: self.OptoAmp if self.optoCheck else None,
            self.CCFkey["OPTOSTD"]: self.OptoStd if self.optoCheck else None,
            self.CCFkey["OPTOIDX"]: self.OptoIdx if self.optoCheck else None,
            self.CCFkey["S_OPTOAMP"]: self.selectOptoAmp if self.optoCheck else None,
            self.CCFkey["S_OPTOSTD"]: self.selectOptoStd if self.optoCheck else None,
            self.CCFkey["S_OPTOIDX"]: self.selectOptoIdx if self.optoCheck else None,
        }
        self.saveNloadUtils.savedict2file(
            dict_to_save=CCFdict,
            dict_name=self.dict_name["CCF"],
            filename=self.dict_name["CCF"],
            filetype_to_save=[self.file_tag["MAT"], self.file_tag["PKL"]],
        )

        self.saveNloadUtils.savedict2file(
            dict_to_save=self.CueCellTable,
            dict_name="CueCellTable",
            filename="Figures/CueCellTableDict",
            filetype_to_save=self.file_tag["JSON"],
        )

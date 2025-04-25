import json
import numpy as np
from typing import Literal
from matplotlib.colors import ListedColormap
from scipy.stats import mannwhitneyu
from CLAH_ImageAnalysis.core import BaseClass as BC
from CLAH_ImageAnalysis.core import run_CLAH_script
from CLAH_ImageAnalysis.decoder import decoder_enum
from CLAH_ImageAnalysis.decoder import GeneralDecoder
from CLAH_ImageAnalysis.unitAnalysis import pks_utils


class TwoOdorDecoder(BC):
    def __init__(
        self,
        path: str,
        sess2process: list,
        num_folds: int,
        null_repeats: int,
        cost_param: float,
        kernel_type: str,
        gamma: float | Literal["auto", "scale"],
        weight: str,
        n_estimators: int,
        max_depth: int,
        learning_rate: float,
    ) -> None:
        self.program_name = "TOD"
        self.class_type = "manager"
        BC.__init__(self, self.program_name, mode=self.class_type)

        self.static_class_var_init(
            path,
            sess2process,
            num_folds,
            null_repeats,
            cost_param,
            kernel_type,
            gamma,
            weight,
            n_estimators,
            max_depth,
            learning_rate,
        )

    def process_order(self) -> None:
        """
        Executes the order of operations for processing the data.

        This method performs the following steps:
        1. Imports CSS, TBD, and SD.
        2. Finds odor times.
        3. Finds trial epochs.
        4. Decodes trial epochs.
        5. Plots the results.

        Returns:
            None
        """
        self.importCSSnTBDnSD()
        self.findOdorTimes()
        self.findOdorDetails()
        self.findPosRateClustering()
        self.findOdorClustering()
        self.DecodeOdorEpochs()
        self.plotResults()
        self.analyze_tuning_similarity_by_cell_type()

    def static_class_var_init(
        self,
        path: str,
        sess2process: list,
        num_folds: int,
        null_repeats: int,
        cost_param: float,
        kernel_type: str,
        gamma: float | Literal["auto", "scale"],
        weight: str,
        n_estimators: int,
        max_depth: int,
        learning_rate: float,
    ) -> None:
        """
        Initializes the static class variables for the TwoOdorDecoder class.

        Args:
            path (str): The folder path.
            sess2process (str): The session to process.
            num_folds (int): The number of folds for cross-validation.
            null_repeats (int): The number of null repeats.
            cost_param (float): The cost parameter.

        Returns:
            None
        """
        BC.static_class_var_init(
            self,
            folder_path=path,
            file_of_interest=self.text_lib["selector"]["tags"]["CSS"],
            selection_made=sess2process,
        )

        # needed to use fig_tools
        BC.enable_fig_tools(self)

        self.PKS_UTILS = pks_utils()

        self.sampling_rate = 10  # 10 frames per second (10 Hz; 100 ms per frame)
        self.pre_cue_time = 1  # second before cue start
        self.post_cue_time = 2  # seconds after cue start
        self.trial_dur = (self.pre_cue_time + self.post_cue_time) * self.sampling_rate

        self.decoder_type = "SVC"  # hardcoded for now

        self.num_folds = num_folds
        self.num_folds4switch = None
        self.null_repeats = null_repeats
        self.params4decoderSVC = {
            "C": cost_param,
            "kernel_type": kernel_type,
            "gamma": gamma,
            "weight": weight,
        }
        # self.params4decoderLSTM = {
        #     "n_estimators": n_estimators,
        #     "max_depth": max_depth,
        #     "learning_rate": learning_rate,
        # }
        self.params4decoder = self.params4decoderSVC.copy()

        self.parse_cost_param = float(cost_param) if cost_param is not None else None
        self.cost_param = cost_param

        self.FigPath = "Figures/Decoder"

        self.random_state = 11

        self.Label_Names = {
            "ODORS": ["ODOR 1", "ODOR 2"],
            "ODORSwSWITCH": ["O1L1", "O2L2", "O1L2", "O2L1"],
        }

        self.LSTM_params = {
            "seq_len": 20,
            "epochs": 50,
            "batch_size": 32,
            "use_early_stopping": True,
            "patience": 10,
        }

        self.cellTypes = ["CUE1", "CUE2", "BOTHCUES", "PLACE"]
        self.cellTypes_wNon = ["CUE1", "CUE2", "BOTHCUES", "PLACE", "NON"]

        self.plot_params = {
            "bar": {
                "alpha": 0.7,
                "width": 0.2,
                "offset": 0.1,
                "offset2": 0.05,
                "width2": 0.1,
                "hatch": "//",
            },
            "scatter": {
                "alpha": 0.8,
                "s_small": 20,
                "s_large": 50,
            },
            "cat_names4plot": ["Odors", "Odors + Switch"],
            "fsize": {
                "axis": 14,
                "title": 18,
            },
            "cmap": {
                "imshow": "magma",
                "odors": self.fig_tools.create_cmap4categories(num_categories=4),
                "cellTypes": self.fig_tools.create_cmap4categories(
                    num_categories=len(self.cellTypes), cmap_name="tab20"
                ),
                "cellTypes_wNon": self.fig_tools.create_cmap4categories(
                    num_categories=len(self.cellTypes_wNon), cmap_name="tab20"
                ),
            },
            "locator": {
                "size": "3%",
                "pad": 0.05,
            },
            "jitter": 0.02,
            "markers": {
                "CUE1": "$1$",
                "CUE2": "$2$",
                "BOTHCUES": "$B$",
                "PLACE": "$P$",
            },
        }

    def forLoop_var_init(self, sess_idx: int, sess_num: int) -> None:
        """
        Initializes variables for the for loop.

        Args:
            sess_idx (int): The index of the current session.
            sess_num (int): The total number of sessions.

        Returns:
            None
        """
        super().forLoop_var_init(sess_idx, sess_num)
        self.CSS = None
        self.TBD = None
        self.SD = None
        self.OdorTimes = None
        self.OdorEpochs = None
        self.CueCellTableDict = None
        if self.parse_cost_param is not None:
            self.params4decoderSVC["C"] = self.parse_cost_param
        else:
            self.params4decoderSVC["C"] = None
        self.Labels = {
            "ODORS": [],
            "ODORSwSWITCH": [],
        }

    def importCSSnTBDnSD(self) -> None:
        """
        Imports CSS, TBD, and SD files for the given folder_name.

        This method loads CSS, TBD, and SD files for the specified folder_name. It iterates over
        the file_types list and calls the _import_file_type method for each file type. After
        importing the files, it extracts the lapCue from the CSS file and assigns it to the
        lapCue attribute. It also extracts the C_Temporal from the SD file and assigns it to the
        C_Temp attribute.

        Returns:
            None
        """
        self.rprint(f"Loading CSS, TBD, and SD for {self.folder_name}:")
        file_types = ["CSS", "TBD", "SD"]

        for file_type in file_types:
            self._import_file_type(file_type)

        # import cell type table as self.CueCellTableDict
        self._import_CellTypeTable()

        self.cellTypeInds = {
            ctype: self.CueCellTableDict[f"{ctype}_IDX"] for ctype in self.cellTypes
        }

        ## extract lapCue from CSS
        self.lapCue = self.CSS["lapCue"]

        # extract C
        self.C_Temp = self.SD["C"]
        self.zC_Temp = []
        for idx in range(self.C_Temp.shape[0]):
            Ca_arr = self.PKS_UTILS.zScoreCa(self.C_Temp[idx, :])
            self.zC_Temp.append(Ca_arr)

        self.zC_Temp = np.array(self.zC_Temp)

        # extract PosRates
        self.PCLappedSess = self.CSS["PCLappedSess"]

        self.print_done_small_proc()

    def _import_file_type(self, file_type: str) -> None:
        """
        Imports the latest file of the specified file type and assigns it to the corresponding attribute.

        Args:
            file_type (str): The type of file to import.

        Returns:
            None
        """
        latest_file = self.findLatest([self.file_tag[file_type], self.file_tag["PKL"]])
        setattr(self, file_type, self.load_file(latest_file))
        self.print_wFrm(f"Loaded {file_type:<3}: {latest_file}")

    def _import_CellTypeTable(self) -> None:
        """
        Imports the cell type table.
        """
        CCT = self.findLatest(["CueCellTableDict"], path2check="Figures")
        with open(CCT, "rb") as f:
            self.CueCellTableDict = json.load(f)

        both_cue = list(
            set(self.CueCellTableDict["CUE1_IDX"])
            & set(self.CueCellTableDict["CUE2_IDX"])
        )
        self.CueCellTableDict["CUE1_IDX"] = list(
            set(self.CueCellTableDict["CUE1_IDX"]) - set(both_cue)
        )
        self.CueCellTableDict["CUE2_IDX"] = list(
            set(self.CueCellTableDict["CUE2_IDX"]) - set(both_cue)
        )
        self.CueCellTableDict["BOTHCUES_IDX"] = both_cue
        for key in ["CUE1_IDX", "CUE2_IDX", "BOTHCUES_IDX"]:
            self.CueCellTableDict[key].sort()

    def findOdorTimes(self) -> None:
        """
        Finds the odor times and labels.

        This method calculates the odor times and labels based on the cue events and lap information.
        It creates an array where the 0th column represents the time, the 1st column represents the odor label,
        and the 2nd column represents the switch label. The switch label is determined based on the lap type.

        Returns:
            None
        """

        def _find_switch_lap(lapTypeArr, lapTypeName):
            lapTypes = np.unique(lapTypeArr)
            for idx, ltn in enumerate(lapTypeName):
                if "switch" == ltn.lower():
                    switch_lap = int(lapTypes[idx])
            return switch_lap

        self.rprint("Finding Odor Times and labels:")

        lapTypeArr = self.lapCue["lap"]["TypeArr"].astype(int)
        # unique_codes = np.unique(lapTypeArr)
        lapTypeName = self.lapCue["lap"]["lapTypeName"]
        lapTimes = self.TBD["lap"]["Time"]
        # code_to_index = {code: idx for idx, code in enumerate(unique_codes)}
        # laptype_indices = np.array([code_to_index[code] for code in lapTypeArr])
        switch_lap = _find_switch_lap(lapTypeArr, lapTypeName)

        all_odor_times = np.array([]).reshape(0, 5)

        self.plot_params["cmap"]["lapType"] = self.fig_tools.create_cmap4categories(
            num_categories=len(lapTypeName), cmap_name="tab20"
        )

        # create an array such that:
        # 0th column is cue time
        # 1st column is lap start time
        # 2nd column is lap number
        # 3rd column is odor label (0 for odor 1, 1 for odor 2)
        # 4th column is switch label (0 for odor 1, 1 for odor 2, 2 and 3 for switch laps for odor 1 and 2 respectively)
        for odor in range(0, 2):
            # get Times, Laps, and Types for each odor
            # types should represent odor & switch
            cueTimes = self.TBD["cueEvents"][f"cue{odor + 1}"]["start"]["Time"]
            cueLaps = self.lapCue[f"CUE{odor + 1}"]["Lap"].astype(int)
            cueTypes = lapTypeArr[cueLaps]

            lapStartIdx = np.searchsorted(lapTimes, cueTimes) - 1
            lapStart = lapTimes[lapStartIdx]

            # Times, laps, types all should have same shape
            num_trials = cueTimes.shape[0]

            # create odor labels
            odor_labels = odor * np.ones(num_trials)

            # create switch labels
            switch_labels = odor_labels.copy()
            switch_labels[cueTypes == switch_lap] = odor + 2

            odor_times = np.column_stack(
                (cueTimes, lapStart, cueLaps, odor_labels, switch_labels)
            )
            all_odor_times = np.vstack((all_odor_times, odor_times))

        # sort by time
        indices = np.argsort(all_odor_times[:, 0])
        self.OdorTimes = all_odor_times[indices]

        odor_labels, odor_counts = np.unique(self.OdorTimes[:, -2], return_counts=True)
        switch_labels, switch_counts = np.unique(
            self.OdorTimes[:, -1], return_counts=True
        )

        for lab, count in [(odor_labels, odor_counts), (switch_labels, switch_counts)]:
            label_count_pairs = [f"{int(lb)}: {c}" for lb, c in zip(lab, count)]
            formatted_str = " / ".join(label_count_pairs)
            if lab.size == 2:
                self.print_wFrm(f"Odor Labels & Counts: {formatted_str}")
            elif lab.size == 4:
                self.print_wFrm(f"Odor with Switch Labels & Counts: {formatted_str}")

        self.print_done_small_proc()
        self.lapTimes = lapTimes
        self.lapTypeArr = lapTypeArr
        self.lapTypeName = lapTypeName
        self.adjFrTimes = np.array(self.TBD["adjFrTimes"])

    def findOdorDetails(self) -> None:
        """
        Organizes times and labels by trial epochs and extracts peak values and times.
        """
        self.rprint(
            "Creating features array (OdorPeaksNTimes, (exposures x features)) and labels (Labels, (exposures,)):"
        )
        self.print_wFrm(
            f"Looking at {int(self.trial_dur / self.sampling_rate)} second windows: {self.pre_cue_time} sec pre-cue; {self.post_cue_time} sec post-cue"
        )
        self.print_wFrm(
            "Finding peak value and time, relative to relative to lap start time within post-cue epoch period"
        )

        # Initialize arrays
        self.OdorEpochs = np.full(
            (self.C_Temp.shape[0], self.trial_dur, self.OdorTimes.shape[0]),
            np.nan,
        )
        self.OdorPeaksNTimes = np.full(
            (
                self.OdorTimes.shape[0],
                self.C_Temp.shape[0] * 2,
            ),  # multiple of 2 for interleaved values
            np.nan,
        )
        self.Labels = {
            "ODORS": np.full(self.OdorTimes.shape[0], np.nan),
            "ODORSwSWITCH": np.full(self.OdorTimes.shape[0], np.nan),
        }

        # use z-scored C_Temp
        # CTEMP2USE = self.zC_Temp.copy()
        CTEMP2USE = self.C_Temp.copy()

        for idx, ot in enumerate(self.OdorTimes):
            cue_start_time = ot[0]
            lap_start_time = ot[1]
            odor_label = ot[-2]
            switch_label = ot[-1]

            trial_start = int(cue_start_time - self.pre_cue_time)
            trial_end = int(cue_start_time + self.post_cue_time)

            # find indices of frames that are within the trial epoch
            trial_indices = np.where(
                (self.adjFrTimes >= trial_start) & (self.adjFrTimes <= trial_end)
            )[0]
            post_cue_indices = np.where(
                (self.adjFrTimes > cue_start_time) & (self.adjFrTimes <= trial_end)
            )[0]

            if trial_indices.shape[0] == self.trial_dur:
                # Get trial data
                trial_data = CTEMP2USE[:, trial_indices]
                self.OdorEpochs[:, :, idx] = trial_data

                # Get post-cue data for peak detection
                post_cue_data = CTEMP2USE[:, post_cue_indices]
                post_time_data = self.adjFrTimes[post_cue_indices]

                # Find peaks and times
                max_peak_val = np.max(post_cue_data, axis=1)
                max_peak_idx = np.argmax(post_cue_data, axis=1)
                max_peak_time = (
                    post_time_data[max_peak_idx] - lap_start_time
                )  # Relative to lap start

                # Store interleaved RAW peaks and times
                self.OdorPeaksNTimes[idx] = np.column_stack(
                    (max_peak_val, max_peak_time)
                ).reshape(-1)

                # Store labels
                self.Labels["ODORS"][idx] = odor_label
                self.Labels["ODORSwSWITCH"][idx] = switch_label

        # check # of switchs labels to set num_folds4switch
        switch_labels, counts = np.unique(
            self.Labels["ODORSwSWITCH"], return_counts=True
        )
        if counts[3] < self.num_folds or counts[4] < self.num_folds:
            count_min = np.min(counts[3:])
            self.num_folds4switch = count_min
            output_str = f"Only {count_min} switch laps found; setting cross-fold validation to {count_min}"
        else:
            output_str = f"Enough switch laps found; setting cross-fold validation to {self.num_folds}"
            self.num_folds4switch = self.num_folds
        self.print_wFrm(output_str)

        self.print_wFrm(
            "Applying normalizing to features array (OdorPeaksNTimes -> normOdorPeaksNTimes)"
        )
        self.normOdorPeaksNTimes = self.dep.feature_normalization(self.OdorPeaksNTimes)

        self.print_done_small_proc()

    def findPosRateClustering(self) -> None:
        """
        Creates a feature vector for position decoder.
        """
        print("Creating similarity matrices from spatial firing rate data")
        self.simMat4PRClustering = {lName: None for lName in self.lapTypeName}
        self.simMat4PRClustering["concat"] = []
        TuningCurvesScaled = []
        for lidx in np.unique(self.lapTypeArr):
            curr_lName = self.lapTypeName[lidx - 1]
            # curr_posRateByLap = self.PCLappedSess[f"lapType{lidx}"]["ByLap"]["posRates"]
            TC = self.PCLappedSess[f"lapType{lidx}"]["posRates"]
            (N_cells, P_bins) = TC.shape
            # --- Feature Scaling (Min-Max per cell/feature) on Downsampled FV ---
            TC_scaled = np.zeros_like(TC)
            for cell_idx in range(N_cells):
                cell_data = TC[cell_idx, :]
                min_val = np.min(cell_data)
                max_val = np.max(cell_data)
                range_val = max_val - min_val

                if range_val > 0:
                    TC_scaled[cell_idx, :] = (cell_data - min_val) / range_val
                else:
                    TC_scaled[cell_idx, :] = 0
            simMat = self.dep.calc_simMatrix(TC_scaled)
            self.simMat4PRClustering[curr_lName] = simMat
            TuningCurvesScaled.append(TC_scaled)
        self.TuningCurvesScaled = TuningCurvesScaled
        TC2simMatConcat = np.concatenate(TuningCurvesScaled, axis=1)
        self.simMat4PRClustering["concat"] = self.dep.calc_simMatrix(TC2simMatConcat)

        N_cells = len(self.simMat4PRClustering["concat"])
        true_cellTypeLabel = np.full(N_cells, np.nan)
        for c in range(N_cells):
            if c in self.cellTypeInds["CUE1"]:
                true_cellTypeLabel[c] = 0
            elif c in self.cellTypeInds["CUE2"]:
                true_cellTypeLabel[c] = 1
            elif c in self.cellTypeInds["BOTHCUES"]:
                true_cellTypeLabel[c] = 2
            elif c in self.cellTypeInds["PLACE"]:
                true_cellTypeLabel[c] = 3
            else:
                true_cellTypeLabel[c] = 4
        self.true_cellTypeLabel = true_cellTypeLabel

        self.cluster_labels4PR, self.cluster_accuracy4PR = (
            self.dep.SpectralClustering_fit2simMatrix(
                similarity_matrix=self.simMat4PRClustering["concat"],
                n_clusters=len(np.unique(true_cellTypeLabel)),
                random_state=self.random_state,
                true_labels=true_cellTypeLabel,
            )
        )

        self.DistanceMapBTW_BCnSW = None
        BC_idx = np.where(self.lapTypeName == "BOTHCUES")[0][0]
        SW_idx = np.where(self.lapTypeName == "SWITCH")[0][0]

        rowwise_corr_coeffs = np.full(len(self.simMat4PRClustering["concat"]), np.nan)
        for c in range(N_cells):
            BC_row = TuningCurvesScaled[BC_idx][c]
            SW_row = TuningCurvesScaled[SW_idx][c]
            if np.std(BC_row) > 0 and np.std(SW_row) > 0:
                rowwise_corr_coeffs[c] = np.corrcoef(BC_row, SW_row)[0, 1]
            elif np.all(BC_row == SW_row):
                rowwise_corr_coeffs[c] = 1
            else:
                rowwise_corr_coeffs[c] = 0

        diff = TuningCurvesScaled[BC_idx] - TuningCurvesScaled[SW_idx]
        prod = TuningCurvesScaled[BC_idx] * TuningCurvesScaled[SW_idx]
        self.DistanceMapBTW_BCnSW = {
            "EUCLID": np.linalg.norm(diff, axis=1),
            "COSINE": np.sum(prod, axis=1)
            / (
                np.linalg.norm(TuningCurvesScaled[BC_idx])
                * np.linalg.norm(TuningCurvesScaled[SW_idx])
            ),
            "1-CORR": 1 - rowwise_corr_coeffs,
        }

        loc_keys = ["L1", "L2"]
        maxDiffs = {key: [] for key in loc_keys}
        minDiffs = {key: [] for key in loc_keys}
        minmaxDiffs = {key: [] for key in loc_keys}
        locations = {"L1": slice(10, 40), "L2": slice(60, 90)}
        for c in range(N_cells):
            for key, loc_slice in locations.items():
                maxD = np.max(diff[c, loc_slice])
                minD = np.min(diff[c, loc_slice])
                maxDiffs[key].append(maxD)
                minDiffs[key].append(minD)
                minmaxDiffs[key].append(
                    (maxD - (np.abs(minD))) / (maxD + (np.abs(minD)) + 1e-10)
                )
        maxDiffs = {key: np.array(maxDiffs[key]) for key in loc_keys}
        minDiffs = {key: np.array(minDiffs[key]) for key in loc_keys}
        minmaxDiffs = {key: np.array(minmaxDiffs[key]) for key in loc_keys}

        self.cluster_labels4BC_SW = {}
        self.cluster_accuracy4BC_SW = {}
        for metric in self.DistanceMapBTW_BCnSW.keys():
            if metric == "EUCLID":
                (
                    self.cluster_labels4BC_SW[metric],
                    self.cluster_accuracy4BC_SW[metric],
                ) = self.dep.KMeans_fit(
                    array=diff,
                    n_clusters=len(np.unique(true_cellTypeLabel)),
                    random_state=self.random_state,
                    true_labels=true_cellTypeLabel,
                )
            elif metric == "COSINE":
                (
                    self.cluster_labels4BC_SW[metric],
                    self.cluster_accuracy4BC_SW[metric],
                ) = self.dep.SpectralClustering_fit2simMatrix(
                    similarity_matrix=self.dep.calc_simMatrix(diff),
                    n_clusters=len(np.unique(true_cellTypeLabel)),
                    random_state=self.random_state,
                    true_labels=true_cellTypeLabel,
                )
            elif metric == "1-CORR":
                # Cluster based on the correlation similarity *between difference vectors*
                # Calculate pairwise Pearson correlations between the rows of 'diff'
                correlation_matrix = np.corrcoef(diff)
                # Handle potential NaNs if rows are constant or all-zero
                correlation_matrix = np.nan_to_num(correlation_matrix)
                # Shift correlation [-1, 1] to similarity [0, 1]
                similarity_matrix_corr = (correlation_matrix + 1) / 2

                (
                    self.cluster_labels4BC_SW[metric],
                    self.cluster_accuracy4BC_SW[metric],
                ) = self.dep.SpectralClustering_fit2simMatrix(
                    similarity_matrix=similarity_matrix_corr,  # Use correlation-based similarity
                    n_clusters=len(np.unique(true_cellTypeLabel)),
                    random_state=self.random_state,
                    true_labels=true_cellTypeLabel,
                    # No 'metric' argument needed here as we provide the similarity matrix
                )

            self.cluster_labels4BC_SW[metric] = (
                self.dep.map_cluster_labels_to_true_labels(
                    true_labels=true_cellTypeLabel,
                    cluster_labels=self.cluster_labels4BC_SW[metric],
                )
            )

        self.BCSW_comparison = {
            "DIFF": diff,
            "PROD": prod,
            "CORREL": rowwise_corr_coeffs,
            "MAXDIFF": maxDiffs,
            "MINDIFF": minDiffs,
            "MINMAXDIFF": minmaxDiffs,
        }

        self.print_done_small_proc()

    def findOdorClustering(self) -> None:
        """
        Performs clustering analysis following the paper's methodology:
        1. Computes cosine similarity between trials
        2. Performs spectral clustering
        3. Calculates projection weights
        """
        self.rprint("Finding odor clustering:")

        # Compute similarity matrix
        self.similarity_matrix = self.dep.calc_simMatrix(self.normOdorPeaksNTimes)

        self.cluster_labels = {}
        self.projection_weights = {}
        self.clustering_accuracy = {}
        self.corrected_cluster_labels = {}

        for label_cat, n_clusters in [
            ("ODORS", int(len(np.unique(self.Labels["ODORS"])))),
            ("ODORSwSWITCH", int(len(np.unique(self.Labels["ODORSwSWITCH"])))),
        ]:
            # Perform spectral clustering
            self.cluster_labels[label_cat], self.clustering_accuracy[label_cat] = (
                self.dep.SpectralClustering_fit2simMatrix(
                    similarity_matrix=self.similarity_matrix,
                    n_clusters=n_clusters,
                    random_state=self.random_state,
                    true_labels=self.Labels[label_cat],
                )
            )

            # Map cluster labels to true labels
            self.corrected_cluster_labels[label_cat] = (
                self.dep.map_cluster_labels_to_true_labels(
                    true_labels=self.Labels[label_cat],
                    cluster_labels=self.cluster_labels[label_cat],
                )
            )

            # Project each trial onto the average maps to get projection weights
            # use corrected cluster labels so colormaps line up later
            self.projection_weights[label_cat] = self.dep.determine_projection_weights(
                feature_array=self.normOdorPeaksNTimes,
                cluster_labels=self.corrected_cluster_labels[label_cat],
            )

        for label_cat, accu in self.clustering_accuracy.items():
            self.print_wFrm(f"Clustering accuracy for {label_cat}: {accu:.3f}")

        self.print_done_small_proc()

    def DecodeOdorEpochs(self) -> None:
        """
        Decodes the trial epochs using a specified cost parameter and labels.

        This method performs the decoding process by calculating the accuracy and confusion matrix for each label.
        It also randomizes the labels multiple times and calculates the accuracy for each randomization.

        Returns:
            None
        """

        def _decodeNfindAccu(label, folds):
            """
            Calculates the accuracy of the decoder based on the given label and number of folds.

            Parameters:
            - label: numpy array
                The label array used for decoding.
            - folds: int
                The number of folds used for cross-validationOdorPeaksNTimes.

            Returns:
            - accuracy: float
                The accuracy of the decoder.

            """
            # see params4decoder in self.static_class_var_init for SVC and GBM parameters
            accuracy, conf_matrices, _ = GeneralDecoder.run_Decoder(
                data_arr=self.normOdorPeaksNTimes,
                label_arr=np.array(label),
                num_folds=folds,
                decoder_type=self.decoder_type,
                random_state=self.random_state,
                **(self.params4decoder if self.params4decoder is not None else {}),
            )
            return accuracy, conf_matrices

        def _determine_fold_count(key: str) -> int:
            """
            Determines the fold count based on the given key.

            Parameters:
            - key (str): The key to determine the fold count for. Should be either "ODORS" or "fold4switch".

            Returns:
            - int: The fold count based on the given key. Returns `self.num_folds` if key is "ODORS", otherwise returns `self.num_folds4switch`.
            """
            return self.num_folds if key == "ODORS" else self.num_folds4switch

        def _print_acc_results(accu: np.ndarray, key: str, max_length: int) -> str:
            """
            Prints the accuracy results.

            Args:
                accu (numpy.ndarray): Array of accuracy values.

            Returns:
                str: A string representation of the accuracy results.

            """
            stats = ["Mean", "Std", "Max", "Min"]
            values = [
                np.mean(accu),
                np.std(accu),
                np.max(accu),
                np.min(accu),
            ]
            accu_str = "Accuracy: " + " / ".join(
                f"{stat}: {value:.4f}" for stat, value in zip(stats, values)
            )
            self.print_wFrm(
                f"{key:{max_length}} = {accu_str}",
                frame_num=1,
            )

        if self.cost_param is None and self.decoder_type == "SVC":
            self.rprint(
                "Finding best cost parameter (will use odor labels given it is more populated per group):"
            )

            self.cost_param = GeneralDecoder.calc_CostParam(
                data_arr=self.normOdorPeaksNTimes,
                label_arr=np.array(self.Labels["ODORS"]),
                num_folds=self.num_folds,
                kernel_type=self.params4decoder["kernel_type"],
                gamma=self.params4decoder["gamma"],
                weight=self.params4decoder["weight"],
            )
            self.params4decoder["C"] = self.cost_param
            self.print_wFrm(f"Best cost parameter: {self.cost_param}\n")
        elif self.cost_param is not None and self.decoder_type == "SVC":
            self.rprint(f"Using cost parameter set by parser: {self.cost_param}")

        self.rprint(f"Decoding via {self.decoder_type}:")
        if self.params4decoder is not None:
            self.print_wFrm("Parameters:")
            for key, value in self.params4decoder.items():
                self.print_wFrm(f"{key}: {value}")

        self.print_wFrm("Using labels as is:")

        self.accuracy = {key: None for key in self.Labels.keys()}
        self.confusion_matrix = {key: None for key in self.Labels.keys()}

        max_length = max(len(key) for key in self.Labels.keys())

        for key, label in self.Labels.items():
            self.accuracy[key], self.confusion_matrix[key] = _decodeNfindAccu(
                label, _determine_fold_count(key)
            )
            _print_acc_results(self.accuracy[key], key, max_length)

        self.print_wFrm(f"Randomizing labels {self.null_repeats} times:")
        # null_accuracy has shape of trial time x num repeats x num folds
        self.null_accuracy = {
            key: np.full(
                (
                    self.null_repeats,
                    _determine_fold_count(key),
                ),
                np.nan,
            )
            for key in self.Labels.keys()
        }

        self.print_wFrm("Decoding each randomization", frame_num=1, end="", flush=True)
        for rep in range(self.null_repeats):
            self.rprint(".", end="", flush=True)
            for key, label in self.Labels.items():
                null_label = np.random.permutation(np.array(label))
                null_result, _ = _decodeNfindAccu(
                    null_label, _determine_fold_count(key)
                )
                self.null_accuracy[key][rep, :] = null_result
        print("complete")

        for key, null_accu in self.null_accuracy.items():
            _print_acc_results(np.mean(null_accu, 1), key, max_length)
        self.print_done_small_proc()

    def plotResults(self) -> None:
        """
        Plots the results of the decoder analysis.

        This method plots the decoder accuracy over time within an epoch and also
        generates a confusion matrix.

        Returns:
            None
        """
        self.rprint("Plotting results:")

        self._plotBar_Accuracy()

        # confusion matrix
        self._plotConfusionMatrix()

        # similarity matrix with clustering
        self._plot_similarity_matrix4OdorDetails()

        # UMAP projection plot based on odor details
        self._plot_umap_projection4OdorDetails()

        # UMAP projection plot based on posRates
        self._plot_umap_projection4PosRates()

        # histogram of distance maps
        self._plot_histograms4DistanceMaps()

        # plot minmax diff
        self._plot_minmax_diff()

        # plot tuning similarity by cell type
        self.analyzing_BCSwitchTuning()

        self.print_done_small_proc()

    def _plotBar_Accuracy(self) -> None:
        fig, axis = self.fig_tools.create_plt_subplots()

        odor_color = self.color_dict["blue"]
        switch_color = self.color_dict["orange"]

        label_types = self.accuracy.keys()

        self.print_wFrm("Bar plots for accuracy")
        for idx, label_type in enumerate(label_types):
            unique_labels = np.unique(self.Labels[label_type])
            # Determine chance level based on number of categories
            y_chance = 1 / len(unique_labels)
            # Define start and end points for the chance line segment
            xmin = idx - (4 * self.plot_params["bar"]["offset"])
            xmax = idx + (4 * self.plot_params["bar"]["offset"])

            color2use = odor_color if label_type == "ODORS" else switch_color

            axis.axhline(
                y=y_chance,
                xmin=xmin,
                xmax=xmax,
                color="black",
                linestyle=":",
                linewidth=2,
            )

            accu2use = self.accuracy[label_type].T.squeeze()
            null_accu2use = np.mean(self.null_accuracy[label_type], axis=0)

            self.fig_tools.create_sig_2samp_annotate(
                ax=axis,
                arr0=accu2use,
                arr1=null_accu2use,
                coords=(idx, 1.0),
                paired=True,
            )

            self.fig_tools.scatter_plot(
                ax=axis,
                X=idx - self.plot_params["bar"]["offset"],
                Y=accu2use,
                color=color2use,
                s=self.plot_params["scatter"]["s_large"],
                jitter=self.plot_params["jitter"],
            )

            self.fig_tools.scatter_plot(
                ax=axis,
                X=idx + self.plot_params["bar"]["offset"],
                Y=null_accu2use,
                color=color2use,
                s=self.plot_params["scatter"]["s_large"],
                jitter=self.plot_params["jitter"],
            )

            self.fig_tools.bar_plot(
                ax=axis,
                X=idx - self.plot_params["bar"]["offset"],
                Y=np.mean(accu2use),
                yerr=np.std(accu2use) / np.sqrt(len(accu2use)),
                width=self.plot_params["bar"]["width"],
                alpha=self.plot_params["bar"]["alpha"],
                color=color2use,
            )
            self.fig_tools.bar_plot(
                ax=axis,
                X=idx + self.plot_params["bar"]["offset"],
                Y=np.mean(null_accu2use),
                yerr=np.std(null_accu2use) / np.sqrt(len(null_accu2use)),
                color=color2use,
                hatch=self.plot_params["bar"]["hatch"],
                width=self.plot_params["bar"]["width"],
                alpha=self.plot_params["bar"]["alpha"],
            )

        axis.set_xticks([0, 1])
        axis.set_xticklabels(
            self.plot_params["cat_names4plot"],
            fontsize=self.plot_params["fsize"]["axis"],
        )

        axis.set_ylabel("Accuracy", fontsize=self.plot_params["fsize"]["axis"])

        title = f"{self.folder_name}"
        title_2ndLYR = [f"Decoder: {self.decoder_type}"]
        if self.params4decoder is not None:
            title_2ndLYR += [
                f"{key}: {value}" for key, value in self.params4decoder.items()
            ]
        title_2ndLYR = " | ".join(title_2ndLYR)
        title_all = f"{title}\n{title_2ndLYR}"

        # title = self.utils.create_multiline_string(title)
        axis.set_title(title_all, fontsize=self.plot_params["fsize"]["title"])

        # axis.legend()

        # self.print_wFrm("Saving figure")
        self.fig_tools.save_figure(
            fig, f"Decoder_Results_{self.decoder_type}", figure_save_path=self.FigPath
        )

    def _plot_similarity_matrix4OdorDetails(self) -> None:
        """
        Plots the similarity matrix with clustering and projection weights.
        Top row: Similarity matrix with label bars.
        Bottom row: Projection weights (shorter height).
        """
        # --- Setup Figure and GridSpec ---
        fig, gs = self.fig_tools.create_plt_GridSpec(
            nrows=2,
            ncols=2,
            figsize=(12, 8),
            height_ratios=[3, 1],
            width_ratios=[1, 1],
        )

        # Create axes using the GridSpec
        # length of all_axes is 4
        # all_axes[0] is the top left axis
        # all_axes[1] is the top right axis
        # all_axes[2] is the bottom left axis
        # all_axes[3] is the bottom right axis
        all_axes = self.fig_tools.add_suplot_to_figViaGridSpec(fig=fig, gs=gs)

        # Combine axes for easier looping if needed
        top_axes = all_axes[:2]
        bottom_axes = all_axes[2:]

        self.print_wFrm("Plotting similarity matrix and weights")

        cmap4labels = self.plot_params["cmap"]["odors"]

        for idx, label_cat in enumerate(self.Labels.keys()):
            ax_top = top_axes[idx]  # Axis for similarity heatmap
            ax_bottom = bottom_axes[idx]  # Axis for weights plot

            # Get labels for the current category
            # unique category labels should have same range of values for true and pred
            true_label = self.Labels[label_cat].astype(int)
            pred_label = self.corrected_cluster_labels[label_cat].astype(int)

            # Determine which colormap to use based on the label category
            if label_cat == "ODORS":
                # Create a colormap using only the first 2 colors
                cmap_to_use = ListedColormap(cmap4labels.colors[:2])
            else:
                # Use the full 4-color map for ODORSwSWITCH
                cmap_to_use = cmap4labels

            # Plot similarity matrix
            im = self.fig_tools.plot_imshow(
                fig=fig,
                axis=ax_top,
                data2plot=self.similarity_matrix,
                cmap=self.plot_params["cmap"]["imshow"],
                return_im=True,
            )

            # Create locator
            divider = self.fig_tools.makeAxesLocatable(ax_top)

            # Create axes for true and predicted labels
            ax_true_bar = self.fig_tools.append_axes2locator(
                locator=divider,
                position="top",
                size=self.plot_params["locator"]["size"],
                pad=self.plot_params["locator"]["pad"],
                sharex=ax_top,
            )
            ax_pred_bar = self.fig_tools.append_axes2locator(
                locator=divider,
                position="left",
                size=self.plot_params["locator"]["size"],
                pad=self.plot_params["locator"]["pad"],
                sharey=ax_top,
            )

            # Plot true and predicted labels using the determined colormap
            ax_true_bar.imshow(
                true_label[np.newaxis, :], cmap=cmap_to_use, aspect="auto"
            )
            ax_pred_bar.imshow(
                pred_label[:, np.newaxis], cmap=cmap_to_use, aspect="auto"
            )

            # Cleanup for top plot
            for ax_bar in [ax_true_bar, ax_pred_bar, ax_top]:
                ax_bar.tick_params(axis="x", labelbottom=False, bottom=False)
                ax_bar.tick_params(axis="y", labelleft=False, left=False)

            ax_true_bar.set_title("True")
            ax_pred_bar.set_ylabel("Pred", rotation=90, size="large")

            # Create color bar
            cax = self.fig_tools.append_axes2locator(
                locator=divider,
                position="right",
                size=self.plot_params["locator"]["size"],
                pad=0.1,
            )
            fig.colorbar(im, cax=cax)

            # Set title for top plot, adjust y to avoid overlap with true labels
            ax_top.set_title(self.plot_params["cat_names4plot"][idx], y=1.1)

            # Plot weights
            # transpose to get shape to (n_clusters, n_odor_exposures)
            weights2use = self.projection_weights[label_cat].T

            # Iterate through weights for each cluster/mean map
            for weight_idx, weight_vec in enumerate(weights2use):
                color = cmap4labels(weight_idx)
                weight_plot = self.dep.normalize_array_MINMAX(weight_vec)
                ax_bottom.plot(
                    weight_plot, color=color, alpha=0.7, label=f"Weight {weight_idx}"
                )

            # Cleanup for bottom plot
            ax_bottom.set_yticks([0, 1])  # Set Y-axis ticks to only 0 and 1
            ax_bottom.set_ylabel("Weight")
            ax_bottom.set_xlabel("Odor Exposure")
            ax_bottom.spines["top"].set_visible(False)
            ax_bottom.spines["right"].set_visible(False)

        fig.suptitle(f"Similarity & Projection Weights - {self.folder_name}")
        # Use constrained_layout or tight_layout with adjustments
        self.fig_tools.tighten_layoutWspecific_axes(coords2tighten=[0, 0.03, 1, 0.95])

        self.fig_tools.save_figure(
            fig, "Similarity_Matrix", figure_save_path=self.FigPath
        )

    def _plot_umap_projection4OdorDetails(self) -> None:
        """
        Performs UMAP dimensionality reduction based on similarity matrix made for Odor Details (OdorPeaksNTimes) and plots the 2D embedding with true labels coloring.
        """
        self.print_wFrm("Plotting UMAP projection based on odor peaks & times")
        n_cats = len(self.Labels.keys())
        fig, axes = self.fig_tools.create_plt_subplots(
            ncols=n_cats, figsize=(6 * n_cats, 5), flatten=True
        )

        self.print_wFrm("Finding distance matrix", frame_num=1)
        # UMAP works with distances (0 = close, high = far)
        distance_matrix = self.dep.create_distance_matrix_from_similarity_matrix(
            self.similarity_matrix
        )

        self.print_wFrm("Finding UMAP embedding", frame_num=1)
        # extract UMAP embedding
        embedding = self.dep.UMAP_fit2distMatrix(distance_matrix)

        for idx, (ax, label_cat) in enumerate(zip(axes, self.Labels.keys())):
            cmap4labels = self.plot_params["cmap"]["odors"]
            true_label = self.Labels[label_cat].astype(int)
            unique_labels = np.unique(true_label).astype(int)

            colors = [cmap4labels(int(lab)) for lab in true_label]

            # Scatter plot of the embedding
            ax.scatter(
                embedding[:, 0],
                embedding[:, 1],
                color=colors,
                s=self.plot_params["scatter"]["s_small"],
                alpha=self.plot_params["scatter"]["alpha"],
            )

            # Create legend
            legend_elements = []
            facecolor = [cmap4labels(lab) for lab in unique_labels]
            label = self.Label_Names[label_cat]
            edgecolor = facecolor
            marker = ["o"] * len(facecolor)
            legend_elements = self.fig_tools.create_legend_patch_fLoop(
                facecolor=facecolor,
                label=label,
                edgecolor=edgecolor,
                marker=marker,
            )
            ax.legend(handles=legend_elements)

            ax.set_title(
                f"{self.plot_params['cat_names4plot'][idx]}\nAccuracy: {self.clustering_accuracy[label_cat]:.3f}",
                fontsize=self.plot_params["fsize"]["title"],
            )
            ax.set_xlabel(
                "UMAP 1",
                fontsize=self.plot_params["fsize"]["axis"],
                fontweight="bold",
            )
            ax.set_ylabel(
                "UMAP 2", fontsize=self.plot_params["fsize"]["axis"], fontweight="bold"
            )
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # Remove tick labels (numbers) and tick marks from axes
            ax.tick_params(axis="x", labelbottom=False, bottom=False)
            ax.tick_params(axis="y", labelleft=False, left=False)

        fig.suptitle(
            f"UMAP Projection (Odor Details) - {self.folder_name}",
            fontsize=self.plot_params["fsize"]["title"],
        )

        self.fig_tools.tighten_layoutWspecific_axes(coords2tighten=[0, 0, 1, 0.96])

        self.fig_tools.save_figure(
            fig, "UMAP_Projection_OdorDetails", figure_save_path=self.FigPath
        )

    def _plot_umap_projection4PosRates(self) -> None:
        """
        Performs UMAP dimensionality reduction based on similarity matrix made for Position Rates (PCLappedSess) and plots the 2D embedding with true labels coloring.
        """
        self.print_wFrm("Plotting UMAP projection based on position rates")

        self.print_wFrm("Finding distance matrix", frame_num=1)
        distance_matrix = []
        for lName in self.lapTypeName:
            curr_simMat = self.simMat4PRClustering[lName]
            curr_distMat = self.dep.create_distance_matrix_from_similarity_matrix(
                curr_simMat
            )
            distance_matrix.append(curr_distMat)

        dist4concat = self.dep.create_distance_matrix_from_similarity_matrix(
            self.simMat4PRClustering["concat"]
        )

        self.print_wFrm("Finding UMAP embedding", frame_num=1)
        embeddings = []
        for distMat in distance_matrix:
            embedding = self.dep.UMAP_fit2distMatrix(distMat)
            embeddings.append(embedding)

        embedding_concat = self.dep.UMAP_fit2distMatrix(dist4concat)

        fig, ax = self.fig_tools.create_plt_subplots(
            ncols=2, flatten=True, figsize=(12, 8)
        )

        ax4mult = ax[0]
        ax4single = ax[1]

        cmap4lapType = self.plot_params["cmap"]["lapType"]

        cmap4cellTypes = self.plot_params["cmap"]["cellTypes"]

        colors = [cmap4lapType(int(idx)) for idx in range(len(self.lapTypeName))]

        colors4cellTypes = [
            cmap4cellTypes(int(idx)) for idx in range(len(self.cellTypes))
        ]

        for idx, embedding in enumerate(embeddings):
            for ctype in self.cellTypes:
                ax4mult.scatter(
                    embedding[self.cellTypeInds[ctype], 0],
                    embedding[self.cellTypeInds[ctype], 1],
                    color=colors[idx],
                    s=self.plot_params["scatter"]["s_small"],
                    alpha=self.plot_params["scatter"]["alpha"],
                    marker=self.plot_params["markers"][ctype],
                )

        for cidx, ctype in enumerate(self.cellTypes):
            ax4single.scatter(
                embedding_concat[self.cellTypeInds[ctype], 0],
                embedding_concat[self.cellTypeInds[ctype], 1],
                color=colors4cellTypes[cidx],
                s=self.plot_params["scatter"]["s_small"],
                alpha=self.plot_params["scatter"]["alpha"],
                marker=self.plot_params["markers"][ctype],
            )

        # Create legend
        legend_elements = []
        facecolor = colors
        legend_elements = self.fig_tools.create_legend_patch_fLoop(
            facecolor=facecolor,
            label=[lName for lName in self.lapTypeName],
            edgecolor=facecolor,
            marker=["o"] * len(facecolor),
        )
        ax4mult.legend(handles=legend_elements)

        legend_elements4single = []
        facecolor4single = colors4cellTypes
        marker4single = [self.plot_params["markers"][ctype] for ctype in self.cellTypes]
        legend_elements4single = self.fig_tools.create_legend_patch_fLoop(
            facecolor=facecolor4single,
            label=self.cellTypes,
            edgecolor=facecolor4single,
            marker=marker4single,
        )
        ax4single.legend(handles=legend_elements4single)

        ax4mult.set_title(
            "lapTypes separated", fontsize=self.plot_params["fsize"]["title"]
        )
        ax4single.set_title(
            f"concatenated\nAccuracy: {self.cluster_accuracy4PR:.3f}",
            fontsize=self.plot_params["fsize"]["title"],
        )

        for a in [ax4mult, ax4single]:
            a.set_xlabel(
                "UMAP 1",
                fontsize=self.plot_params["fsize"]["axis"],
                fontweight="bold",
            )
            a.set_ylabel(
                "UMAP 2", fontsize=self.plot_params["fsize"]["axis"], fontweight="bold"
            )
            a.spines["top"].set_visible(False)
            a.spines["right"].set_visible(False)

            # Remove tick labels (numbers) and tick marks from axes
            a.tick_params(axis="x", labelbottom=False, bottom=False)
            a.tick_params(axis="y", labelleft=False, left=False)

        fig.suptitle(
            f"UMAP Projection (Position Rates) - {self.folder_name}",
            fontsize=self.plot_params["fsize"]["title"],
        )

        self.fig_tools.save_figure(
            fig, "UMAP_Projection_PosRates", figure_save_path=self.FigPath
        )

    def _plotConfusionMatrix(self) -> None:
        """
        Plot the confusion matrices.

        This method plots the confusion matrices based on the stored data in the `confusion_matrix` attribute.
        It creates a figure with subplots for each confusion matrix and saves the figure.

        Returns:
            None
        """
        self.print_wFrm("Confusion Matrices")

        n2use = len(self.confusion_matrix.keys())

        fig, axes = self.fig_tools.create_plt_subplots(
            nrows=n2use,
            ncols=2,
        )
        cm_suptitle = self.utils.create_multiline_string(
            [f"Confusion Matrices: {self.folder_name}", f"Decoder: {self.decoder_type}"]
        )
        fig.suptitle(cm_suptitle)
        axes = axes.flatten()
        for idx, (_, cm) in enumerate(self.confusion_matrix.items()):
            plt_idx_cm = idx * 2
            plt_idx_metrics = plt_idx_cm + 1
            ax_cm = axes[plt_idx_cm]
            ax_metrics = axes[plt_idx_metrics]

            if cm.ndim == 4:
                axis2sum = (0, 1)
            elif cm.ndim == 3:
                axis2sum = 0
            aggregate_cm = np.sum(cm, axis=axis2sum)
            cm_metrics = GeneralDecoder.calculate_ConfusionMatrix_metrics(aggregate_cm)

            # plot confusion matrix
            if aggregate_cm.shape[0] == 2:
                group_labels = self.Label_Names["ODORS"]
            elif aggregate_cm.shape[0] == 4:
                group_labels = self.Label_Names["ODORSwSWITCH"]
            self.fig_tools.plot_confusion_matrix(
                ax=ax_cm, cm=aggregate_cm, group_labels=group_labels
            )

            # plot metrics
            # metrics2bar = ["accuracy", "precision", "recall"]
            # values = [cm_metrics[key] for key in metrics2bar]
            precision = {}
            recall = {}
            accuracy = cm_metrics["accuracy"]
            for idx, group in enumerate(group_labels):
                precision[group] = cm_metrics["precision"][idx]
                recall[group] = cm_metrics["recall"][idx]
            metrics2bar = (
                ["accu"]
                + [f"prec_{key}" for key in precision.keys()]
                + [f"recall_{key}" for key in recall.keys()]
            )
            values = (
                [accuracy]
                + [precision[key] for key in precision.keys()]
                + [recall[key] for key in recall.keys()]
            )

            self.fig_tools.bar_plot(
                ax=ax_metrics,
                X=metrics2bar,
                Y=values,
                ylim=(0, 1),
            )
            # Explicitly set tick locations before setting labels to address UserWarning
            ax_metrics.set_xticks(np.arange(len(metrics2bar)))
            ax_metrics.set_xticklabels(metrics2bar, rotation=45)

            textstr = [
                f"F1 {key}: {cm_metrics['f1'][idx]:.2f}"
                for idx, key in enumerate(precision.keys())
            ]
            # join strings together with \n
            textstr = self.utils.create_multiline_string(textstr)

            self.fig_tools.add_text_box(
                ax=ax_metrics,
                text=textstr,
                fontsize=8,
                transform=axes[plt_idx_metrics].transAxes,
                va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        self.fig_tools.save_figure(
            fig, f"Confusion_Matrix_{self.decoder_type}", figure_save_path=self.FigPath
        )

    def _plot_histograms4DistanceMaps(self) -> None:
        """
        Plots histograms of distance maps for different lap types.

        This method creates a figure with subplots for each distance map and saves the figure.
        """
        self.print_wFrm("Plotting histograms of distance maps")

        fig, axes = self.fig_tools.create_plt_subplots(
            ncols=len(self.DistanceMapBTW_BCnSW.keys()),
            nrows=3,
            figsize=(20, 16),
        )

        # NA_idx = self.CueCellTableDict["NON_IDX"]

        diff_embed = {
            "euclidean": None,
            "cosine": None,
            "correlation": None,
        }
        for metric in diff_embed.keys():
            diff_embed[metric] = self.dep.UMAP_fit2distMatrix(
                self.BCSW_comparison["DIFF"], metric=metric
            )

        labels_to_use = self.cellTypes
        colors = [
            self.plot_params["cmap"]["cellTypes"](int(idx))
            for idx in range(len(labels_to_use))
        ]

        histo_axes = axes[0, :]
        bar1_axes = axes[1, :]
        umap_axes = axes[2, :]

        for idx, (key, distMap) in enumerate(self.DistanceMapBTW_BCnSW.items()):
            ax_histo = histo_axes[idx]
            ax_bar = bar1_axes[idx]
            data_to_plot = {
                ctype: np.array([distMap[idx] for idx in self.cellTypeInds[ctype]])
                for ctype in self.cellTypes
            }
            means = {
                ctype: np.mean(data_to_plot[ctype])
                for ctype in self.cellTypes
                if len(data_to_plot[ctype]) > 0
            }
            sterr = {
                ctype: np.std(data_to_plot[ctype]) / np.sqrt(len(data_to_plot[ctype]))
                for ctype in self.cellTypes
                if len(data_to_plot[ctype]) > 0
            }

            # self.fig_tools.create_sig_2samp_annotate(
            #     ax=ax_bar,
            #     arr0=data_to_plot[0],
            #     arr1=data_to_plot[1],
            #     coords=(0.5, None),
            #     parametric=False,
            # )

            for cidx, ctype in enumerate(self.cellTypes):
                self.fig_tools.bar_plot(
                    ax=ax_bar,
                    X=labels_to_use[cidx],
                    Y=means[ctype],
                    yerr=sterr[ctype],
                    color=colors[cidx],
                    alpha=self.plot_params["bar"]["alpha"],
                )
                self.fig_tools.scatter_plot(
                    ax=ax_bar,
                    X=cidx,
                    Y=data_to_plot[ctype],
                    color=colors[cidx],
                    s=self.plot_params["scatter"]["s_small"],
                    # alpha=self.plot_params["scatter"]["alpha"],
                )

            ax_bar.set_ylabel("Distance")
            ax_bar.set_xlabel("Cell Type")
            ax_bar.set_xticks(np.arange(len(labels_to_use)))
            ax_bar.set_xticklabels(labels_to_use)

            data4histo = [
                data_to_plot[ctype]
                for ctype in self.cellTypes
                if len(data_to_plot[ctype]) > 0
            ]
            ax_histo.hist(
                data4histo,
                bins=20,
                color=colors,
                label=labels_to_use,
                histtype="barstacked",
                edgecolor="black",
            )

            ax_histo.set_title(f"{key}")
            ax_histo.set_xlabel("Distance")
            ax_histo.set_ylabel("Count")
            ax_histo.legend()

        for uidx, (key, embed) in enumerate(diff_embed.items()):
            for iidx, cidx in enumerate(self.cellTypes):
                umap_axes[uidx].scatter(
                    embed[self.cellTypeInds[cidx], 0],
                    embed[self.cellTypeInds[cidx], 1],
                    color=colors[iidx],
                    s=self.plot_params["scatter"]["s_large"],
                )
            if key == "euclidean":
                accuKey2use = "EUCLID"
            elif key == "cosine":
                accuKey2use = "COSINE"
            elif key == "correlation":
                accuKey2use = "1-CORR"

            umap_axes[uidx].set_title(
                f"Clustering Accuracy: {self.cluster_accuracy4BC_SW[accuKey2use]:.3f}"
            )
            umap_axes[uidx].set_xlabel(
                "UMAP 1",
                fontsize=self.plot_params["fsize"]["axis"],
                fontweight="bold",
            )
            umap_axes[uidx].set_ylabel(
                "UMAP 2", fontsize=self.plot_params["fsize"]["axis"], fontweight="bold"
            )
            umap_axes[uidx].spines["top"].set_visible(False)
            umap_axes[uidx].spines["right"].set_visible(False)
            # Remove tick labels (numbers) and tick marks from axes
            umap_axes[uidx].tick_params(axis="x", labelbottom=False, bottom=False)
            umap_axes[uidx].tick_params(axis="y", labelleft=False, left=False)

        self.fig_tools.save_figure(
            fig, "Distance_Maps_BothcuesVSswitch", figure_save_path=self.FigPath
        )

    def _plot_minmax_diff(self) -> None:
        """
        Plots the min-max difference between L1 and L2 for each cell type.
        """
        self.print_wFrm("Plotting min-max difference")
        labels_to_use = self.cellTypes
        colors = [
            self.plot_params["cmap"]["cellTypes"](int(idx))
            for idx in range(len(labels_to_use))
        ]

        barTypes = [("MINMAXDIFF", "Min-Max Difference")]
        fig, ax = self.fig_tools.create_plt_subplots(nrows=len(barTypes))
        bar2_ax = ax

        for idx, (diff_type, diff_name) in enumerate(barTypes):
            bar2_ax.set_title(diff_name)
            bar2_ax.set_xlabel("Cell Type")
            bar2_ax.set_ylabel(diff_name)
            bar2_ax.set_xticks(np.arange(len(labels_to_use)))
            bar2_ax.set_xticklabels(labels_to_use)

            for cidx, ctype in enumerate(self.cellTypes):
                data2use = [
                    self.BCSW_comparison[diff_type]["L1"][self.cellTypeInds[ctype]],
                    self.BCSW_comparison[diff_type]["L2"][self.cellTypeInds[ctype]],
                ]
                means = [np.nanmean(data2use[0]), np.nanmean(data2use[1])]
                sterr = [
                    np.nanstd(data2use[0]) / np.sqrt(len(data2use[0])),
                    np.nanstd(data2use[1]) / np.sqrt(len(data2use[1])),
                ]
                for iidx, (d2plot, mean, sterr) in enumerate(
                    zip(data2use, means, sterr)
                ):
                    if iidx == 0:
                        off = -1
                    elif iidx == 1:
                        off = 1
                    self.fig_tools.bar_plot(
                        ax=bar2_ax,
                        X=cidx + off * self.plot_params["bar"]["offset"],
                        Y=mean,
                        yerr=sterr,
                        width=self.plot_params["bar"]["width"],
                        color=colors[cidx],
                        alpha=self.plot_params["bar"]["alpha"],
                    )
                    self.fig_tools.scatter_plot(
                        ax=bar2_ax,
                        X=cidx + off * self.plot_params["bar"]["offset"],
                        Y=d2plot,
                        color=colors[cidx],
                        s=self.plot_params["scatter"]["s_small"],
                        jitter=0,
                    )
                bar2_ax.plot(
                    [
                        cidx - self.plot_params["bar"]["offset"],
                        cidx + self.plot_params["bar"]["offset"],
                    ],
                    [data2use[0], data2use[1]],
                    color=colors[cidx],
                )
                bar2_ax.set_ylim(-1.1, 1.1)  # Adjust limits to make space

                # Annotation for L1 dominance (top)
                bar2_ax.annotate(
                    "More L1 active",
                    xy=(1.02, 0.9),
                    xycoords="axes fraction",
                    xytext=(10, 0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    arrowprops=dict(arrowstyle="->", color="black"),
                )

                # Annotation for L2 dominance (bottom)
                bar2_ax.annotate(
                    "More L2 active",
                    xy=(1.02, 0.1),
                    xycoords="axes fraction",
                    xytext=(10, 0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    arrowprops=dict(arrowstyle="->", color="black"),
                )

        self.fig_tools.save_figure(fig, "MinMaxDiff", figure_save_path=self.FigPath)

    def analyzing_BCSwitchTuning(self) -> None:
        """
        Analyzes the tuning similarity between different cell types.
        """
        self.print_wFrm("Plotting tuning similarity by cell type/cluster")
        labels_to_use = self.cellTypes_wNon
        colors = [
            self.plot_params["cmap"]["cellTypes_wNon"](int(idx))
            for idx in range(len(labels_to_use))
        ]
        fig, ax = self.fig_tools.create_plt_subplots()

        PerBinTuning = np.zeros(
            (len(labels_to_use), self.BCSW_comparison["DIFF"].shape[1])
        )
        for cidx, ctype in enumerate(np.unique(self.true_cellTypeLabel)):
            cell_indices = np.where(self.true_cellTypeLabel == ctype)[0]
            cell_tuning_per_bin = np.nanmean(
                self.BCSW_comparison["DIFF"][cell_indices], axis=0
            )
            PerBinTuning[cidx, :] = cell_tuning_per_bin

        im = self.fig_tools.plot_imshow(
            fig=fig,
            axis=ax,
            data2plot=PerBinTuning,
            cmap="RdYlGn",
            aspect="auto",
            title="BC-SW Tuning by cell type",
            xlabel="Position",
            ylabel="Cell Type",
            yticks=np.arange(len(labels_to_use)),
            return_im=True,
        )

        fig.colorbar(im, ax=ax, label="BC-SW Tuning")
        ax.set_yticklabels(labels_to_use)

        self.fig_tools.save_figure(
            fig, "BCSwitch_Tuning", figure_save_path=self.FigPath
        )

    def analyze_tuning_similarity_by_cell_type(self):
        """
        Calculates and compares the average spatial tuning similarity (from concatenated
        tuning curves across lap types) within and between different cell types
        (Cue, Place, NA).
        """

        def get_similarity_values(indices1, indices2, matrix):
            """Extracts similarity values between two sets of indices."""
            if len(indices1) == 0 or len(indices2) == 0:
                # return empty array if no cells in one group
                return np.array([])

            sub_matrix = matrix[np.ix_(indices1, indices2)]

            if np.array_equal(indices1, indices2):
                # Within-group: get upper triangle excluding diagonal (k=1)
                if sub_matrix.shape[0] < 2:
                    return np.array([])  # Need at least 2 cells for pairwise similarity
                vals = sub_matrix[np.triu_indices_from(sub_matrix, k=1)]
            else:
                # Between-group: get all values from the rectangular submatrix
                vals = sub_matrix.flatten()
            return vals[~np.isnan(vals)]  # Remove NaNs if any

        def print_stats(name: str, data: np.ndarray) -> None:
            if len(data) > 0:
                self.print_wFrm(
                    f"  {name:<15}: {np.mean(data):.3f} +/- {np.std(data) / np.sqrt(len(data)):.3f} (n={len(data)})",
                    frame_num=2,
                )

        def print_comparison(name: str, data1: np.ndarray, data2: np.ndarray) -> None:
            if len(data1) > 0 and len(data2) > 0:
                self.print_wFrm(
                    f"  {name:<25}: p = {mannwhitneyu(data1, data2, alternative='two-sided').pvalue:.4f}",
                    frame_num=2,
                )

        self.rprint("Analyzing spatial tuning similarity by cell type:")

        sim_matrix = self.simMat4PRClustering.get("concat", None)
        cell_dict = self.CueCellTableDict

        if sim_matrix is None:
            self.print_wFrm(
                "Concatenated similarity matrix not found. Skipping analysis.",
                frame_num=1,
            )
            return None

        if not cell_dict:
            self.print_wFrm(
                "CueCellTableDict not found. Skipping analysis.", frame_num=1
            )
            return None

        cue_indices = np.array(cell_dict.get("CUE_IDX", []))
        place_indices = np.array(cell_dict.get("PLACE_IDX", []))
        n_cells = sim_matrix.shape[0]
        all_indices = np.arange(n_cells)
        na_indices = np.setdiff1d(
            all_indices, np.concatenate((cue_indices, place_indices))
        )

        self.print_wFrm(
            f"Found {len(cue_indices)} Cue, {len(place_indices)} Place, {len(na_indices)} NA cells.",
            frame_num=1,
        )

        results = {}

        # --- 3. Calculate Within-Group Similarities ---
        sim_within_cue = get_similarity_values(cue_indices, cue_indices, sim_matrix)
        sim_within_place = get_similarity_values(
            place_indices, place_indices, sim_matrix
        )
        sim_within_na = get_similarity_values(na_indices, na_indices, sim_matrix)

        results["Within_Cue_Sim"] = sim_within_cue
        results["Within_Place_Sim"] = sim_within_place
        results["Within_NA_Sim"] = sim_within_na

        # --- 4. Calculate Between-Group Similarities ---
        sim_cue_place = get_similarity_values(cue_indices, place_indices, sim_matrix)
        sim_cue_na = get_similarity_values(cue_indices, na_indices, sim_matrix)
        sim_place_na = get_similarity_values(place_indices, na_indices, sim_matrix)

        results["Cue_Place_Sim"] = sim_cue_place
        results["Cue_NA_Sim"] = sim_cue_na
        results["Place_NA_Sim"] = sim_place_na

        # --- 5. Report Mean Similarities ---
        self.print_wFrm("Mean Similarities (Concatenated Tuning Curves):", frame_num=1)
        print_stats("Within Cue", sim_within_cue)

        print_stats("Within Cue", sim_within_cue)
        print_stats("Within Place", sim_within_place)
        print_stats("Within NA", sim_within_na)
        print_stats("Cue vs Place", sim_cue_place)
        print_stats("Cue vs NA", sim_cue_na)
        print_stats("Place vs NA", sim_place_na)

        self.print_wFrm("Statistical Comparisons (Mann-Whitney U):", frame_num=1)
        print_comparison("Within-Cue vs Cue-Place", sim_within_cue, sim_cue_place)
        print_comparison("Within-Place vs Cue-Place", sim_within_place, sim_cue_place)
        print_comparison("Within-Cue vs Within-Place", sim_within_cue, sim_within_place)

        self.print_done_small_proc()


if __name__ == "__main__":
    run_CLAH_script(TwoOdorDecoder, parser_enum=decoder_enum.Parser4TOD_DEV)

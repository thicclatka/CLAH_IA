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
        # self.extractLapsByType()
        self.findPosRateClustering()
        self.findOdorClustering()
        self.DecodeOdorEpochs()
        # self.DecodeLaps_viaLSTM()
        # self.DecodePosition()
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

        self.plot_params = {
            "bar": {
                "alpha": 0.7,
                "width": 0.2,
                "offset": 0.1,
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
                "posrates": self.fig_tools.create_cmap4categories(
                    num_categories=5, cmap_name="tab10"
                ),
                "cellTypes": self.fig_tools.create_cmap4categories(
                    num_categories=3, cmap_name="tab20"
                ),
            },
            "locator": {
                "size": "3%",
                "pad": 0.05,
            },
            "jitter": 0.02,
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

        CCTD = self.findLatest(["CueCellTableDict"], path2check="Figures")
        with open(CCTD, "rb") as f:
            self.CueCellTableDict = json.load(f)

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

    def extractLapsByType(self) -> None:
        """
        Extracts laps by type from the lapTypeArr.
        """
        CTEMP2USE = self.C_Temp.copy()
        for cell in range(CTEMP2USE.shape[0]):
            CTEMP2USE[cell, :] = (CTEMP2USE[cell, :] - np.min(CTEMP2USE[cell, :])) / (
                np.max(CTEMP2USE[cell, :]) - np.min(CTEMP2USE[cell, :])
            )
        self.lapsByType = {lName: [] for lName in self.lapTypeName}
        for lidx, (lType, lTime) in enumerate(zip(self.lapTypeArr, self.lapTimes)):
            lName = self.lapTypeName[lType - 1]
            start_time = lTime

            if lidx == len(self.lapTypeArr) - 2:
                end_time = self.adjFrTimes[-1]
            elif lidx == len(self.lapTypeArr) - 1:
                continue
            else:
                end_time = self.lapTimes[lidx + 1]

            lap_indices = np.where(
                (self.adjFrTimes >= start_time) & (self.adjFrTimes <= end_time)
            )[0]
            lap_data = CTEMP2USE[:, lap_indices]
            self.lapsByType[lName].append(lap_data)

        seq_len = self.LSTM_params["seq_len"]
        self.LSTM = {"train": [], "target": [], "labels": []}
        for idx, lName in enumerate(self.lapTypeName):
            curr_laps = self.lapsByType[lName]

            for lap in curr_laps:
                lapT = lap.T

                for i in range(0, lapT.shape[0] - seq_len, seq_len):
                    self.LSTM["train"].append(lapT[i : i + seq_len])
                    self.LSTM["target"].append(lapT[i + seq_len])
                    self.LSTM["labels"].append(lName)

        self.X_LSTM = np.stack(self.LSTM["train"], axis=0)
        self.y_LSTM = np.stack(self.LSTM["target"], axis=0)
        self.labels_LSTM = np.array(self.LSTM["labels"])

    def findPosRateClustering(self) -> None:
        """
        Creates a feature vector for position decoder.
        """
        print("Creating similarity matrices from spatial firing rate data")
        self.simMat4PRClustering = {lName: None for lName in self.lapTypeName}
        self.simMat4PRClustering["concat"] = []
        FVconcat2simMat = []
        for lidx in np.unique(self.lapTypeArr):
            curr_lName = self.lapTypeName[lidx - 1]
            # curr_posRateByLap = self.PCLappedSess[f"lapType{lidx}"]["ByLap"]["posRates"]
            FV = self.PCLappedSess[f"lapType{lidx}"]["posRates"]
            (N_cells, P_bins) = FV.shape
            # --- Feature Scaling (Min-Max per cell/feature) on Downsampled FV ---
            FV_scaled = np.zeros_like(FV)
            for cell_idx in range(N_cells):
                cell_data = FV[cell_idx, :]
                min_val = np.min(cell_data)
                max_val = np.max(cell_data)
                range_val = max_val - min_val

                if range_val > 0:
                    FV_scaled[cell_idx, :] = (cell_data - min_val) / range_val
                else:
                    FV_scaled[cell_idx, :] = 0
            simMat = self.dep.calc_simMatrix(FV_scaled)
            self.simMat4PRClustering[curr_lName] = simMat
            FVconcat2simMat.append(FV_scaled)
        FVconcat2simMat = np.concatenate(FVconcat2simMat, axis=1)
        self.simMat4PRClustering["concat"] = self.dep.calc_simMatrix(FVconcat2simMat)
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
            self.cluster_labels[label_cat] = self.dep.SpectralClustering_fit2simMatrix(
                similarity_matrix=self.similarity_matrix,
                n_clusters=n_clusters,
                random_state=self.random_state,
            )

            # Calculate accuracy & get row & col indices from linear_sum_assignment of confusion matrix
            self.clustering_accuracy[label_cat] = (
                self.dep.determineSpectralClustering_accuracy(
                    true_labels=self.Labels[label_cat],
                    cluster_labels=self.cluster_labels[label_cat],
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

    def DecodeLaps_viaLSTM(self) -> None:
        """
        Decodes the laps using LSTM.
        """
        print(self.LSTM_params)
        mse, mae, r2 = GeneralDecoder.decode_via_LSTM_prediction(
            X_lstm=self.X_LSTM,
            y_lstm=self.y_LSTM,
            sequence_labels=self.labels_LSTM,
            num_folds=self.num_folds4switch,
            random_state=self.random_state,
            **self.LSTM_params,
        )
        pass

    # def DecodePosition(self) -> None:
    #     """
    #     Decodes the position using LSTM.
    #     """
    #     self.rprint("Decoding position")
    #     self.PositionDecoderResults = {lName: {} for lName in self.lapTypeName}
    #     for lName in self.lapTypeName:
    #         accu, conf_matrices, medae_scores = GeneralDecoder.run_Decoder(
    #             data_arr=self.FV4PositionDecoder[lName]["features"],
    #             label_arr=self.FV4PositionDecoder[lName]["labels"],
    #             num_folds=self.num_folds4switch,
    #             decoder_type="KNN",
    #             n_neighbors=10,
    #             metric="cosine",
    #         )
    #         self.PositionDecoderResults[lName] = {
    #             "accu": accu,
    #             "conf_matrices": conf_matrices,
    #             "medae_scores": medae_scores,
    #         }
    #     self.print_done_small_proc()

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
        self._plot_similarity_matrixWClustering()

        # UMAP projection plot based on odor details
        self._plot_umap_projection4OdorDetails()

        # UMAP projection plot based on posRates
        self._plot_umap_projection4PosRates()

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

    def _plot_similarity_matrixWClustering(self) -> None:
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
                c=colors,
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

        cmap4labels = self.plot_params["cmap"]["posrates"]

        cmap4cellTypes = self.plot_params["cmap"]["cellTypes"]

        colors = [cmap4labels(int(idx)) for idx in range(len(self.lapTypeName))]

        colors4cellTypes = [cmap4cellTypes(int(idx)) for idx in range(3)]

        cueEmbeds2use = []
        for idx, embedding in enumerate(embeddings):
            cue_idx = self.CueCellTableDict["CUE_IDX"]
            place_idx = self.CueCellTableDict["PLACE_IDX"]
            cue_embedding = embedding[cue_idx, :]
            place_embedding = embedding[place_idx, :]
            rest_embedding = embedding[
                ~np.isin(np.arange(embedding.shape[0]), cue_idx)
                & ~np.isin(np.arange(embedding.shape[0]), place_idx),
                :,
            ]
            ax4mult.scatter(
                cue_embedding[:, 0],
                cue_embedding[:, 1],
                c=colors[idx],
                s=self.plot_params["scatter"]["s_small"] + 5,
                alpha=self.plot_params["scatter"]["alpha"],
                marker="*",
            )
            ax4mult.scatter(
                place_embedding[:, 0],
                place_embedding[:, 1],
                c=colors[idx],
                s=self.plot_params["scatter"]["s_small"],
                alpha=self.plot_params["scatter"]["alpha"],
                marker="+",
            )
            ax4mult.scatter(
                rest_embedding[:, 0],
                rest_embedding[:, 1],
                c=colors[idx],
                s=self.plot_params["scatter"]["s_small"],
                alpha=self.plot_params["scatter"]["alpha"],
                marker="o",
            )
            cueEmbeds2use.append(cue_embedding)

        # for cell_idx in range(self.CueCellTableDict["CUE"]):
        #     x_coords = [embedding[cell_idx, 0] for embedding in cueEmbeds2use]
        #     y_coords = [embedding[cell_idx, 1] for embedding in cueEmbeds2use]
        #     ax4mult.plot(
        #         x_coords,
        #         y_coords,
        #         color="grey",  # Or another subtle color
        #         linewidth=0.5,  # Make lines thin
        #         # marker='o', # Optionally add small markers on the line vertices
        #         # markersize=1
        #     )

        cue_embed_concat = embedding_concat[cue_idx, :]
        place_embed_concat = embedding_concat[place_idx, :]
        rest_embed_concat = embedding_concat[
            ~np.isin(np.arange(embedding_concat.shape[0]), cue_idx)
            & ~np.isin(np.arange(embedding_concat.shape[0]), place_idx),
            :,
        ]

        ax4single.scatter(
            cue_embed_concat[:, 0],
            cue_embed_concat[:, 1],
            c=colors4cellTypes[0],
            s=self.plot_params["scatter"]["s_small"] + 5,
            alpha=self.plot_params["scatter"]["alpha"],
            marker="*",
        )
        ax4single.scatter(
            place_embed_concat[:, 0],
            place_embed_concat[:, 1],
            c=colors4cellTypes[1],
            s=self.plot_params["scatter"]["s_small"],
            alpha=self.plot_params["scatter"]["alpha"],
            marker="x",
        )
        ax4single.scatter(
            rest_embed_concat[:, 0],
            rest_embed_concat[:, 1],
            c=colors4cellTypes[2],
            s=self.plot_params["scatter"]["s_small"],
            alpha=self.plot_params["scatter"]["alpha"],
            marker="o",
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
        legend_elements4single = self.fig_tools.create_legend_patch_fLoop(
            facecolor=facecolor4single,
            label=["cue", "place", "NA"],
            edgecolor=facecolor4single,
            marker=["*", "x", "o"],
        )
        ax4single.legend(handles=legend_elements4single)

        ax4mult.set_title(
            "lapTypes separated", fontsize=self.plot_params["fsize"]["title"]
        )
        ax4single.set_title("concatenated", fontsize=self.plot_params["fsize"]["title"])

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

        # # Hide non-diagonal subplots
        # for i, ax in enumerate(axes):
        #     if i % (n2use + 1) != 0:  # Check if not on diagonal
        #         ax.set_visible(False)  # Hide the subplot

        self.fig_tools.save_figure(
            fig, f"Confusion_Matrix_{self.decoder_type}", figure_save_path=self.FigPath
        )

    def analyze_tuning_similarity_by_cell_type(self):
        """
        Calculates and compares the average spatial tuning similarity (from concatenated
        tuning curves across lap types) within and between different cell types
        (Cue, Place, NA).
        """
        self.rprint("Analyzing spatial tuning similarity by cell type:")

        # --- 1. Get Data ---
        sim_matrix = self.simMat4PRClustering.get("concat", None)
        cell_dict = self.CueCellTableDict

        if sim_matrix is None:
            self.print_wFrm(
                "Concatenated similarity matrix not found. Skipping analysis.",
                frame_num=1,
            )
            return None  # Or raise an error

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

        # --- 2. Define Helper to Get Similarities ---
        def get_similarity_values(indices1, indices2, matrix):
            """Extracts similarity values between two sets of indices."""
            if len(indices1) == 0 or len(indices2) == 0:
                return np.array([])  # Return empty array if no cells in one group

            # Use np.ix_ for safe indexing even if indices are empty
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
        print_stats = lambda name, data: self.print_wFrm(
            f"  {name:<15}: {np.mean(data):.3f} +/- {np.std(data) / np.sqrt(len(data)):.3f} (n={len(data)})"
            if len(data) > 0
            else f"  {name:<15}: N/A (no pairs)",
            frame_num=2,
        )

        print_stats("Within Cue", sim_within_cue)
        print_stats("Within Place", sim_within_place)
        print_stats("Within NA", sim_within_na)
        print_stats("Cue vs Place", sim_cue_place)
        print_stats("Cue vs NA", sim_cue_na)
        print_stats("Place vs NA", sim_place_na)

        # --- 6. (Optional) Statistical Comparisons ---
        self.print_wFrm("Statistical Comparisons (Mann-Whitney U):", frame_num=1)
        print_comparison = lambda name, data1, data2: self.print_wFrm(
            f"  {name:<25}: p = {mannwhitneyu(data1, data2, alternative='two-sided').pvalue:.4f}"
            if len(data1) > 0 and len(data2) > 0
            else f"  {name:<25}: N/A (insufficient data)",
            frame_num=2,
        )

        print_comparison("Within-Cue vs Cue-Place", sim_within_cue, sim_cue_place)
        print_comparison("Within-Place vs Cue-Place", sim_within_place, sim_cue_place)
        print_comparison("Within-Cue vs Within-Place", sim_within_cue, sim_within_place)
        # Add more comparisons as needed (e.g., involving NA cells)

        self.print_done_small_proc()


if __name__ == "__main__":
    run_CLAH_script(TwoOdorDecoder, parser_enum=decoder_enum.Parser4TOD_DEV)

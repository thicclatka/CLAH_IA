import numpy as np
from scipy.stats import ttest_ind
from typing import Literal
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import SpectralClustering
from sklearn.linear_model import LinearRegression
from CLAH_ImageAnalysis.core import BaseClass as BC
from CLAH_ImageAnalysis.core import run_CLAH_script
from CLAH_ImageAnalysis.decoder import decoder_enum
from CLAH_ImageAnalysis.decoder import GeneralDecoder
from CLAH_ImageAnalysis.unitAnalysis import pks_utils
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import umap


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
        self.findOdorClustering()
        self.DecodeOdorEpochs()
        self.plotResults()

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

        self.PKS_UTILS = pks_utils()

        self.sampling_rate = 10  # 10 frames per second (10 Hz; 100 ms per frame)
        self.pre_cue_time = 1  # second before cue start
        self.post_cue_time = 2  # seconds after cue start
        self.trial_dur = (self.pre_cue_time + self.post_cue_time) * self.sampling_rate

        self.decoder_type = "SVC"  # hardcoded for now

        self.num_folds = num_folds
        self.num_fold4switch = None
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
        # 0th column is time
        # 1st column is odor label
        # 2nd column is switch label
        for odor in range(1, 3):
            # get Times, Laps, and Types for each odor
            # types should represent odor & switch
            cueTimes = self.TBD["cueEvents"][f"cue{odor}"]["start"]["Time"]
            cueLaps = self.lapCue[f"CUE{odor}"]["Lap"].astype(int)
            cueTypes = lapTypeArr[cueLaps]

            lapStartIdx = np.searchsorted(lapTimes, cueTimes) - 1
            lapStart = lapTimes[lapStartIdx]
            # Times, laps, types all should have same shape
            num_trials = cueTimes.shape[0]
            # create odor labels
            odor_labels = odor * np.ones(num_trials)
            # switch labels are odor + switch (1, 2 for odor presents at location 1, 2, 3, 4 for switch laps)
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

    def findOdorDetails(self) -> None:
        """
        Organizes times and labels by trial epochs and extracts peak values and times.
        """
        self.rprint("Organizing Times & Labels by trial epochs:")
        self.print_wFrm(
            f"{int(self.trial_dur / self.sampling_rate)} second windows: {self.pre_cue_time} sec pre-cue; {self.post_cue_time} sec post-cue"
        )
        adjFrTimes = np.array(self.TBD["adjFrTimes"])

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
        self.normOdorPeaksNTimes = np.full(
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
                (adjFrTimes >= trial_start) & (adjFrTimes <= trial_end)
            )[0]
            post_cue_indices = np.where(
                (adjFrTimes > cue_start_time) & (adjFrTimes <= trial_end)
            )[0]

            if trial_indices.shape[0] == self.trial_dur:
                # Get trial data
                trial_data = CTEMP2USE[:, trial_indices]
                self.OdorEpochs[:, :, idx] = trial_data

                # Get post-cue data for peak detection
                post_cue_data = CTEMP2USE[:, post_cue_indices]
                post_time_data = adjFrTimes[post_cue_indices]

                # Find peaks and times
                max_peak_val = np.max(post_cue_data, axis=1)
                max_peak_idx = np.argmax(post_cue_data, axis=1)
                max_peak_time = (
                    post_time_data[max_peak_idx] - lap_start_time
                )  # Relative to lap start

                # normalize peak val
                norm_peak_val = (max_peak_val - max_peak_val.min()) / (
                    max_peak_val.max() - max_peak_val.min() + 1e-10
                )

                # Store interleaved peaks and times
                self.normOdorPeaksNTimes[idx] = np.column_stack(
                    (norm_peak_val, max_peak_time)
                ).reshape(-1)
                self.OdorPeaksNTimes[idx] = np.column_stack(
                    (max_peak_val, max_peak_time)
                ).reshape(-1)

                # Store labels
                self.Labels["ODORS"][idx] = odor_label
                self.Labels["ODORSwSWITCH"][idx] = switch_label

        # check # of switchs labels to set num_fold4switch
        switch_labels, counts = np.unique(
            self.Labels["ODORSwSWITCH"], return_counts=True
        )
        if counts[3] < self.num_folds or counts[4] < self.num_folds:
            count_min = np.min(counts[3:])
            self.num_fold4switch = count_min
            output_str = f"Switch labels are not balanced; setting cross-fold validation to {count_min}"
        else:
            output_str = f"Switch labels are balanced; setting cross-fold validation to {self.num_folds}"
            self.num_fold4switch = self.num_folds
        self.print_wFrm(output_str)
        self.print_done_small_proc()

    def findOdorClustering(self) -> None:
        """
        Performs clustering analysis following the paper's methodology:
        1. Computes cosine similarity between trials
        2. Performs spectral clustering
        3. Calculates projection weights
        """
        self.rprint("Finding odor clustering:")

        # 1. Compute cosine similarity matrix
        self.similarity_matrix = cosine_similarity(self.OdorPeaksNTimes)

        # Transform similarity matrix to be non-negative [0, 1] range
        # Cosine similarity ranges from -1 to 1. Shift and scale to 0 to 1.
        if np.min(self.similarity_matrix) < 0:
            self.similarity_matrix = (self.similarity_matrix + 1) / 2

        # Ensure diagonal is 1 after transformation
        np.fill_diagonal(self.similarity_matrix, 1)

        self.cluster_labels = {}
        self.projection_weights = {}
        self.clustering_accuracy = {}

        # 2. Perform spectral clustering
        for label_cat, n_clusters in [
            ("ODORS", int(np.max(self.Labels["ODORS"]))),
            ("ODORSwSWITCH", int(np.max(self.Labels["ODORSwSWITCH"]))),
        ]:
            spectral = SpectralClustering(
                n_clusters=n_clusters, affinity="precomputed", random_state=42
            )
            # Use the non-negative matrix for prediction
            self.cluster_labels[label_cat] = spectral.fit_predict(
                self.similarity_matrix
            )

            # 3. Calculate average maps for each cluster
            mean_map_val = []
            for label in range(n_clusters):
                mean_map = np.mean(
                    self.OdorPeaksNTimes[self.cluster_labels[label_cat] == label],
                    axis=0,
                )
                mean_map_val.append(mean_map)

            # 4. Project each trial onto the average maps
            self.projection_weights[label_cat] = np.zeros(
                (self.OdorPeaksNTimes.shape[0], n_clusters)
            )
            for k in range(self.OdorPeaksNTimes.shape[0]):
                # Multiple linear regression
                X_k = self.OdorPeaksNTimes[k]
                reg = LinearRegression().fit(np.column_stack(mean_map_val), X_k)
                weights = reg.coef_
                # Normalize to L2 norm
                self.projection_weights[label_cat][k] = weights / np.linalg.norm(
                    weights
                )

            # Calculate fraction of correct trials
            correct_trials = np.sum(
                self.cluster_labels[label_cat] == self.Labels[label_cat] - 1
            )  # Assuming labels are 1 and 2
            self.clustering_accuracy[label_cat] = correct_trials / len(
                self.cluster_labels[label_cat]
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
            accuracy, conf_matrices = GeneralDecoder.run_Decoder(
                data_arr=self.normOdorPeaksNTimes,
                label_arr=np.array(label),
                num_folds=folds,
                decoder_type=self.decoder_type,
                **(self.params4decoder if self.params4decoder is not None else {}),
            )
            return accuracy, conf_matrices

        def _determine_fold_count(key: str) -> int:
            """
            Determines the fold count based on the given key.

            Parameters:
            - key (str): The key to determine the fold count for. Should be either "ODORS" or "fold4switch".

            Returns:
            - int: The fold count based on the given key. Returns `self.num_folds` if key is "ODORS", otherwise returns `self.num_fold4switch`.
            """
            return self.num_folds if key == "ODORS" else self.num_fold4switch

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
                data_arr=self.OdorPeaksNTimes,
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
        # needed to use fig_tools
        BC.enable_fig_tools(self)

        # plot decoder accuracy over time w/in epoch
        # self._plotSEM_Accuracy()

        self._plotBar_Accuracy()

        # confusion matrix
        self._plotConfusionMatrix()

        # similarity matrix with clustering
        self._plot_similarity_matrixWClustering()

        # UMAP projection plot
        self._plot_umap_projection()

        self.print_done_small_proc()

    def _plotSEM_Accuracy(self) -> None:
        """
        Plots the standard error of the mean (SEM) for accuracy.

        This method plots the SEM for accuracy based on the data stored in the `accuracy` and `null_accuracy` dictionaries.
        It calculates the p-value using the t-test and adds it to the plot as well.

        Returns:
            None
        """

        def _calc_PVal_via_TTest(test, null):
            """
            Calculate the p-value using a t-test.

            Parameters:
            test (array-like): The test data.
            null (array-like): The null data.

            Returns:
            float: The calculated p-value.
            """
            _, pvalue = ttest_ind(test, null)
            return pvalue

        fig, axis = self.fig_tools.create_plt_subplots()
        x_ind = [
            -self.pre_cue_time * self.sampling_rate,
            self.post_cue_time * self.sampling_rate,
        ]

        odor_color = self.color_dict["blue"]
        switch_color = self.color_dict["orange"]

        self.print_wFrm("SEM plots for accuracy")
        for key, accu in self.accuracy.items():
            self.fig_tools.plot_SEM(
                arr=accu,
                ax=axis,
                color=odor_color if key == "ODORS" else switch_color,
                x_ind=x_ind,
                label=key,
            )

        for key, naccu in self.null_accuracy.items():
            self.fig_tools.plot_SEM(
                arr=naccu,
                ax=axis,
                color=odor_color if key == "ODORS" else switch_color,
                x_ind=x_ind,
                linestyle=":",
                label=f"{key} (null)",
            )

        self.print_wFrm("Determining significance")
        pval = {key: None for key in self.accuracy.keys()}
        pval_text = []
        for key in self.accuracy.keys():
            pval[key] = _calc_PVal_via_TTest(
                np.mean(self.accuracy[key], axis=1),
                np.mean(self.null_accuracy[key], axis=(1, 2)),
            )
            pval_text.append(r"$P_{{\mathrm{{{}}}}} = {:.2e}$".format(key, pval[key]))
        pval_text = self.utils.create_multiline_string(pval_text)

        tick_positions = range(
            -self.pre_cue_time * self.sampling_rate,
            (self.post_cue_time * self.sampling_rate) + 1,
            self.sampling_rate,
        )
        tick_labels = [
            str(tick / self.sampling_rate) for tick in tick_positions
        ]  # Convert frame positions to seconds

        # Set the new ticks and labels on the x-axis
        axis.set_xticks(tick_positions)
        axis.set_xticklabels(tick_labels)

        axis.axvline(0, color="black", linewidth=0.25)
        axis.set_xlabel("Time (s)")
        axis.set_ylabel("Accuracy")
        axis.set_ylim(0, 1)
        title = [f"{self.folder_name}", f"Decoder: {self.decoder_type}"]
        if self.params4decoder is not None:
            title += [f"{key}: {value}" for key, value in self.params4decoder.items()]
        title = self.utils.create_multiline_string(title)
        axis.set_title(title)
        self.fig_tools.add_text_box(
            ax=axis,
            text=pval_text,
            transform=axis.transAxes,
            fontsize=12,
            va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
        axis.legend()

        # self.print_wFrm("Saving figure")
        self.fig_tools.save_figure(
            fig, f"Decoder_Results_{self.decoder_type}", figure_save_path=self.FigPath
        )

    def _plotBar_Accuracy(self) -> None:
        fig, axis = self.fig_tools.create_plt_subplots()

        odor_color = self.color_dict["blue"]
        switch_color = self.color_dict["orange"]

        self.print_wFrm("Bar plots for accuracy")
        for idx, (key, accu) in enumerate(self.accuracy.items()):
            self.fig_tools.bar_plot(
                ax=axis,
                X=idx,
                Y=np.mean(accu),
                yerr=np.std(accu),
                color=odor_color if key == "ODORS" else switch_color,
                ylim=(0, 1),
                label=key,
            )

        for idx, (key, naccu) in enumerate(self.null_accuracy.items()):
            mean_naccu_by_perm = np.mean(naccu, axis=1)
            self.fig_tools.bar_plot(
                ax=axis,
                X=idx + 2,
                Y=np.mean(mean_naccu_by_perm),
                yerr=np.std(mean_naccu_by_perm),
                color=odor_color if key == "ODORS" else switch_color,
                linestyle=":",
                ylim=(0, 1),
                label=f"{key} (null)",
            )

        axis.set_xticks([0, 1, 2, 3])
        axis.set_xticklabels(["ODORS", "SW", "ODORS (null)", "SW (null)"])

        axis.set_ylabel("Accuracy")

        title = [f"{self.folder_name}", f"Decoder: {self.decoder_type}"]
        if self.params4decoder is not None:
            title += [f"{key}: {value}" for key, value in self.params4decoder.items()]
        title = self.utils.create_multiline_string(title)
        axis.set_title(title)

        # axis.legend()

        # self.print_wFrm("Saving figure")
        self.fig_tools.save_figure(
            fig, f"Decoder_Results_{self.decoder_type}", figure_save_path=self.FigPath
        )

    def _plot_similarity_matrixWClustering(self) -> None:
        """
        Plots the similarity matrix with clustering.
        """
        fig, axis = self.fig_tools.create_plt_subplots(ncols=2, nrows=2, flatten=True)

        for idx, (ax, label_cat) in enumerate(zip(axis[:2], self.Labels.keys())):
            # Get labels for the current category
            true_label = self.Labels[label_cat]
            pred_label = self.cluster_labels[label_cat]

            # --- Color Mapping ---
            # Combine true and predicted labels to find all unique possible labels
            all_labels = np.unique(np.concatenate((true_label, pred_label)))
            n_unique_total = len(all_labels)

            # Create ONE discrete colormap based on the total number of unique labels
            cmap = plt.get_cmap(
                "tab10", n_unique_total
            )  # Use a suitable colormap like tab10

            # Create ONE mapping from label value to 0-based index for the colormap
            label_to_index_map = {label: i for i, label in enumerate(all_labels)}

            # Map both true and predicted labels using the SAME map and cmap
            true_label_indices = np.array(
                [label_to_index_map[lab] for lab in true_label]
            )
            pred_label_indices = np.array(
                [label_to_index_map[lab] for lab in pred_label]
            )

            # --- Plotting ---
            # Plot main similarity matrix heatmap
            # Note: Replaces self.fig_tools.plot_imshow to get the image object 'im'
            im = self.fig_tools.plot_imshow(
                fig=fig,
                axis=ax,
                data2plot=self.similarity_matrix,
                cmap="magma",
                return_im=True,
            )
            # Create divider to add new axes (for the bars)
            divider = make_axes_locatable(ax)

            # Append axes to the top (for true labels) and left (for predicted labels)
            ax_true_bar = divider.append_axes("top", size="3%", pad=0.05, sharex=ax)
            ax_pred_bar = divider.append_axes("left", size="3%", pad=0.05, sharey=ax)

            # Plot color bars using imshow on the new axes
            ax_true_bar.imshow(
                true_label_indices[np.newaxis, :], cmap=cmap, aspect="auto"
            )
            ax_pred_bar.imshow(
                pred_label_indices[:, np.newaxis], cmap=cmap, aspect="auto"
            )

            # --- Cleanup ---
            # Remove ticks and labels from the color bars
            for ax_bar in [ax_true_bar, ax_pred_bar]:
                ax_bar.tick_params(axis="x", labelbottom=False, bottom=False)
                ax_bar.tick_params(axis="y", labelleft=False, left=False)

            # Optionally remove ticks from main heatmap as well
            ax.tick_params(axis="x", labelbottom=False, bottom=False)
            ax.tick_params(axis="y", labelleft=False, left=False)

            # Add titles to identify the bars
            ax_true_bar.set_title("True")
            ax_pred_bar.set_ylabel("Pred", rotation=90, size="large")

            # Add a colorbar for the main similarity matrix heatmap
            cax = divider.append_axes("right", size="1%", pad=0.1)
            fig.colorbar(
                im,
                cax=cax,
            )
            # Set title on main axis but push it higher using the 'y' parameter
            ax.set_title(label_cat, y=1.1)  # Adjust the y value (e.g., 1.08) as needed

            ax4weights = axis[idx + 2]

            weights2use = self.projection_weights[label_cat].T
            for weight, label in zip(weights2use, all_labels):
                weight = (weight - np.min(weight)) / (np.max(weight) - np.min(weight))
                ax4weights.plot(
                    weight, color=cmap(label_to_index_map[label]), alpha=0.7
                )

        self.fig_tools.save_figure(
            fig, "Similarity_Matrix", figure_save_path=self.FigPath
        )

    def _plot_umap_projection(self) -> None:
        """
        Performs UMAP dimensionality reduction and plots the 2D embedding,
        colored by true labels.
        """
        self.print_wFrm("Plotting UMAP projection")
        n_cats = len(self.Labels.keys())
        fig, axes = self.fig_tools.create_plt_subplots(
            ncols=n_cats, figsize=(6 * n_cats, 5), flatten=True
        )

        # UMAP works with distances (0 = close, high = far)
        distance_matrix = 1 - self.similarity_matrix
        # Ensure diagonal is 0
        np.fill_diagonal(distance_matrix, 0)
        # Symmetrize (should already be symmetric, but enforce)
        distance_matrix = (distance_matrix + distance_matrix.T) / 2

        # --- Run UMAP ---
        reducer = umap.UMAP(
            metric="precomputed", random_state=42, n_neighbors=15, min_dist=0.1
        )
        embedding = reducer.fit_transform(distance_matrix)

        self.Label_Names = {
            "ODORS": ["ODOR 1", "ODOR 2"],
            "ODORSwSWITCH": ["O1L1", "O2L2", "O1L2", "O2L1"],
        }

        for ax, label_cat in zip(axes, self.Labels.keys()):
            true_label = self.Labels[label_cat]
            unique_labels = np.unique(true_label)
            n_unique = len(unique_labels)
            cmap = plt.get_cmap("tab10", n_unique)
            label_map = {label: i for i, label in enumerate(unique_labels)}
            colors = [cmap(label_map[lab]) for lab in true_label]

            # Scatter plot of the embedding
            scatter = ax.scatter(
                embedding[:, 0], embedding[:, 1], c=colors, s=15, alpha=0.7
            )

            # Create legend
            legend_elements = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label=f"{self.Label_Names[label_cat][i]}",
                    markerfacecolor=cmap(label_map[lab]),
                    markersize=5,
                )
                for i, lab in enumerate(unique_labels)
            ]
            ax.legend(handles=legend_elements)

            ax.set_title(f"UMAP Projection ({label_cat})")
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        fig.suptitle(f"UMAP Projection based on Similarity - {self.folder_name}")
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        self.fig_tools.save_figure(
            fig, "UMAP_Projection", figure_save_path=self.FigPath
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
                group_labels = ["Od 1", "Od 2"]
            elif aggregate_cm.shape[0] == 4:
                group_labels = ["Od 1", "Od2", "SW Od1", "SW Od2"]
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


if __name__ == "__main__":
    run_CLAH_script(TwoOdorDecoder, parser_enum=decoder_enum.Parser4TOD_DEV)

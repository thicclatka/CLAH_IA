import numpy as np
from scipy.stats import ttest_ind
from typing import Literal

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
        useZscore: bool,
        decoder_type: Literal["SVC", "GBM", "LSTM"],
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
            useZscore,
            decoder_type,
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
        self.findTrialEpochs()
        self.DecodeTrialEpochs()
        self.plotResults()
        pass

    def static_class_var_init(
        self,
        path: str,
        sess2process: list,
        num_folds: int,
        null_repeats: int,
        useZscore: bool,
        decoder_type: Literal["SVC", "GBM", "LSTM"],
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

        self.num_folds = num_folds
        self.num_fold4switch = None
        self.null_repeats = null_repeats

        self.decoder_type = decoder_type
        self.params4decoder = None

        self.useZscore = useZscore

        # SVC parameters needed separately for easier access for condition check later
        self.parse_cost_param = None

        if decoder_type is None:
            self._select_decoder_type()
        elif decoder_type == "SVC":
            # if cost param is not set in parser, it will be found via hyperparameter tuning (calc_CostParam)
            # & then decoded
            self.parse_cost_param = float(cost_param)

            if isinstance(gamma, float) and gamma > 0 or gamma in ["auto", "scale"]:
                gamma = gamma
            else:
                self.rprint(
                    "\nGamma parameter must be 'auto', 'scale', or a positive float. Setting to default value: 'scale'."
                )
                gamma = "scale"

            self.params4decoder = {
                "C": self.parse_cost_param,
                "kernel_type": kernel_type,
                "gamma": gamma,
                "weight": weight,
            }
        elif decoder_type == "GBM":
            self.params4decoder = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "learning_rate": learning_rate,
            }

        self.FigPath = "Figures/Decoder"

    def _select_decoder_type(self) -> None:
        def _get_positive_float(
            prompt: str, default: float = None, str_opts: list = []
        ) -> float | str | None:
            invalid_str = "Invalid input. Please enter a positive float."
            if str_opts is not None:
                invalid_str = f"Invalid input. Please enter a positive float or one of the following options: {', '.join(str_opts)}"
            while True:
                user_input = input(prompt).strip()
                if user_input == "":
                    return default
                if isinstance(user_input, str) and str_opts and user_input in str_opts:
                    return user_input
                try:
                    value = float(user_input)
                    if value > 0:
                        return value
                    else:
                        print(invalid_str)
                except ValueError:
                    print(invalid_str)

        decoder_type = (
            input("Select decoder type (options: SVC, GBM, LSTM): ").strip().upper()
        )

        if decoder_type not in ["SVC", "GBM", "LSTM"]:
            self.rprint("Invalid decoder type. Please select from the given options.")
            self._select_decoder_type()

        self.decoder_type = decoder_type

        print(f"Selected decoder type: {decoder_type}")
        self.print_wFrm(
            "Input parameters for the selected decoder type. Press Enter to use default values."
        )
        if decoder_type == "SVC":
            C = _get_positive_float(
                "Enter cost parameter for the SVM (float: > 0) [default: None]: ",
                default=None,
            )

            kernel_type = input(
                "Enter kernel type (options: 'linear', 'rbf', 'poly', 'sigmoid') [default: 'rbf']: "
            ).strip()
            kernel_type = kernel_type if kernel_type else "rbf"

            gamma = _get_positive_float(
                "Enter gamma value (options: 'auto', 'scale', or a positive float) [default: 'scale']: ",
                default="scale",
                str_opts=["auto", "scale"],
            )

            weight = input(
                "Enter class weight (options: 'balanced', None, or a dictionary) [default: 'balanced']: "
            ).strip()
            weight = weight if weight else "balanced"
            if weight == "None":
                weight = None

            self.params4decoder = {
                "C": C,
                "kernel_type": kernel_type,
                "gamma": gamma,
                "weight": weight,
            }

        elif decoder_type == "GBM":
            n_estimators = _get_positive_float(
                "Enter number of estimators (int: > 0) [default: 100]: ",
                default=100,
            )
            max_depth = _get_positive_float(
                "Enter max depth (int: > 0) [default: 3]: ", default=3
            )
            learning_rate = _get_positive_float(
                "Enter learning rate (float: 0 to 1) [default: 0.1]: ",
                default=0.1,
            )

            self.params4decoder = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "learning_rate": learning_rate,
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
        self.TrialEpochs = None
        if self.parse_cost_param is not None and self.decoder_type == "SVC":
            self.cost_param = self.parse_cost_param
        else:
            self.cost_param = None
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

        # extract C & zscore it
        self.C_Temp = self.SD["C"]

        if self.useZscore:
            self.print_wFrm("Z-scoring Temporal Data", frame_num=1)
            self.zC_Temp = []
            for idx in range(self.C_Temp.shape[0]):
                Ca_arr = self.PKS_UTILS.zScoreCa(self.C_Temp[idx, :])
                self.zC_Temp.append(Ca_arr)

            self.zC_Temp = np.array(self.zC_Temp)

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
        lapTypeName = self.lapCue["lap"]["lapTypeName"]
        switch_lap = _find_switch_lap(lapTypeArr, lapTypeName)

        all_odor_times = np.array([]).reshape(0, 3)

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
            # Times, laps, types all should have same shape
            num_trials = cueTimes.shape[0]
            # create odor labels
            odor_labels = odor * np.ones(num_trials)
            # switch labels are odor + switch (1, 2 for odor presents at location 1, 2, 3, 4 for switch laps)
            switch_labels = odor_labels.copy()
            switch_labels[cueTypes == switch_lap] = odor + 2
            odor_times = np.column_stack((cueTimes, odor_labels, switch_labels))
            all_odor_times = np.vstack((all_odor_times, odor_times))

        # sort by time
        indices = np.argsort(all_odor_times[:, 0])
        self.OdorTimes = all_odor_times[indices]

        odor_labels, odor_counts = np.unique(self.OdorTimes[:, 1], return_counts=True)
        switch_labels, switch_counts = np.unique(
            self.OdorTimes[:, 2], return_counts=True
        )

        for lab, count in [(odor_labels, odor_counts), (switch_labels, switch_counts)]:
            label_count_pairs = [f"{int(lb)}: {c}" for lb, c in zip(lab, count)]
            formatted_str = ", ".join(label_count_pairs)
            if lab.size == 2:
                self.print_wFrm(f"Odor Labels & Counts: {formatted_str}")
            elif lab.size == 4:
                self.print_wFrm(f"Odor with Switch Labels & Counts: {formatted_str}")

        self.print_done_small_proc()

    def findTrialEpochs(self) -> None:
        """
        Organizes times and labels by trial epochs.

        This method finds trial epochs based on cue start times and organizes the data accordingly.
        It calculates the trial start and end times based on the pre-cue and post-cue times.
        It then finds the indices of frames that fall within the trial epoch and extracts the corresponding data.
        The extracted data is stored in the TrialEpochs array along with the corresponding odor labels and switch labels.

        Returns:
            None
        """
        self.rprint("Organizing Times & Labels by trial epochs:")
        self.print_wFrm(
            f"{int(self.trial_dur / self.sampling_rate)} second windows: {self.pre_cue_time} sec pre-cue; {self.post_cue_time} sec post-cue"
        )
        adjFrTimes = np.array(self.TBD["adjFrTimes"])

        # Trial epochs needs shape of cells x time x trials
        self.TrialEpochs = np.full(
            (self.C_Temp.shape[0], self.trial_dur, self.OdorTimes.shape[0]),
            np.nan,
        )

        # set which Temporal to use
        CTEMP2USE = self.C_Temp if not self.useZscore else self.zC_Temp

        for idx, ot in enumerate(self.OdorTimes):
            cue_start_time = ot[0]
            odor_label = ot[1]
            switch_label = ot[2]

            trial_start = int(cue_start_time - self.pre_cue_time)
            trial_end = int(cue_start_time + self.post_cue_time)

            # find indices of frames that are within the trial epoch
            trial_indices = np.where(
                (adjFrTimes >= trial_start) & (adjFrTimes <= trial_end)
            )[0]
            if trial_indices.shape[0] == self.trial_dur:
                trial_data = CTEMP2USE[:, trial_indices]
                self.TrialEpochs[:, :, idx] = trial_data
                self.Labels["ODORS"].append(odor_label)
                self.Labels["ODORSwSWITCH"].append(switch_label)

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

    def DecodeTrialEpochs(self) -> None:
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
                The number of folds used for cross-validation.

            Returns:
            - accuracy: float
                The accuracy of the decoder.

            """
            # see params4decoder in self.static_class_var_init for SVC and GBM parameters
            accuracy, conf_matrices = GeneralDecoder.run_Decoder(
                data_arr=self.TrialEpochs,
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
                self.TrialEpochs,
                np.array(self.Labels["ODORS"]),
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
        if self.decoder_type not in ["LSTM"]:
            self.null_accuracy = {
                key: np.full(
                    (
                        self.trial_dur,
                        self.null_repeats,
                        _determine_fold_count(key),
                    ),
                    np.nan,
                )
                for key in self.Labels.keys()
            }
        else:
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
                if self.decoder_type not in ["LSTM"]:
                    self.null_accuracy[key][:, rep, :] = null_result
                else:
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

        if self.decoder_type not in ["LSTM"]:
            # plot decoder accuracy over time w/in epoch
            self._plotSEM_Accuracy()
        else:
            self._plotBar_Accuracy()

        # confusion matrix
        self._plotConfusionMatrix()
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
                color=odor_color if key == "ODORS" else switch_color,
                ylim=(0, 1),
                label=key,
            )

        for idx, (key, naccu) in enumerate(self.null_accuracy.items()):
            self.fig_tools.bar_plot(
                ax=axis,
                X=idx + 2,
                Y=np.mean(naccu),
                color=odor_color if key == "ODORS" else switch_color,
                linestyle=":",
                ylim=(0, 1),
                label=f"{key} (null)",
            )

        axis.set_ylabel("Accuracy")

        title = [f"{self.folder_name}", f"Decoder: {self.decoder_type}"]
        if self.params4decoder is not None:
            title += [f"{key}: {value}" for key, value in self.params4decoder.items()]
        title = self.utils.create_multiline_string(title)
        axis.set_title(title)

        axis.legend()

        # self.print_wFrm("Saving figure")
        self.fig_tools.save_figure(
            fig, f"Decoder_Results_{self.decoder_type}", figure_save_path=self.FigPath
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
    run_CLAH_script(TwoOdorDecoder, parser_enum=decoder_enum.Parser4TOD)

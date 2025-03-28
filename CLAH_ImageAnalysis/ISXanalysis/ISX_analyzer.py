import os
import numpy as np
from tqdm import tqdm
from CLAH_ImageAnalysis.core import BaseClass as BC
from CLAH_ImageAnalysis.core import run_CLAH_script
from CLAH_ImageAnalysis.unitAnalysis import pks_utils
from CLAH_ImageAnalysis.unitAnalysis import UA_enum
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
import pandas as pd


class ISX_analyzer(BC):
    def __init__(
        self,
        path: str | list,
        sess2process: list,
        fps: int,
        window_size: int = 5,
        epochCutoff: int = 2,
        pseudoEpochSize: int = 2,
    ) -> None:
        self.program_name = "ISX_ANLZR"
        self.class_type = "manager"
        BC.__init__(
            self,
            program_name=self.program_name,
            mode=self.class_type,
            sess2process=sess2process,
        )

        self.static_class_var_init(
            folder_path=path,
            sess2process=sess2process,
            fps=fps,
            window_size=window_size,
            epochCutoff=epochCutoff,
            pseudoEpochSize=pseudoEpochSize,
        )

    def static_class_var_init(
        self,
        folder_path: str | list,
        sess2process: list,
        fps: int,
        window_size: int,
        epochCutoff: int,
        pseudoEpochSize: int,
    ) -> None:
        BC.static_class_var_init(
            self,
            folder_path=folder_path,
            file_of_interest=self.text_lib["selector"]["tags"]["SD"],
            selection_made=sess2process,
            noTDML4SD=True,
        )

        self.fig_tools = self.utils.fig_tools
        self.fps = fps
        self.window_size = window_size
        self.epochCutoff = epochCutoff
        self.pseudoEpochSize = pseudoEpochSize
        self.fig_path = self.text_lib["Folders"]["GENFIG"]

        self.pseudoEpochDur = int(2 * self.pseudoEpochSize * self.fps)

        self.fs_title = 30
        self.fs_axis = 25
        self.fs_axis_small = 20
        self.fs_cbar = 15
        self.fs_legend = 20

        self.colors4plt = {
            "A": self.color_dict["red"],
            "B": self.color_dict["darkblue"],
            "FRZ": self.color_dict["blue"],
            "NONFRZ": self.color_dict["green"],
            "SPECIAL_A": self.color_dict["turquoise"],
            "SPECIAL_B": self.color_dict["orange"],
        }

        self.markers = {
            "A": "o",
            "B": "X",
            "SPECIAL": "*",
        }

        self.cmap = "afmhot"
        self.aspect = "auto"

        self.FRZ = "FRZ"
        self.NFRZ = "NONFRZ"

        self.frz_names = [self.FRZ, self.NFRZ]

        self.alpha4frzspan = 0.4

        # bootstrapping params
        self.bootstrap_params = {
            "n_runs": 1000,
            "n_bins": 10,
            "min_significant_bins": 5,
            "p_threshold": 0.05,
        }

        self.cellCategories = [
            ("A_ONLY", "A Only", ""),
            ("B_ONLY", "B Only", ""),
            ("BOTH", "Both", ""),
            ("NA", "NA", ""),
        ]
        self.columns4df = (
            ["ID", "Experiment"]
            + [f"{cc[0]}_prop" for cc in self.cellCategories]
            + [f"{cc[0]}_count" for cc in self.cellCategories]
            + ["Total_cells"]
        )
        self.cellPropsDF = pd.DataFrame(columns=self.columns4df)

    def forLoop_var_init(self, sess_idx: int, sess_num: int) -> None:
        super().forLoop_var_init(sess_idx, sess_num)

        self.frameWindow4pks = UA_enum.Parser4QT.ARG_DICT.value[("fps", "f")]["DEFAULT"]
        self.sdThresh = UA_enum.Parser4QT.ARG_DICT.value[("sdThresh", "sdt")]["DEFAULT"]
        self.timeout = UA_enum.Parser4QT.ARG_DICT.value[("timeout", "to")]["DEFAULT"]

        self.PKSUtils = pks_utils(
            fps=self.frameWindow4pks,
            sdThresh=self.sdThresh,
            timeout=self.timeout,
        )

    def process_order(self) -> None:
        self.import_segDict()
        self.find_eventRate()
        self.determine_context_spikes()

        print("Plotting:")
        self.plot_CTemp()
        self.plot_eventRate()
        self.plot_PCA() if not self.NoBorisCheck else None
        self.plot_cellProps()
        self.print_done_small_proc()

        self.export_segDict_postAnalysis()

    def import_segDict(self) -> None:
        print("Importing segDict")
        sd_pkl_fname = self.findLatest([self.file_tag["SD"], self.file_tag["PKL"]])
        segDict = self.load_file(sd_pkl_fname)
        self.sd_fname_gen = sd_pkl_fname.split(".")[0]
        self.print_wFrm(f"Loaded: {sd_pkl_fname}")
        self.segDict = segDict

        self.CTemp = {"RAW": segDict["C"]}
        self.cell_num = self.CTemp["RAW"].shape[0]
        # self.CTemp = {"RAW": segDict["C"], "ZSCORE": []}
        self.FrameTimes = segDict["CFrameTimes"]

        if "CtxtOrder" in segDict.keys():
            self.CtxtOrder = segDict["CtxtOrder"]
        else:
            self.CtxtOrder = None
        if "FrzDict" in segDict.keys():
            self.FrzDict = segDict["FrzDict"]
            self.FRZ_BY_TIME = self.FrzDict["FRZ_BY_TIME"]
            self.CtxtByCFrameTimes = segDict["CtxtByCFrameTimes"]
        else:
            self.FrzDict = None
            self.FRZ_BY_TIME = None
            self.CtxtByCFrameTimes = None

        self.NoBorisCheck = False
        if self.FrzDict is not None and self.FrzDict["START"] is None:
            self.NoBorisCheck = True

        self.pks = []
        for Ca_arr in self.CTemp["RAW"]:
            pks, _, _ = self.PKSUtils.find_CaTransients(Ca_arr)
            self.pks.append(pks.astype(int))

        self.spikes = np.zeros_like(self.CTemp["RAW"])
        for cell_idx, pks in enumerate(self.pks):
            self.spikes[cell_idx, pks] = 1

        # for cell_idx, spks in enumerate(self.spikes):
        #     print(f"Cell {cell_idx} has {np.sum(spks)} spikes")
        self._print_stats("RAW", self.CTemp["RAW"])
        # self._print_stats("ZSCORE", self.CTemp["ZSCORE"])

    def _print_stats(self, name: str, data: np.ndarray) -> None:
        print(f"\n{name} data:")
        print(f"Min: {data.min():.2f}")
        print(f"Max: {data.max():.2f}")
        print(f"Mean: {np.mean(data):.2f}")
        print(f"Percentiles (1, 50, 99): {np.percentile(data, [1, 50, 99])}")
        print()

    def find_eventRate(self) -> None:
        def _find_meanER_by_cell(data: np.ndarray, epochs: list) -> np.ndarray:
            return np.hstack(
                [
                    np.nanmean(data[:, st:end], axis=1, keepdims=True)
                    for st, end in epochs
                ]
            )

        def _find_midNendpoint4epoch_by_idx(start_times: list) -> tuple:
            total_frames = len(self.FrameTimes)
            midpoint = np.sum(start_times <= int(total_frames / 2))
            endpoint = np.sum(start_times > int(total_frames / 2))
            return midpoint, endpoint

        def _create_ctxt_by_epoch(
            ctxt_order: list, epoch_midpoint: int, epoch_endpoint: int
        ) -> np.ndarray:
            return np.hstack(
                [
                    [ctxt_order[0]] * epoch_midpoint,
                    [ctxt_order[1]] * epoch_endpoint,
                ]
            )

        def _get_startNstop_idx(start_times: list, stop_times: list) -> tuple:
            return np.searchsorted(self.FrameTimes, start_times), np.searchsorted(
                self.FrameTimes, stop_times
            )

        def _fill_in_epochDict(
            epochDict: dict,
            start_times: list,
            stop_times: list,
            epochKeys: list = ["IDX", "TIMES", "BOOL_BY_TIME", "CTXT_BY_EPOCH"],
        ) -> None:
            epochDict = {key: [] for key in epochDict.keys()}
            start_idx, stop_idx = _get_startNstop_idx(start_times, stop_times)
            epoch_midpoint, epoch_endpoint = _find_midNendpoint4epoch_by_idx(start_idx)
            epochDict["CTXT_BY_EPOCH"] = _create_ctxt_by_epoch(
                ctxt_order=self.CtxtOrder,
                epoch_midpoint=epoch_midpoint,
                epoch_endpoint=epoch_endpoint,
            )

            epochDict["TIMES"] = list(zip(start_times, stop_times))
            epochDict["IDX"] = list(zip(start_idx, stop_idx))

            epochDict["BOOL_BY_TIME"] = np.zeros_like(self.FrameTimes)
            for start, stop in zip(start_idx, stop_idx):
                epochDict["BOOL_BY_TIME"][start:stop] = 1

            return epochDict

        print("Finding event rate", end="", flush=True)
        self.eventRate = {key: [] for key in self.CTemp.keys()}
        for key in self.CTemp.keys():
            for Ca_arr in self.CTemp[key]:
                self.eventRate[key].append(
                    self.dep.apply_convolution(Ca_arr, window_size=self.window_size)
                )
        self.eventRate = {
            key: np.array(self.eventRate[key]) for key in self.CTemp.keys()
        }

        self.segDict["eventRate"] = self.eventRate

        newFrzDict = {"START": [], "STOP": []}
        pseudoEpochs = {"START": [], "STOP": []}
        self.pseudoEpochs = {}
        self.FrzEpochs = {frz_name: {} for frz_name in self.frz_names}
        if not self.NoBorisCheck:
            for idx, (start, stop) in enumerate(
                zip(self.FrzDict["START"], self.FrzDict["STOP"])
            ):
                diff = stop - start
                if diff > self.epochCutoff:
                    newFrzDict["START"].append(start)
                    newFrzDict["STOP"].append(stop)
                    if idx > 0:
                        prev_stop = self.FrzDict["STOP"][idx - 1]
                        prevDiff = start - prev_stop
                        if prevDiff > self.epochCutoff:
                            pseudoEpochs["START"].append(start - self.pseudoEpochSize)
                            pseudoEpochs["STOP"].append(start + self.pseudoEpochSize)

            for key in newFrzDict.keys():
                newFrzDict[key] = np.array(newFrzDict[key])

            unfrz_start = np.hstack([0, newFrzDict["STOP"]])
            unfrz_stop = np.hstack(
                [newFrzDict["START"], len(self.FrameTimes) / (self.fps)]
            )

            diff = unfrz_stop - unfrz_start
            unfrz_start = unfrz_start[diff > self.epochCutoff]
            unfrz_stop = unfrz_stop[diff > self.epochCutoff]

            self.pseudoEpochs = _fill_in_epochDict(
                epochDict=self.pseudoEpochs,
                start_times=pseudoEpochs["START"],
                stop_times=pseudoEpochs["STOP"],
            )

            for frz in self.frz_names:
                if frz == self.FRZ:
                    start2use = newFrzDict["START"]
                    stop2use = newFrzDict["STOP"]
                else:
                    start2use = unfrz_start
                    stop2use = unfrz_stop

                self.FrzEpochs[frz] = _fill_in_epochDict(
                    epochDict=self.FrzEpochs[frz],
                    start_times=start2use,
                    stop_times=stop2use,
                )

            for ctxt in self.CtxtOrder:
                ctxt_mask = self.CtxtByCFrameTimes == ctxt
                frz_mask = self.FrzEpochs[self.FRZ]["BOOL_BY_TIME"] == 1
                unfrz_mask = self.FrzEpochs[self.NFRZ]["BOOL_BY_TIME"] == 1
                pe_mask = self.pseudoEpochs["BOOL_BY_TIME"] == 1
                self.eventRate[ctxt] = self.eventRate["RAW"][:, ctxt_mask]
                self.eventRate[f"{ctxt}_{self.FRZ}"] = self.eventRate["RAW"][
                    :, ctxt_mask & frz_mask
                ]
                self.eventRate[f"{ctxt}_{self.NFRZ}"] = self.eventRate["RAW"][
                    :, ctxt_mask & unfrz_mask
                ]
                self.eventRate[f"{ctxt}_PE"] = self.eventRate["RAW"][
                    :, ctxt_mask & pe_mask
                ]

                # find eventRate for pseudo epochs differently
                # dimensions for ctxt_PE_BY_EPOCH: (cell_num, numPEs for ctxt, pseudoEpochDur)
                # dimensions for ctxt_PE_BY_EPOCH_MEAN: (cell_num, pseudoEpochDur)
                adjusted_e_idx = 0
                for e_idx, epochs in enumerate(self.pseudoEpochs["IDX"]):
                    ctxt_check = self.pseudoEpochs["CTXT_BY_EPOCH"][e_idx] == ctxt
                    if ctxt_check:
                        numPEs = np.sum(self.pseudoEpochs["CTXT_BY_EPOCH"] == ctxt)
                        if f"{ctxt}_PE_BY_EPOCH" not in self.eventRate.keys():
                            self.eventRate[f"{ctxt}_PE_BY_EPOCH"] = np.zeros(
                                (
                                    self.cell_num,
                                    numPEs,
                                    self.pseudoEpochDur,
                                )
                            )
                        for cell_idx in range(self.cell_num):
                            self.eventRate[f"{ctxt}_PE_BY_EPOCH"][
                                cell_idx, adjusted_e_idx, :
                            ] = self.eventRate["RAW"][cell_idx, epochs[0] : epochs[1]]
                        adjusted_e_idx += 1
                if f"{ctxt}_PE_BY_EPOCH" in self.eventRate.keys():
                    self.eventRate[f"{ctxt}_PE_BY_EPOCH_MEAN"] = np.nanmean(
                        self.eventRate[f"{ctxt}_PE_BY_EPOCH"], axis=1
                    )
        else:
            self.pseudoEpochs = None
            self.FrzEpochs = None
            for ctxt in self.CtxtOrder:
                ctxt_mask = self.CtxtByCFrameTimes == ctxt
                self.eventRate[ctxt] = self.eventRate["RAW"][:, ctxt_mask]
                self.eventRate[f"{ctxt}_{self.FRZ}"] = None
                self.eventRate[f"{ctxt}_{self.NFRZ}"] = None

        if not self.NoBorisCheck:
            for ftype in [self.FRZ, self.NFRZ]:
                if self.FrzEpochs[ftype]["IDX"]:
                    self.eventRate[f"MEAN_{ftype}_CELL_BY_EPOCH"] = (
                        _find_meanER_by_cell(
                            data=self.eventRate["RAW"],
                            epochs=self.FrzEpochs[ftype]["IDX"],
                        )
                    )
                else:
                    self.eventRate[f"MEAN_{ftype}_CELL_BY_EPOCH"] = None

        self.print_done_small_proc()

    def determine_context_spikes(self) -> None:
        def _print_cell_counts(ctxt_type: str, cell_num: int) -> None:
            self.print_wFrm(
                f"{ctxt_type}: {cell_num} cells ({cell_num / self.cell_num * 100:.2f}%)"
            )
            self.print_wFrm(f"Cells: {self.context_cells[ctxt_type]}", frame_num=1)

        context_separation_point = self.CtxtByCFrameTimes.shape[0] // 2
        n_bootstraps = self.bootstrap_params["n_runs"]
        p_threshold = self.bootstrap_params["p_threshold"]
        n_bins = self.bootstrap_params["n_bins"]
        min_significant_bins = self.bootstrap_params["min_significant_bins"]

        bin_size = context_separation_point // n_bins
        self.FR_by_ctxt_binned = {ctxt: [] for ctxt in self.CtxtOrder}

        print("Finding significant cells:")
        self.print_wFrm(f"Bins: {n_bins}")
        self.print_wFrm(f"Frames per bin: {bin_size}")
        self.print_wFrm(
            f"Min significant bins: {min_significant_bins} ({(min_significant_bins / n_bins) * 100:.1f}%)"
        )

        # Calculate actual firing rates for each bin
        for ctxt in self.CtxtOrder:
            cell_rates = []
            for cell_idx in range(self.cell_num):
                bin_rates = []
                curr_spks = self.eventRate[ctxt][cell_idx]
                for bin_idx in range(n_bins):
                    start_idx = bin_idx * bin_size
                    end_idx = (bin_idx + 1) * bin_size
                    bin_fr = curr_spks[start_idx:end_idx]
                    bin_rate = np.mean(bin_fr)
                    bin_rates.append(bin_rate)
                cell_rates.append(bin_rates)
            self.FR_by_ctxt_binned[ctxt] = np.array(cell_rates)

        self.context_cells = {ctxt: [] for ctxt in self.CtxtOrder}
        self.context_cells["pVal_by_bin"] = {
            ctxt: np.zeros((len(self.pks), n_bins)) for ctxt in self.CtxtOrder
        }
        self.context_cells["pVal_mean"] = {ctxt: [] for ctxt in self.CtxtOrder}
        self.context_cells["significant_bins"] = {ctxt: [] for ctxt in self.CtxtOrder}

        for cell_idx in tqdm(range(self.cell_num), desc="Running bootstrap"):
            cell_data = {
                ctxt: self.eventRate[ctxt][cell_idx] for ctxt in self.CtxtOrder
            }

            # Store p-values for each bin for each context
            bin_p_values = {ctxt: np.zeros(n_bins) for ctxt in self.CtxtOrder}

            # Calculate shuffled distributions for this bin
            shuffled_rates = {
                ctxt: np.zeros((n_bootstraps, n_bins)) for ctxt in self.CtxtOrder
            }

            for bootstrap_idx in range(n_bootstraps):
                for ctxt_idx, ctxt in enumerate(self.CtxtOrder):
                    shuffled_spikes = np.random.permutation(cell_data[ctxt])
                    for b in range(n_bins):
                        bin_start = b * bin_size
                        bin_end = bin_start + bin_size
                        shuffled_rates[ctxt][bootstrap_idx, b] = np.nanmean(
                            shuffled_spikes[bin_start:bin_end]
                        )

                # Compare actual rates to shuffled distributions for each context
            for ctxt in self.CtxtOrder:
                for bin_idx in range(n_bins):
                    actual_rate = self.FR_by_ctxt_binned[ctxt][cell_idx, bin_idx]
                    shuffled_bin_rates = shuffled_rates[ctxt][:, bin_idx]

                    # Calculate p-value for each bin
                    # tests shuffled < actual_rate
                    bin_p_values[ctxt][bin_idx] = stats.ttest_1samp(
                        shuffled_bin_rates, actual_rate, alternative="less"
                    )[1]

                    self.context_cells["pVal_by_bin"][ctxt][cell_idx, bin_idx] = (
                        bin_p_values[ctxt][bin_idx]
                    )

            # Determine context specificity based on number of significant bins
            for ctxt in self.CtxtOrder:
                significant_bins = np.nansum(bin_p_values[ctxt] < p_threshold)
                self.context_cells["significant_bins"][ctxt].append(significant_bins)
                self.context_cells["pVal_mean"][ctxt].append(
                    np.nanmean(bin_p_values[ctxt])
                )

                if significant_bins >= min_significant_bins:
                    self.context_cells[ctxt].append(cell_idx)

        self.context_cells["BOTH"] = list(
            set(self.context_cells["A"]) & set(self.context_cells["B"])
        )

        for ctxt in self.CtxtOrder:
            ctxt_cells = set(self.context_cells[ctxt])
            ctxt_only_cells = ctxt_cells - set(self.context_cells["BOTH"])
            self.context_cells[f"{ctxt}_ONLY"] = list(ctxt_only_cells)

        cell_num_range = set(range(self.cell_num))
        self.context_cells["NA"] = list(
            cell_num_range
            - set(self.context_cells["BOTH"])
            - set(self.context_cells["A"])
            - set(self.context_cells["B"])
        )

        print("Results:")
        self.print_wFrm(f"Total cells: {self.cell_num}")
        for ctxt in self.CtxtOrder:
            _print_cell_counts(f"{ctxt}_ONLY", len(self.context_cells[f"{ctxt}_ONLY"]))
        _print_cell_counts("BOTH", len(self.context_cells["BOTH"]))
        _print_cell_counts("NA", len(self.context_cells["NA"]))
        self.print_done_small_proc()

    def plot_CTemp(self) -> None:
        self.print_wFrm("Heatmap of Ca signal across all cells")
        fig, axes = self.fig_tools.create_plt_subplots(
            nrows=1, ncols=1, figsize=(20, 20), flatten=True
        )
        data = [
            self.CTemp["RAW"],
            # self.CTemp["ZSCORE"],
            # self.eventRate["RAW"],
            # self.eventRate["ZSCORE"],
        ]
        titles = [
            f"Raw Ca Signal - {self.dayDir[self.sess_idx]}",
            # "Z-scored Ca signals",
            # "Event rate",
            # "Z-scored event rate",
        ]
        labels = ["Time (frames)", "Cell"]

        data2use = (data[0] - data[0].min()) / (data[0].max() - data[0].min())
        im = self.fig_tools.plot_imshow(
            fig=fig,
            axis=axes,
            data2plot=data2use,
            cmap=self.cmap,
            # vmin=np.percentile(data2use, 0.5),
            # vmax=np.percentile(data2use, 99.5),
            aspect=self.aspect,
            return_im=True,
        )

        if self.FrzDict["START"] is not None or not self.NoBorisCheck:
            frz_periods = self.FrzEpochs[self.FRZ]["IDX"]

            for start, end in frz_periods:
                axes.axvspan(
                    start, end, color=self.color_dict["blue"], alpha=self.alpha4frzspan
                )

            for start, end in frz_periods:
                axes.annotate(
                    "",
                    xy=(start, 1.01),
                    xytext=(end, 1.01),
                    xycoords=("data", "axes fraction"),  # This puts it above the plot
                    arrowprops=dict(
                        arrowstyle="-",
                        color=self.colors4plt["FRZ"],
                        linewidth=7,
                        shrinkA=0,
                        shrinkB=0,
                    ),
                )
        if self.CtxtOrder is not None:
            total_time = self.FrameTimes.shape[0]
            context_periods = [
                (0, (total_time / 2) - 1),
                (total_time / 2, total_time - 1),
            ]
            for i, (start, end) in enumerate(context_periods):
                color = self.colors4plt[self.CtxtOrder[i]]
                axes.annotate(
                    "",
                    xy=(start, 1.03),
                    xytext=(end, 1.03),
                    xycoords=("data", "axes fraction"),
                    arrowprops=dict(
                        arrowstyle="-",
                        color=color,
                        linewidth=7,
                        shrinkA=0,
                        shrinkB=0,
                    ),
                )
                # Letter annotation
                mid_point = (start + end) / 2
                axes.annotate(
                    self.CtxtOrder[i],
                    xy=(mid_point, 1.04),  # Position slightly above the line
                    xycoords=("data", "axes fraction"),
                    ha="center",
                    va="bottom",
                    color=color,
                    fontsize=self.fs_axis,
                    fontweight="bold",
                )

        cbar = fig.colorbar(im, ax=axes)
        cbar.ax.tick_params(labelsize=self.fs_cbar)

        fig.suptitle(titles[0], fontsize=self.fs_title)
        axes.set_xlabel(labels[0], fontsize=self.fs_axis)
        axes.set_ylabel(labels[1], fontsize=self.fs_axis)
        axes.tick_params(axis="both", labelsize=self.fs_axis)

        # axes[1].set_xlabel(labels[0], fontsize=self.fs_axis)
        # axes[1].set_ylabel(labels[1], fontsize=self.fs_axis)
        # axes[1].tick_params(axis="both", labelsize=self.fs_axis)

        self.fig_tools.save_figure(
            plt_figure=fig,
            fig_name="CTemp_HeatPlot",
            figure_save_path=self.fig_path,
            tight_layout=False,
        )

    def plot_eventRate(self) -> None:
        self.print_wFrm("Event rate by context")

        def _get_meanNsem(data: np.ndarray) -> tuple:
            cell_means = np.mean(data, axis=1)
            grand_mean = np.mean(cell_means)
            grand_sem = np.std(cell_means) / np.sqrt(cell_means.shape[0])
            return cell_means, grand_mean, grand_sem

        bar_spacing = 0.3
        alpha = 0.5
        frz_hatch = "+"
        unfrz_hatch = "/"

        if not self.NoBorisCheck:
            fig, axes = self.fig_tools.create_plt_subplots(figsize=(20, 20))

            for idx, ctxt in enumerate(np.sort(self.CtxtOrder)):
                special_cells = self.context_cells[ctxt]
                cell_mask = np.zeros(self.eventRate[ctxt].shape[0], dtype=bool)
                cell_mask[special_cells] = True
                cmeans, mean, sem = _get_meanNsem(self.eventRate[ctxt])
                cmeans_frz, mean_frz, sem_frz = _get_meanNsem(
                    self.eventRate[f"{ctxt}_{self.FRZ}"]
                )
                cmeans_unfrz, mean_unfrz, sem_unfrz = _get_meanNsem(
                    self.eventRate[f"{ctxt}_{self.NFRZ}"]
                )
                cmeans_all = [cmeans, cmeans_frz, cmeans_unfrz]
                means = [mean, mean_frz, mean_unfrz]
                sems = [sem, sem_frz, sem_unfrz]
                labels = [ctxt, f"{ctxt}_{self.FRZ}", f"{ctxt}_{self.NFRZ}"]
                x_vals = [idx - bar_spacing, idx, idx + bar_spacing]
                hatch_vals = [None, frz_hatch, unfrz_hatch]
                for cmean, mean, sem, label, x_val, hatch in zip(
                    cmeans_all, means, sems, labels, x_vals, hatch_vals
                ):
                    self.fig_tools.bar_plot(
                        ax=axes,
                        X=x_val,
                        Y=mean,
                        yerr=sem,
                        color=self.colors4plt[ctxt],
                        label=label,
                        hatch=hatch,
                        width=bar_spacing,
                        alpha=alpha,
                    )
                    self.fig_tools.scatter_plot(
                        ax=axes,
                        X=x_val,
                        Y=cmean[~cell_mask],
                        color=self.colors4plt[ctxt],
                        label=label,
                    )
                    self.fig_tools.scatter_plot(
                        ax=axes,
                        X=x_val,
                        Y=cmean[special_cells],
                        color=self.colors4plt[f"SPECIAL_{ctxt}"],
                        marker=self.markers["SPECIAL"],
                    )

            fig.suptitle(
                f"Event rate by context - {self.dayDir[self.sess_idx]}",
                fontsize=self.fs_title,
            )
            # axes.set_xlabel("Context", fontsize=self.fs_axis)
            axes.set_ylabel(r"Event rate (frames$^{-1}$)", fontsize=self.fs_axis)
            axes.tick_params(axis="both", labelsize=self.fs_axis)
            axes.set_xticks([0, 1])
            # axes.set_xticklabels(np.sort(self.CtxtOrder))
            axes.set_xticklabels([])

            legend = self.fig_tools.create_legend_patch_fLoop(
                facecolor=[self.colors4plt[ctxt] for ctxt in np.sort(self.CtxtOrder)]
                + ["none"] * 2,
                label=[ctxt for ctxt in np.sort(self.CtxtOrder)]
                + [self.FRZ, self.NFRZ],
                edgecolor=[self.colors4plt[ctxt] for ctxt in np.sort(self.CtxtOrder)]
                + [self.color_dict["black"]] * 2,
                hatch=[None] * len(np.sort(self.CtxtOrder)) + [frz_hatch, unfrz_hatch],
                alpha=[alpha] * len(np.sort(self.CtxtOrder)) + [None] * 2,
            )
            axes.legend(handles=legend, fontsize=self.fs_legend, loc="upper left")

            self.fig_tools.save_figure(
                plt_figure=fig,
                fig_name="EventRate_byContext",
                figure_save_path=self.fig_path,
                tight_layout=False,
            )

        self.print_wFrm("Event rate for significant cells by context")
        fig0, axes0 = self.fig_tools.create_plt_subplots(
            nrows=2, ncols=1, figsize=(20, 20)
        )
        fig1, axes1 = self.fig_tools.create_plt_subplots(
            nrows=2, ncols=1, figsize=(20, 20), flatten=True
        )

        max_ER = np.nanmax(self.eventRate["RAW"])
        min_ER = np.nanmin(self.eventRate["RAW"])
        cell_nums = []
        for ctxt_idx, ctxt in enumerate(self.CtxtOrder):
            cells = self.context_cells[ctxt]
            ERs = np.transpose(self.eventRate[ctxt][cells])
            ERs = (ERs - min_ER) / (max_ER - min_ER)
            cell_nums.append(ERs.shape[1])
            self.fig_tools.plot_SEM(
                arr=ERs,
                color=self.colors4plt[ctxt],
                ax=axes0[ctxt_idx],
                x_ind=[0, ERs.shape[0]],
            )
            for cell_idx in range(ERs.shape[1]):
                self.fig_tools.line_plot(
                    ax=axes1[ctxt_idx],
                    X=np.arange(ERs.shape[0]),
                    Y=ERs[:, cell_idx],
                    linewidth=1.5,
                    # color=self.colors4plt[ctxt],
                )

        for figs in [fig0, fig1]:
            figs.suptitle(
                f"Event rate for special cells by context - {self.dayDir[self.sess_idx]}",
                fontsize=self.fs_title,
            )
            figs.supylabel(
                r"Normalized Event rate (frames$^{-1}$)", fontsize=self.fs_axis
            )
        for axes in [axes0, axes1]:
            for ctxt_idx, ctxt in enumerate(self.CtxtOrder):
                if ctxt_idx == 0:
                    start, end = 0, len(self.CtxtByCFrameTimes) // 2
                elif ctxt_idx == 1:
                    axes[ctxt_idx].set_xlabel("Time (frames)", fontsize=self.fs_axis)
                    start, end = (
                        len(self.CtxtByCFrameTimes) // 2,
                        len(self.CtxtByCFrameTimes),
                    )
                axes[ctxt_idx].set_title(
                    f"Context: {ctxt}", fontsize=self.fs_axis_small
                )
                axes[ctxt_idx].text(
                    0.05,
                    0.95,
                    f"N$_{{{ctxt}}}$ = {cell_nums[ctxt_idx]}",
                    transform=axes[ctxt_idx].transAxes,
                    fontsize=self.fs_axis_small,
                )
                axes[ctxt_idx].tick_params(axis="both", labelsize=self.fs_axis_small)
                if not self.NoBorisCheck:
                    for st, en in self.FrzEpochs[self.FRZ]["IDX"]:
                        if st > start and en <= end:
                            if ctxt_idx == 1:
                                st = st - len(self.CtxtByCFrameTimes) // 2
                                en = en - len(self.CtxtByCFrameTimes) // 2
                            axes[ctxt_idx].axvspan(
                                st,
                                en,
                                color=self.color_dict["blue"],
                                alpha=self.alpha4frzspan,
                            )
        fnames2save = ["MeanSEM", "IndividualLines"]
        for fig, fname in zip([fig0, fig1], fnames2save):
            self.fig_tools.save_figure(
                plt_figure=fig,
                fig_name=f"EventRate_byContext_CellSpecific_{fname}",
                figure_save_path=self.fig_path,
                tight_layout=False,
            )

        if not self.NoBorisCheck:
            fig_ERPE, axes_ERPE = self.fig_tools.create_plt_subplots(
                nrows=2, ncols=2, figsize=(20, 20), flatten=True
            )

            for c_idx, ctxt in enumerate(self.CtxtOrder):
                if f"{ctxt}_PE_BY_EPOCH" not in self.eventRate.keys():
                    continue

                cells = self.context_cells[ctxt]
                sem_ax = axes_ERPE[c_idx]
                ind_ax = axes_ERPE[c_idx + 2]
                er_by_pe = np.transpose(
                    self.eventRate[f"{ctxt}_PE_BY_EPOCH_MEAN"][cells, :]
                )
                time_array = np.linspace(
                    -self.pseudoEpochSize, self.pseudoEpochSize, er_by_pe.shape[0]
                )
                self.fig_tools.plot_SEM(
                    arr=er_by_pe,
                    color=self.colors4plt[ctxt],
                    ax=sem_ax,
                    x_ind=[time_array[0], time_array[-1]],
                )
                for pe_idx in range(er_by_pe.shape[1]):
                    self.fig_tools.line_plot(
                        ax=ind_ax,
                        X=time_array,
                        Y=er_by_pe[:, pe_idx],
                        linewidth=1.5,
                        color=self.colors4plt[ctxt],
                    )
                sem_ax.set_title(f"Context: {ctxt}", fontsize=self.fs_axis)
                for ax in [sem_ax, ind_ax]:
                    if c_idx == 0:
                        ax.set_ylabel(
                            r"Event rate (frames$^{-1}$)", fontsize=self.fs_axis
                        )
                    # ax.set_ylim([0, 50])
                    # if ax == sem_ax:
                    #     ax.set_ylim([5, 12])
                    if ax == ind_ax:
                        ax.set_ylim([-10, 50])
                        ax.set_xlabel("Onset from freezing (s)", fontsize=self.fs_axis)
                    ax.set_xlim([time_array[0], time_array[-1]])
                    ax.tick_params(axis="both", labelsize=self.fs_axis_small)

            fig_ERPE.suptitle(
                f"Event rate for special cells per pseudo-epoch by context - {self.dayDir[self.sess_idx]}",
                fontsize=self.fs_title,
            )
            self.fig_tools.save_figure(
                plt_figure=fig_ERPE,
                fig_name="EventRate_PE_byContext",
                figure_save_path=self.fig_path,
                tight_layout=True,
            )

    def plot_PCA(self) -> None:
        self.print_wFrm("PCA of event rate by context")
        size = 150
        fig, axes = self.fig_tools.create_plt_subplots(figsize=(20, 20))
        pca = PCA(n_components=2)

        # baseline/nonfrz period
        pca_nonfreezing = pca.fit_transform(
            self.eventRate[f"MEAN_{self.NFRZ}_CELL_BY_EPOCH"].T
        )
        # frz period projected onto baseline
        pca_freezing = pca.transform(self.eventRate[f"MEAN_{self.FRZ}_CELL_BY_EPOCH"].T)

        similarity_freezing = cosine_similarity(
            self.eventRate[f"MEAN_{self.FRZ}_CELL_BY_EPOCH"].T,
            self.eventRate[f"MEAN_{self.NFRZ}_CELL_BY_EPOCH"].T,
        )

        plot_order = []
        for ctxt in self.CtxtOrder:
            for frz_type in self.frz_names:
                plot_order.append((frz_type, ctxt))

        for frz_type, ctxt in plot_order:
            pca2use = pca_freezing if frz_type == self.FRZ else pca_nonfreezing
            self.fig_tools.scatter_plot(
                ax=axes,
                X=pca2use[self.FrzEpochs[frz_type]["CTXT_BY_EPOCH"] == ctxt, 0],
                Y=pca2use[self.FrzEpochs[frz_type]["CTXT_BY_EPOCH"] == ctxt, 1],
                color=self.colors4plt[frz_type],
                label=frz_type,
                s=size,
                marker=self.markers[ctxt],
            )

        legend = self.fig_tools.create_legend_patch_fLoop(
            facecolor=[self.colors4plt[frz_type] for frz_type in self.frz_names]
            + ["none"] * len(self.CtxtOrder),
            label=self.frz_names + list(self.CtxtOrder),
            marker=[None] * len(self.frz_names)
            + [self.markers[ctxt] for ctxt in self.CtxtOrder],
        )

        title = self.utils.create_multiline_string(
            [
                f"PCA of population activity - {self.dayDir[self.sess_idx]}",
                f"Similarity: {similarity_freezing.mean():.3f}",
            ]
        )

        fig.suptitle(title, fontsize=self.fs_title)
        axes.set_xlabel("PCA 1", fontsize=self.fs_axis)
        axes.set_ylabel("PCA 2", fontsize=self.fs_axis)
        axes.tick_params(axis="both", labelsize=self.fs_axis)
        axes.legend(handles=legend, fontsize=self.fs_legend)

        self.fig_tools.save_figure(
            plt_figure=fig,
            fig_name="EventRate_PCA",
            figure_save_path=self.fig_path,
            tight_layout=False,
        )

    def plot_cellProps(self) -> None:
        def autopct_format(pct: float) -> str:
            """
            Format the percentage for the pie chart.

            Parameters:
                pct (float): The percentage.

            Returns:
                str: The formatted percentage.
            """
            return f"{pct:.1f}%" if pct > 2 else ""

        self.print_wFrm("Cell type proportions")
        fig, axes = self.fig_tools.create_plt_subplots(figsize=(20, 20))

        proportions = []
        labels = []
        for key, label, marker in self.cellCategories:
            prop = len(self.context_cells[key]) / self.cell_num
            proportions.append(prop)
            marker_text = f" ({marker})" if marker else ""
            labels.append(f"{label}{marker_text} - {prop * 100:.1f}%")

        data_dict4df = {
            "ID": [self.ID],
            "Experiment": [self.etype.split("-")[0]],
            "Total_cells": [self.cell_num],
        }
        for (key, _, _), prop in zip(self.cellCategories, proportions):
            data_dict4df[f"{key}_prop"] = [prop]
            data_dict4df[f"{key}_count"] = [len(self.context_cells[key])]

        curr_df = pd.DataFrame(data_dict4df)

        self.cellPropsDF = pd.concat([self.cellPropsDF, curr_df], ignore_index=True)

        wedges, _, autotexts = axes.pie(
            proportions,
            labels=None,
            autopct=autopct_format,
            startangle=90,
            textprops=dict(fontsize=self.fs_axis),
            pctdistance=0.90,
        )

        axes.legend(
            wedges,
            labels,
            title=f"Cell Types (Total ROIs: {self.cell_num})",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1),
            fontsize=self.fs_axis,
            title_fontsize=self.fs_title,
        )

        axes.axis("equal")
        axes.set_title("Cell Type Proportions", fontsize=self.fs_title)

        self.fig_tools.save_figure(
            plt_figure=fig,
            fig_name="CellProps",
            figure_save_path=self.fig_path,
        )

    def export_segDict_postAnalysis(self) -> None:
        self.savedict2file(
            dict_to_save=self.segDict,
            dict_name="segDict",
            filename=self.sd_fname_gen,
            filetype_to_save=[self.file_tag["PKL"], self.file_tag["H5"]],
        )
        self.savedict2file(
            dict_to_save=self.context_cells,
            dict_name="context_cells",
            filename="BootStrap_Results",
            filetype_to_save=[self.file_tag["PKL"]],
        )
        if self.pseudoEpochs is not None:
            self.savedict2file(
                dict_to_save=self.pseudoEpochs,
                dict_name="pseudoEpochs",
                filename="pseudoEpochs",
                filetype_to_save=[self.file_tag["PKL"]],
            )
        if self.FrzEpochs is not None:
            self.savedict2file(
                dict_to_save=self.FrzEpochs,
                dict_name="FrzEpochs",
                filename="FrzEpochs",
                filetype_to_save=[self.file_tag["PKL"]],
            )
        self.savedict2file(
            dict_to_save=self.eventRate,
            dict_name="eventRate",
            filename="eventRate",
            filetype_to_save=[self.file_tag["PKL"]],
        )

    def post_proc_run(self) -> None:
        os.chdir(self.dayPath)
        self.cellPropsDF = self.cellPropsDF.sort_values(by=["ID", "Experiment"])
        self.cellPropsDF.to_csv(f"CellCounts{self.file_tag['CSV']}", index=False)


if "__main__" == __name__:
    from CLAH_ImageAnalysis.ISXanalysis import ISX_enum

    run_CLAH_script(ISX_analyzer, parser_enum=ISX_enum.Parser4ISX_ANLZR)

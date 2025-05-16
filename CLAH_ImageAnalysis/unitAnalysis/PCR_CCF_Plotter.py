import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import sem

from CLAH_ImageAnalysis.core import BaseClass as BC
from CLAH_ImageAnalysis.unitAnalysis import UA_enum


class PCR_CCF_Plotter(BC):
    def __init__(
        self,
        numSess: int,
        cue_list: list,
        groups: list,
        sessFocus: int = None,
        numSessByID: list = None,
        export_legend: bool = True,
        fig_fname: str = None,
        forPres: bool = False,
    ) -> None:
        """
        Initialize the PCR_CCF_Plotter class.

        Parameters:
            numSess (int): The number of sessions.
            cue_list (list): The list of cue types.
            groups (list): The list of groups.
            sessFocus (int, optional): The session to focus on. Defaults to None.
            numSessByID (list, optional): The number of sessions per ID. Defaults to None.
            export_legend (bool, optional): Whether to export the legend. Defaults to True.
            fig_fname (str, optional): The filename of the figure. Defaults to None.
            forPres (bool, optional): Whether to export svgs of figures. Defaults to False.
        """
        self.program_name = "PCR_CFF"
        self.class_type = "utils"
        BC.__init__(self, self.program_name, mode=self.class_type)

        # self.refLapType = refLapType_list
        # self.lapTypeNameArr = lapTypeNameArr
        self.numSess = numSess
        self.numSess2use = self.numSessChecker(sessFocus)
        self.numSessByID = numSessByID
        self.numSess2use4means = self.numSessByIDChecker(sessFocus)

        self.xlabels4means = None

        self.sess_idx4groups = np.arange(1, self.numSess2use4means + 1)

        self.groups = groups

        if fig_fname is None:
            self.fig_save_path = self.text_lib["FIGSAVE"]["DEFAULT"]
        else:
            self.fig_save_path = fig_fname

        self.figsize = (15, 15)
        self.fontsize_ylabel = 18
        self.fontsize = 18
        self.fs_tk_lbl = 16
        self.title_fs = 18
        self.axis_fs = 18
        self.medium_fs = 12
        self.small_fs = 10
        self.tiny_fs = 8

        self.fontweight = "bold"
        self.colors = self.utils.color_dict_4cues()
        self.cueType_abbrev = self.text_lib["cueType_abbrev"]
        self.CCFPtxt = self.enum2dict(UA_enum.CCF_PLOT)
        self.desired_order = self.CCFPtxt["ORDER"]

        self.CCFkey = self.enum2dict(UA_enum.CCF)
        self.ind = self.CCFkey["IND"]
        self.baseline_slice = slice((-self.ind[0] - 5), -self.ind[0] + 1)
        self.cueTypes_set = set(cue_list)

        self.session_colors = [
            self.color_dict["red"],
            self.color_dict["blue"],
            self.color_dict["green"],
            self.color_dict["orange"],
            self.color_dict["violet"],
            self.color_dict["cyan"],
            self.color_dict["gray"],
        ]

        self.other_colors = [
            self.color_dict["orange"],
            self.color_dict["violet"],
            self.color_dict["cyan"],
            self.color_dict["gray"],
        ]

        self.forPres = forPres

        self.mv_thresh = 50

        self.cellTypes = ["Tuned", "Cue"]
        self.cTypes_All = ["ALL", "TC", "CC"]
        self.cueHatch = "//"
        self.expHatch = "xx"

        self.group_spacing = 0.2
        self.bwidth = 0.4
        self.mwidth = 0.3
        self.swidth = 0.2
        self.Sswidth = 0.15

        if "AD" in self.groups:
            self.colors4barGroup = {
                "AD": self.color_dict["red"],
                "CTL": self.color_dict["blue"],
            }
            self.groups4sig = ["CTL", "AD"]
        elif "AGED" in self.groups and "NONAGED" not in self.groups:
            self.colors4barGroup = {
                "WT": self.color_dict["green"],
                "AGED": self.color_dict["red"],
                "DK": self.color_dict["blue"],
            }
            self.groups4sig = ["DK", "AGED"]
        elif "NONAGED" in self.groups:
            self.colors4barGroup = {
                "NONAGED": self.color_dict["blue"],
                "AGED": self.color_dict["red"],
            }
            self.groups4sig = ["AGED", "NONAGED"]
        elif "OPTO" in self.groups:
            self.colors4barGroup = {
                "OPTO": self.color_dict["blue"],
            }
            self.groups4sig = ["OPTO"]
        elif "eOPN3" in self.groups:
            self.colors4barGroup = {
                "eOPN3": self.color_dict["green"],
            }
            self.groups4sig = ["eOPN3"]
            self.xlabels4means = [
                f"{prefix}S{sess + 1}"
                for sess in range(int(self.numSess2use4means / 2))
                for prefix in ["pre", "post"]
            ]
        elif "KETA" in self.groups:
            self.colors4barGroup = {
                "KETA": self.color_dict["red"],
            }
            self.groups4sig = ["KETA"]
            if self.numSess2use4means > 2:
                self.xlabels4means = [
                    "preK1",
                    "sal1",
                    "preK2a",
                    "preK2b",
                    "posK1",
                    "posK7",
                    "posK14",
                    "posK20",
                ]
            elif self.numSess2use4means == 2:
                self.xlabels4means = ["preK1", "preK2"]
        elif "MINISCOPE" in self.groups:
            self.colors4barGroup = {
                "MINISCOPE": self.color_dict["blue"],
            }
            self.groups4sig = ["MINISCOPE"]

        self.cmap = "hot"

        # export legend
        if export_legend:
            self._create_sep_legend()

    def plot_cueTrigSig_1PDF(
        self, TrigSig: dict, isCueCell: dict, numSess: int, clust_per_page: int = 4
    ) -> None:
        """
        Plots cue-triggered signals for each cluster and saves the plots as a PDF.

        Parameters:
            TrigSig (dict): A dictionary containing cue-triggered signals for each cluster.
            isCueCell (dict): A dictionary indicating whether each cluster is a cue cell.
            numSess (int): The number of sessions.
            clust_per_page (int, optional): The number of clusters to process per page. Defaults to 4.
        """
        color_map = self.colors
        clusters = sorted(TrigSig.keys(), key=lambda x: int(x[clust_per_page:]))

        with PdfPages(f"{self.fig_save_path}/TrigSig_byCluster.pdf") as pdf:
            for i in range(
                0, len(clusters), clust_per_page
            ):  # Process 4 clusters at a time
                fig, axes = self.fig_tools.create_plt_subplots(
                    ncols=numSess, nrows=clust_per_page, figsize=self.figsize
                )
                for j in range(clust_per_page):
                    if i + j < len(clusters):
                        cluster = clusters[i + j]
                        trigSig4cluster = TrigSig[cluster]
                        for cueType, trig_sig_arrs in trigSig4cluster.items():
                            for idx, tsarr in enumerate(trig_sig_arrs):
                                if (
                                    isinstance(tsarr, np.ndarray)
                                    and not np.isnan(tsarr).all()
                                ):
                                    curr_ax = axes[j, idx]
                                    self.fig_tools.plot_SEM(
                                        arr=tsarr,
                                        color=color_map[cueType],
                                        ax=curr_ax,
                                        x_ind=self.ind,
                                        vline=True,
                                        # baseline=self.baseline_slice,
                                    )

                        for idx, isCC in enumerate(isCueCell[cluster]):
                            plot_title = ""
                            if idx == 0:
                                self._create_bold_yLabel(
                                    ax=axes[j, idx], ylabel=cluster
                                )
                            if j == 0:
                                plot_title += self._printf_Session(idx)
                            if isCC:
                                plot_title += "\nCUE CELL"
                            axes[j, idx].set_title(plot_title, fontsize=self.title_fs)

                # fig.suptitle(f"{cluster}")
                pdf.savefig(fig)
                self.fig_tools.close_all_figs()

    def plot_meanTrigSig(self, mean_byCC_byCueType: dict) -> None:
        """
        Plots the mean triggered signal for each cue type and cortical column.

        Parameters:
            mean_byCC_byCueType (dict): A dictionary containing the mean triggered signal
                for each cue type and cortical column. The dictionary should have the
                following structure:
                {
                    CUE: {
                        cue_type_1: [mean_arr_1, mean_arr_2, ...],
                        cue_type_2: [mean_arr_1, mean_arr_2, ...],
                        ...
                    },
                    NONCUE: {
                        cue_type_1: [mean_arr_1, mean_arr_2, ...],
                        cue_type_2: [mean_arr_1, mean_arr_2, ...],
                        ...
                    },
                }
        """

        color_map = self.colors

        fig, axes = self.fig_tools.create_plt_subplots(
            ncols=self.numSess, nrows=2, figsize=self.figsize
        )
        axes = axes.flatten()

        plt_idx = 0
        for cc_idx, (CC, cueTypes) in enumerate(mean_byCC_byCueType.items()):
            print(CC)
            cc_idx_adj = cc_idx * self.numSess
            for cueType, mean_arrs in cueTypes.items():
                self.print_wFrm(cueType)
                plt_idx = 0 + cc_idx_adj
                for idx, marr in enumerate(mean_arrs):
                    if cueType == "OMITCUE1_SWITCH":
                        continue
                    else:
                        self._plot_SEM_wCellCount(
                            axis2plot=axes[plt_idx],
                            mean_arr=marr,
                            cmap2use=color_map[cueType],
                        )
                    if cc_idx == 0:
                        axes[plt_idx].set_title(
                            self._printf_Session(idx), fontsize=self.title_fs
                        )
                    if idx == 0:
                        self._create_bold_yLabel(ax=axes[plt_idx], ylabel=CC)
                    plt_idx += 1

        self.fig_tools.save_figure(
            plt_figure=fig,
            fig_name="MeanTrigSig.png",
            figure_save_path=f"{self.fig_save_path}",
            forPres=self.forPres,
        )

    def plot_meanTS_byGroup(self, mean_byGroup: dict, yLim: list = [-2, 14]) -> None:
        """
        Plots the mean time series by group.

        Parameters:
            mean_byGroup (dict): A dictionary containing the mean time series data for each group.
        """

        color_map = self.colors

        if self.xlabels4means is not None:
            sessLabels2use = self.xlabels4means
        else:
            sessLabels2use = [
                self._printf_Session(sess, base_zero=False)
                for sess in self.sess_idx4groups
            ]

        with PdfPages(
            f"{self.fig_save_path}/TrigSig_byGroup{self.file_tag['PDF']}"
        ) as pdf:
            for CC, gdict in mean_byGroup.items():
                fig, axes = self.fig_tools.create_plt_subplots(
                    ncols=self.numSess2use4means,
                    nrows=len(self.groups),
                    figsize=self.figsize,
                )
                axes = axes.flatten()
                plt_idx = 0
                for group, cueTypes in gdict.items():
                    for idx in range(self.numSess2use4means):
                        for cueType, mean_arrs in cueTypes.items():
                            if cueType == "OMITCUE1_SWITCH":
                                continue
                            else:
                                curr_mean_arr = mean_arrs[idx]
                                cell_num = curr_mean_arr.shape[-1]
                                mean_arr2use = []
                                for cn in range(cell_num):
                                    cell_mean = curr_mean_arr[:, cn]
                                    cmax, cmin = np.max(cell_mean), np.min(cell_mean)
                                    min_check = np.abs(cmin) < self.mv_thresh
                                    max_check = cmax < self.mv_thresh
                                    if min_check and max_check:
                                        mean_arr2use.append(cell_mean)
                                mean_arr2use = np.transpose(np.array(mean_arr2use))
                                self._plot_SEM_wCellCount(
                                    axis2plot=axes[plt_idx],
                                    mean_arr=mean_arr2use,
                                    cmap2use=color_map[cueType],
                                )
                        if CC == "CUE":
                            axes[plt_idx].set_ylim(yLim)
                        if group == self.groups[0]:
                            axes[plt_idx].set_title(
                                sessLabels2use[idx], fontsize=self.title_fs
                            )
                        if idx == 0:
                            self._create_bold_yLabel(ax=axes[plt_idx], ylabel=group)
                        if self.numSess > 4:
                            axes[plt_idx].tick_params(
                                axis="both",
                                # which="major",
                                labelsize=self.tiny_fs,
                            )
                        plt_idx += 1

                fig.suptitle(f"{CC}")
                pdf.savefig(fig)
                if self.forPres:
                    self.fig_tools.save_figure(
                        plt_figure=fig,
                        fig_name=f"TrigSig_{CC}",
                        figure_save_path=f"{self.fig_save_path}",
                        forPres=self.forPres,
                        NOPNG=True,
                    )
                self.fig_tools.close_all_figs()

    def plot_meanMV_byGroup(
        self,
        meanMV_byGroup: dict,
        yLim: float = 25,
    ) -> None:
        """
        Plots the mean maximum value by group.

        Parameters:
            meanMV_byGroup (dict): A dictionary containing the mean maximum value by group.
            yLim (float, optional): The y-axis limit. Defaults to 20.
        """

        def _create_legend_name_axes(axes, legend_handles):
            axes.legend(
                handles=legend_handles,
                loc="upper right",
                fontsize=self.fs_tk_lbl,
            )
            axes.set_ylabel("Cue Triggered Peak (Z-score)", fontsize=self.axis_fs)
            axes.set_xlabel("Session", fontsize=self.axis_fs)

        color_map = self.colors

        with PdfPages(
            f"{self.fig_save_path}/MaxVal_byGroup{self.file_tag['PDF']}"
        ) as pdf:
            for CC, gdict in meanMV_byGroup.items():
                fig, axes = self.fig_tools.create_plt_subplots(
                    nrows=len(self.groups), figsize=self.figsize
                )
                if len(self.groups) > 1:
                    axes = axes.flatten()

                for g_idx, (group, cueTypes) in enumerate(gdict.items()):
                    if g_idx == 0:
                        cTypes = cueTypes.keys()
                    if len(self.groups) > 1:
                        axis2plot = axes[g_idx]
                    else:
                        axis2plot = axes

                    for idx in range(self.numSess2use4means):
                        for cueType, mean_arrs in cueTypes.items():
                            mean_arr = np.array(mean_arrs[idx].copy())
                            if mean_arr.size == 0 or np.isnan(mean_arr.flatten()).all():
                                continue
                            else:
                                mean_arr = mean_arr[np.abs(mean_arr) < self.mv_thresh]
                                marr = np.nanmean(mean_arr)
                                sem2plot = sem(mean_arr, nan_policy="omit")
                                x = idx + 1
                                if cueType == "CUE1":
                                    x -= self.group_spacing
                                elif cueType == "OMITCUE1":
                                    x += self.group_spacing
                                self.fig_tools.bar_plot(
                                    ax=axis2plot,
                                    X=x,
                                    Y=marr,
                                    yerr=sem2plot,
                                    color=color_map[cueType],
                                    width=self.group_spacing,
                                )
                    axis2plot.set_ylim(0, yLim)
                    axis2plot.set_title(
                        group, fontsize=self.title_fs, fontweight="bold"
                    )

                fc = []
                labels = []
                for cueType in cTypes:
                    fc.append(color_map[cueType])
                    labels.append(cueType)

                legend_handles = self.fig_tools.create_legend_patch_fLoop(
                    facecolor=fc,
                    label=labels,
                )
                if len(self.groups) > 1:
                    for ax in axes:
                        _create_legend_name_axes(ax, legend_handles)

                else:
                    _create_legend_name_axes(axes, legend_handles)

                fig.suptitle(f"{CC}")
                pdf.savefig(fig)
                if self.forPres:
                    self.fig_tools.save_figure(
                        plt_figure=fig,
                        fig_name=f"MaxVal_{CC}",
                        figure_save_path=f"{self.fig_save_path}",
                        forPres=self.forPres,
                        NOPNG=True,
                    )
                self.fig_tools.close_all_figs()

    def plot_meanMV_cueFocus(self, meanMV_byGroup: dict, yLim: float = 20) -> None:
        """
        Plots the mean maximum value for cue focus.

        Parameters:
            meanMV_byGroup (dict): A dictionary containing the mean maximum value by group.
            yLim (float, optional): The y-axis limit. Defaults to 20.
        """

        mean_MV4CC = meanMV_byGroup["CUE"]

        color_map = self.colors4barGroup

        fig, axes = self.fig_tools.create_plt_subplots(figsize=self.figsize)
        # width = self.swidth if len(self.groups) == 3 else self.bwidth
        width = self.bwidth
        numCC = {gr: [] for gr in self.groups4sig}

        mean_semVals = {
            gr: {f"S{i}": {"MEAN": np.nan, "SEM": np.nan} for i in self.sess_idx4groups}
            for gr in self.groups4sig
        }
        for g_idx, (group, cueTypes) in enumerate(mean_MV4CC.items()):
            for idx in range(self.numSess2use):
                for cueType, mean_arrs in cueTypes.items():
                    if cueType == "CUE1":
                        mean_arr = np.array(mean_arrs[idx].copy())
                        if mean_arr.size == 0 or np.isnan(mean_arr.flatten()).all():
                            continue
                        else:
                            mean_arr = mean_arr[np.abs(mean_arr) < self.mv_thresh]
                            numCC[group].append(len(mean_arr))
                            marr = np.nanmean(mean_arr)
                            sem2plot = sem(mean_arr, nan_policy="omit")

                            mean_semVals[group][f"S{idx + 1}"] = {
                                "MEAN": marr,
                                "SEM": sem2plot,
                            }

                            x = idx + 1
                            x += (
                                self.group_spacing
                                if group in ["AD", "AGED"]
                                else -self.group_spacing
                                if group in ["CTL", "DK", "NONAGED"]
                                else 0
                            )
                            self.fig_tools.bar_plot(
                                ax=axes,
                                X=x,
                                Y=marr,
                                yerr=sem2plot,
                                color=color_map[group],
                                width=width,
                            )
        self.total_Cell_bySession_textBox(
            title="Number of Cue Cells", numCellDict_byGroup=numCC, axis2plot=axes
        )

        pVals = {f"S{i + 1}": np.nan for i in range(self.numSess2use)}
        for s_idx, (g1_arr, g2_arr) in enumerate(
            zip(
                mean_MV4CC[self.groups4sig[0]]["CUE1"],
                mean_MV4CC[self.groups4sig[-1]]["CUE1"],
            )
        ):
            g1_arr = np.array(g1_arr)
            g2_arr = np.array(g2_arr)
            g1_arr = g1_arr[np.abs(g1_arr) < self.mv_thresh]
            g2_arr = g2_arr[np.abs(g2_arr) < self.mv_thresh]
            pVal = self.fig_tools.create_sig_2samp_annotate(
                ax=axes,
                arr0=g1_arr,
                arr1=g2_arr,
                coords=(s_idx + 1, None),
                return_Pval=True,
            )
            pVals[f"S{s_idx + 1}"] = pVal

        legend_handles = self.fig_tools.create_legend_patch_fLoop(
            facecolor=[color_map[group] for group in self.groups4sig],
            label=self.groups4sig,
        )

        axes.legend(handles=legend_handles, loc="upper right", fontsize=self.fs_tk_lbl)
        axes.set_xticks(self.sess_idx4groups)
        axes.tick_params(axis="both", labelsize=self.fs_tk_lbl)
        axes.set_ylim(0, yLim)
        self._create_bold_yLabel(
            ax=axes, ylabel="Cue Triggered Peak for CUE1 lapType (Z-score)"
        )
        axes.set_xlabel("Session", fontsize=self.axis_fs, fontweight=self.fontweight)
        axes.set_title("Cue Cells per session across subjects", fontsize=self.title_fs)

        self.fig_tools.save_figure(
            plt_figure=fig,
            fig_name="MaxVal_CueCells",
            figure_save_path=f"{self.fig_save_path}",
            forPres=self.forPres,
            pValDict=pVals,
            mean_semDict=mean_semVals,
        )

    def _plot_SEM_wCellCount(
        self,
        axis2plot: object,
        mean_arr: list,
        cmap2use: str,
    ) -> None:
        """
        Plot the standard error of the mean (SEM) with cell count.

        Parameters:
            axis2plot (matplotlib.axes.Axes): The axes object to plot on.
            mean_arr (list or numpy.ndarray): The array of mean values.
            cmap2use (str): The colormap to use for plotting.
        """

        marr = np.array(mean_arr)
        if marr.size == 0 or np.isnan(marr.flatten()).all():
            cell_count = 0
        else:
            cell_count = np.count_nonzero(np.any(~np.isnan(marr), axis=0))
            self.fig_tools.plot_SEM(
                arr=marr,
                color=cmap2use,
                ax=axis2plot,
                x_ind=self.ind,
                vline=True,
                # baseline=self.baseline_slice,
            )
        self.fig_tools.add_text_box(ax=axis2plot, text=f"N = {cell_count}", va="top")

    def plot_PCR_Tuning(
        self, posRate_byGroup: dict, posRate_byGroup_CC: dict, yLim4max: float = 0.4
    ) -> None:
        """
        Plots the PCR tuning.

        Parameters:
            posRate_byGroup (dict): A dictionary containing the position rate by group.
            posRate_byGroup_CC (dict): A dictionary containing the position rate by group and cue.
            yLim4max (float, optional): The y-axis limit for the maximum position rate. Defaults to 0.4.
        """

        def _find_meanNmaxPR(arr4mean: np.ndarray) -> tuple:
            """
            Find the mean and maximum position rate.

            Parameters:
                arr4mean (np.ndarray): The array of position rates.

            Returns:
                tuple: A tuple containing the mean position rate, the standard error of the mean position rate,
                the mean maximum position rate, and the standard error of the mean maximum position rate.
            """
            mean_PR = np.mean(arr4mean, axis=0)
            sem_PR = sem(arr4mean, axis=0)

            max_PR = np.max(arr4mean, axis=1)
            mean_maxPR = np.mean(max_PR)
            sem_maxPR = sem(max_PR)

            return mean_PR, sem_PR, mean_maxPR, sem_maxPR

        def _nan_check4numCC(numCC: list, arr: np.ndarray) -> list:
            """
            Check for NaN values and update the cell count.

            Parameters:
                numCC (list): The list of cell counts.
                arr (np.ndarray): The array of position rates.

            Returns:
                list: The updated list of cell counts.
            """
            if np.all(np.isnan(arr)):
                numCC.append(0)
            else:
                numCC.append(arr.shape[0])

            return numCC

        xticks = [0, 25, 50, 75, 100]
        xlabel = "Position"
        colors = self.session_colors
        colors4barGroup = self.colors4barGroup

        mean_semVals = {gr: {} for gr in self.groups4sig}

        if self.xlabels4means is not None:
            sessLabels2use = self.xlabels4means
        else:
            sessLabels2use = [f"S{i}" for i in self.sess_idx4groups]

        maxVal = 0
        for group, posRate_byLT in posRate_byGroup.items():
            mean_semVals[group] = {
                lt: {
                    f"S{i}": {"MEAN": np.nan, "SEM": np.nan}
                    for i in self.sess_idx4groups
                }
                for lt in posRate_byLT.keys()
            }
            for lt, posRate_bySess in posRate_byLT.items():
                for posRate in posRate_bySess:
                    maxVal = max(maxVal, np.percentile(posRate, 99.9))

            with PdfPages(
                f"{self.fig_save_path}/posRateTuning_{group}{self.file_tag['PDF']}"
            ) as pdf:
                for lt, posRate_bySess in posRate_byLT.items():
                    # maxVal = 0
                    numCols = int(np.ceil(np.sqrt(self.numSess2use4means * 1.5)))
                    numRows = int(np.ceil(self.numSess2use4means / numCols))

                    if numCols + numRows == 3:
                        numCols = 2
                        numRows = 2
                    elif numCols + numRows == 4:
                        numCols = 3
                    elif self.numSess2use4means == 3:
                        numCols = 3
                        numRows = 2
                    elif self.numSess2use4means == 5:
                        numCols = 4
                        numRows = 2
                    elif self.numSess2use4means == 7:
                        # numRows = 3
                        numCols = 5

                    fig, axes = self.fig_tools.create_plt_subplots(
                        ncols=numCols, nrows=numRows, figsize=self.figsize
                    )
                    axes = axes.flatten()
                    ax4max = axes[-2]
                    ax4mean = axes[-1]

                    numCC = []
                    for posRate in posRate_bySess:
                        # maxVal = max(maxVal, np.percentile(posRate, 99.5))
                        numCC.append(posRate.shape[0])
                    for sess, posRate in enumerate(posRate_bySess):
                        if sess >= self.numSess2use4means:
                            break
                        ax2plot = axes[sess]
                        sortInd = np.argsort(np.argmax(posRate, axis=1))
                        posRate = posRate[sortInd, :]
                        posRate_CC = posRate_byGroup_CC[group][lt][sess]

                        if posRate_CC.size == 0:
                            posRate_CC = np.full_like(posRate, np.nan)

                        im = self.fig_tools.plot_imshow(
                            fig=fig,
                            axis=ax2plot,
                            data2plot=posRate,
                            xlabel=xlabel,
                            xticks=xticks,
                            title=sessLabels2use[sess],
                            aspect="auto",
                            cmap=self.cmap,
                            vmin=0,
                            vmax=maxVal,
                            return_im=True,
                        )
                        mean_posRate, sem_posRate, mean_maxPR, sem_maxPR = (
                            _find_meanNmaxPR(posRate)
                        )
                        _, _, mean_maxPR_CC, sem_maxPR_CC = _find_meanNmaxPR(posRate_CC)

                        Xs = [
                            sess + 1 - self.group_spacing,
                            sess + 1 + self.group_spacing,
                        ]
                        HTCHs = [None, self.cueHatch]
                        Ys = [(mean_maxPR, sem_maxPR), (mean_maxPR_CC, sem_maxPR_CC)]
                        mean_semVals[group][lt][f"S{sess + 1}"] = {
                            "MEAN": mean_maxPR,
                            "SEM": sem_maxPR,
                        }

                        for x, y, htch in zip(Xs, Ys, HTCHs):
                            self.fig_tools.bar_plot(
                                ax=ax4max,
                                X=x,
                                Y=y[0],
                                yerr=y[1],
                                color=colors[sess],
                                edgecolor=self.color_dict["black"],
                                width=self.bwidth,
                                hatch=htch,
                            )

                        ax4mean.plot(
                            mean_posRate, color=colors[sess], label=sessLabels2use[sess]
                        )
                        ax4mean.fill_between(
                            range(len(mean_posRate)),
                            mean_posRate - sem_posRate,
                            mean_posRate + sem_posRate,
                            color=colors[sess],
                            alpha=0.2,
                        )
                        cbar_ref = im
                    self.fig_tools.set_cbar_location(fig, cbar_ref)

                    ax4mean.legend()
                    ax4mean.set_title("Mean posRate")
                    ax4mean.set_xlabel(xlabel)
                    ax4mean.set_xticks(xticks)

                    ax4max.set_title("Mean peak amplitude across cells")
                    ax4max.set_xticks(self.sess_idx4groups)
                    ax4max.set_xticklabels(sessLabels2use, rotation=45)
                    ax4max.set_xlabel("Session")
                    ax4max.set_ylim(0, yLim4max)
                    ax4max.set_rasterized(False)

                    cc_patch = self.fig_tools.create_legend_patch_fLoop(
                        facecolor=["none"] * len(self.cellTypes),
                        edgecolor=[self.color_dict["black"]] * len(self.cellTypes),
                        hatch=[None, self.cueHatch],
                        label=self.cellTypes,
                        alpha=[None] * len(self.cellTypes),
                    )
                    ax4max.legend(
                        handles=cc_patch,
                        loc="upper right",
                        fontsize=self.tiny_fs,
                    )
                    self.total_Cell_bySession_textBox(
                        title="Tuned Cell Total",
                        numCellDict_byGroup=numCC,
                        axis2plot=ax4max,
                        fontsize=self.tiny_fs,
                        byGroup=False,
                    )

                    self.fig_tools.delete_axes(fig)

                    fig.suptitle(f"{group} - {lt}")

                    for ax in axes:
                        if not ax.has_data():
                            ax.axis("off")

                    pdf.savefig(fig)

                    if self.forPres:
                        self.fig_tools.save_figure(
                            plt_figure=fig,
                            fig_name=f"posRate_{group}_{lt}",
                            figure_save_path=self.fig_save_path,
                            tight_layout=False,
                            forPres=self.forPres,
                            NOPNG=True,
                        )
                    self.fig_tools.close_all_figs()

        mean_semVal_fname = "mean_sem/peakSpatialRate_TunedCells_byLT"
        self.savedict2file(
            mean_semVals,
            mean_semVal_fname,
            filename=f"{self.fig_save_path}/{mean_semVal_fname}",
            filetype_to_save=self.file_tag["JSON"],
        )

        fig, ax = self.fig_tools.create_plt_subplots()
        numCC = {gr: [] for gr in self.groups4sig}
        pVals = {f"S{i}": np.nan for i in self.sess_idx4groups}
        mean_semVals = {
            gr: {f"S{i}": {"MEAN": np.nan, "SEM": np.nan} for i in self.sess_idx4groups}
            for gr in self.groups4sig
        }
        for s_idx, (posArr0, posArr1) in enumerate(
            zip(
                posRate_byGroup_CC[self.groups4sig[0]]["CUE1"],
                posRate_byGroup_CC[self.groups4sig[-1]]["CUE1"],
            )
        ):
            if posArr0.size == 0:
                posArr0 = np.full_like(posRate, np.nan)
            if posArr1.size == 0:
                posArr1 = np.full_like(posRate, np.nan)

            if len(self.groups4sig) == 1:
                Xs = [s_idx + 1]
            else:
                Xs = [
                    s_idx + 1 - self.group_spacing,
                    s_idx + 1 + self.group_spacing,
                ]

            numCC[self.groups4sig[0]] = _nan_check4numCC(
                numCC[self.groups4sig[0]], posArr0
            )
            if len(self.groups4sig) > 1:
                numCC[self.groups4sig[-1]] = _nan_check4numCC(
                    numCC[self.groups4sig[-1]], posArr1
                )

            max0, max1 = np.max(posArr0, axis=1), np.max(posArr1, axis=1)
            Ys = [np.nanmean(max0), np.nanmean(max1)]
            Yerrs = [sem(max0, nan_policy="omit"), sem(max1, nan_policy="omit")]

            if len(self.groups4sig) == 1:
                Ys = [Ys[0]]
                Yerrs = [Yerrs[0]]

            for group in self.groups4sig:
                mean_semVals[group][f"S{s_idx + 1}"] = {
                    "MEAN": Ys[self.groups4sig.index(group)],
                    "SEM": Yerrs[self.groups4sig.index(group)],
                }

            for b_idx, (x, y, yerr) in enumerate(zip(Xs, Ys, Yerrs)):
                if b_idx <= len(self.groups4sig):
                    self.fig_tools.bar_plot(
                        ax=ax,
                        X=x,
                        Y=y,
                        yerr=yerr,
                        color=colors4barGroup[self.groups4sig[b_idx]],
                        width=self.bwidth,
                    )

            pVal = self.fig_tools.create_sig_2samp_annotate(
                ax=ax,
                arr0=max0,
                arr1=max1,
                coords=(s_idx + 1, None),
                return_Pval=True,
            )
            pVals[f"S{s_idx + 1}"] = pVal

        self.total_Cell_bySession_textBox(
            title="Number of Cue Cells", numCellDict_byGroup=numCC, axis2plot=ax
        )

        if len(self.groups4sig) > 1:
            fc = [colors4barGroup[gr] for gr in self.groups4sig]
            legend_handles = self.fig_tools.create_legend_patch_fLoop(
                facecolor=fc, label=self.groups4sig
            )
            ax.legend(handles=legend_handles, loc="upper right")

        ax.set_xticks(self.sess_idx4groups)
        ax.set_xticklabels(sessLabels2use)
        ax.set_ylim(0, yLim4max)
        ax.set_ylabel("Peak Spatial Firing Rate (Hz)", fontsize=self.fontsize)
        ax.set_title("Cue Cells per session across subjects", fontsize=self.title_fs)

        ax.tick_params(axis="both", labelsize=self.fs_tk_lbl)

        self.fig_tools.save_figure(
            plt_figure=fig,
            fig_name="peakSpatialRate_CueCells",
            figure_save_path=f"{self.fig_save_path}",
            forPres=self.forPres,
            pValDict=pVals,
            mean_semDict=mean_semVals,
        )

    def plot_cellProps_byGroup(self, cellTotals_byGroup: dict) -> None:
        """
        Plots the cell proportions by group.


        Parameters:
            cellTotals_byGroup (dict): A dictionary containing the cell totals by group.
        """

        fig, ax2plot = self.fig_tools.create_plt_subplots(figsize=self.figsize)
        colors = self.colors4barGroup

        if self.xlabels4means is not None:
            sessLabels2use = self.xlabels4means
        else:
            sessLabels2use = self.sess_idx4groups

        width = 0.2
        ratios = [
            key
            for key in cellTotals_byGroup[self.groups4sig[0]].keys()
            if key not in self.cTypes_All
        ]
        # cellProps_byGroup = {gr: {rat: [] for rat in ratios} for gr in self.groups4sig}

        numCC = {gr: {ct: [] for ct in self.cTypes_All} for gr in self.groups4sig}
        for group in self.groups4sig:
            for ct in self.cTypes_All:
                for sess in range(self.numSess2use4means):
                    mean_cells = np.nanmean(cellTotals_byGroup[group][ct][sess])
                    sem_cells = sem(
                        cellTotals_byGroup[group][ct][sess], nan_policy="omit"
                    )
                    numCC[group][ct].append((mean_cells, sem_cells))
        # for sess in range(self.numSess2use):
        #     cellProps_byGroup[group]["CC/TC"].append(
        #         numCC4prop[group]["CC"][sess] / numCC4prop[group]["TC"][sess]
        #     )
        #     cellProps_byGroup[group]["TC/TOT"].append(
        #         numCC4prop[group]["TC"][sess] / numCC4prop[group]["ALL"][sess]
        #     )
        self.total_Cell_bySession_textBox(
            title="Mean Total Cells:",
            numCellDict_byGroup=numCC,
            axis2plot=ax2plot,
            fontsize=self.small_fs,
        )

        pVals = {r: {f"S{i}": np.nan for i in self.sess_idx4groups} for r in ratios}
        mean_semVals = {
            r: {
                gr: {
                    f"S{i}": {"MEAN": np.nan, "SEM": np.nan}
                    for i in self.sess_idx4groups
                }
                for gr in self.groups4sig
            }
            for r in ratios
        }
        for r_idx, rat in enumerate(ratios):
            if rat == "CC/TC":
                hatch2use = self.cueHatch
            elif rat == "TC/TOT":
                hatch2use = None
            arr4sig = []

            for g_idx, group in enumerate(self.groups4sig):
                for sess in self.sess_idx4groups:
                    arr2check = cellTotals_byGroup[group][rat][sess - 1]
                    if arr2check.size < self.numSess2use4means:
                        arr2check = np.append(arr2check, np.nan)
                        cellTotals_byGroup[group][rat][sess - 1] = arr2check
                arr2use = np.array(cellTotals_byGroup[group][rat])
                arr4sig.append(arr2use)
                meanVal = np.nanmean(arr2use, axis=1)
                semVal = sem(arr2use, nan_policy="omit", axis=1)
                for sess in range(self.numSess2use4means):
                    mean_semVals[rat][group][f"S{sess + 1}"] = {
                        "MEAN": meanVal[sess],
                        "SEM": semVal[sess],
                    }

                bar_pos = self.fig_tools.create_index4grouped_barplot(
                    n_bars=len(self.groups4sig) * len(ratios),
                    index=self.sess_idx4groups,
                    width=width,
                    measureLoop_idx=r_idx,
                    num_groups=len(self.groups4sig),
                    gLoop_idx=g_idx,
                )

                self.fig_tools.bar_plot(
                    ax=ax2plot,
                    X=bar_pos,
                    Y=meanVal,
                    width=width,
                    label=group,
                    yerr=semVal,
                    hatch=hatch2use,
                    color=colors[group],
                )
            for s_idx, (arr0, arr1) in enumerate(zip(arr4sig[0], arr4sig[-1])):
                pVal = self.fig_tools.create_sig_2samp_annotate(
                    ax=ax2plot,
                    arr0=arr0,
                    arr1=arr1,
                    coords=(
                        self.sess_idx4groups[s_idx]
                        - width
                        + (width * r_idx * len(ratios)),
                        None,
                    ),
                    xytext=(-width, 0),
                    return_Pval=True,
                )
                pVals[rat][f"S{s_idx + 1}"] = pVal

        fc = [colors[gr] for gr in self.groups4sig] + ["none"] * len(ratios)
        legend_handles = self.fig_tools.create_legend_patch_fLoop(
            facecolor=fc,
            label=[group for group in self.groups4sig] + list(ratios),
            hatch=[None] * len(self.groups4sig) + [self.cueHatch, None],
            edgecolor=[self.color_dict["black"]] * len(fc),
            alpha=[1.0] * len(self.groups4sig) + [None] * 2,
        )
        ax2plot.set_xticks(self.sess_idx4groups)
        ax2plot.set_xticklabels(sessLabels2use)
        ax2plot.tick_params(axis="both", labelsize=self.fs_tk_lbl)

        self._create_bold_yLabel(ax=ax2plot, ylabel="Proportion")
        ax2plot.set_ylim(0, 0.7)
        ax2plot.set_title(
            "Proportion by Cell Type per session per subject",
            fontsize=self.title_fs,
        )
        ax2plot.legend(handles=legend_handles, loc="upper right", fontsize="large")
        self.fig_tools.save_figure(
            plt_figure=fig,
            fig_name="cellProps_byGroup",
            figure_save_path=f"{self.fig_save_path}",
            forPres=self.forPres,
            pValDict=pVals,
            mean_semDict=mean_semVals,
        )

    def plot_genRatesbyGroup(
        self, rates_ByGroup: dict, ylabel: str, fname_pre: str, yLim: float | None
    ) -> None:
        """
        Function that plots several rate types by group (eventRates, InfoPerSpk, InfoPerSec).

        Parameters:
            rates_ByGroup (dict): A dictionary containing the rates by group.
            ylabel (str): The y-axis label.
            fname_pre (str): The prefix for the filename.
            yLim (float | None): The y-axis limit.
        """

        fig, ax2plot = self.fig_tools.create_plt_subplots(figsize=self.figsize)
        colors = self.other_colors
        width2use = self.Sswidth
        n_bars = len(self.groups4sig) * len(self.cTypes_All)
        offset_scaling = 1.5

        if self.xlabels4means is not None:
            sessLabels2use = self.xlabels4means
        else:
            sessLabels2use = self.sess_idx4groups

        numCC = {gr: {ct: [] for ct in self.cTypes_All} for gr in self.groups4sig}
        pVals = {
            ct: {f"S{i}": np.nan for i in self.sess_idx4groups}
            for ct in self.cTypes_All
        }
        mean_semVals = {
            gr: {
                ct: {
                    f"S{i}": {"MEAN": np.nan, "SEM": np.nan}
                    for i in self.sess_idx4groups
                }
                for ct in self.cTypes_All
            }
            for gr in self.groups4sig
        }
        hatch_list = []
        for group in self.groups4sig:
            if group in ["AD", "AGED"]:
                hatch_list.append(self.expHatch)
            else:
                hatch_list.append(None)
        for c_idx, ctype in enumerate(self.cTypes_All):
            arr4sig = []
            bar_posits = []
            for g_idx, group in enumerate(self.groups4sig):
                arr2use = rates_ByGroup[group][ctype]
                for arr in arr2use:
                    numCC[group][ctype].append(len(arr))

                arr4sig.append(arr2use)
                meanVal = np.array([np.nanmean(arr) for arr in arr2use])
                semVal = np.array([sem(arr, nan_policy="omit") for arr in arr2use])
                for sess in range(self.numSess2use4means):
                    mean_semVals[group][ctype][f"S{sess + 1}"] = {
                        "MEAN": meanVal[sess],
                        "SEM": semVal[sess],
                    }

                bar_pos = self.fig_tools.create_index4grouped_barplot(
                    n_bars=n_bars,
                    index=self.sess_idx4groups,
                    width=width2use,
                    measureLoop_idx=c_idx,
                    num_groups=len(self.groups4sig),
                    gLoop_idx=g_idx,
                    offset_scaling=offset_scaling,
                )

                bar_posits.append(bar_pos)

                self.fig_tools.bar_plot(
                    ax=ax2plot,
                    X=bar_pos,
                    Y=meanVal,
                    width=width2use,
                    label=group,
                    yerr=semVal,
                    hatch=hatch_list[g_idx],
                    color=colors[c_idx],
                )
            for s_idx, (arr0, arr1, bp0, bp1) in enumerate(
                zip(arr4sig[0], arr4sig[-1], bar_posits[0], bar_posits[-1])
            ):
                center_pos = (bp0 + bp1) / 2
                pVal = self.fig_tools.create_sig_2samp_annotate(
                    ax=ax2plot,
                    arr0=arr0,
                    arr1=arr1,
                    coords=(
                        center_pos,
                        None,
                    ),
                    xytext=(-2 * width2use, 0),
                    fontsize=10,
                    return_Pval=True,
                )
                pVals[ctype][f"S{s_idx + 1}"] = pVal

        self.total_Cell_bySession_textBox(
            title="Number of Cells",
            numCellDict_byGroup=numCC,
            axis2plot=ax2plot,
            fontsize=self.small_fs,
        )

        ax2plot.set_xticks(self.sess_idx4groups)
        ax2plot.set_xticklabels(sessLabels2use)
        if yLim is not None:
            ax2plot.set_ylim(0, yLim)
        ax2plot.tick_params(axis="both", labelsize=self.fs_tk_lbl)

        self._create_bold_yLabel(ax=ax2plot, ylabel=ylabel)

        fc = [colors[idx] for idx in range(len(self.cTypes_All))] + ["none"] * len(
            self.groups4sig
        )
        labels = (
            self.cTypes_All + self.groups4sig
            if len(self.groups4sig) > 1
            else self.cTypes_All
        )

        hatches = [None] * len(self.cTypes_All) + hatch_list
        alpha = [1.0] * len(self.cTypes_All) + [None] * 2
        if len(self.groups4sig) == 1:
            fc = fc[:-1]
            hatches = hatches[:-1]
            alpha = alpha[:-1]

        legend_handles = self.fig_tools.create_legend_patch_fLoop(
            facecolor=fc,
            label=labels,
            edgecolor=[self.color_dict["black"]] * len(fc),
            hatch=hatches,
            alpha=alpha,
        )
        ax2plot.legend(
            handles=legend_handles, loc="upper right", fontsize=self.fs_tk_lbl
        )
        ax2plot.set_title(
            "Mean Rate by Cell Type per session across subjects", fontsize=self.title_fs
        )

        self.fig_tools.save_figure(
            plt_figure=fig,
            fig_name=f"{fname_pre}_byGroup",
            figure_save_path=f"{self.fig_save_path}",
            forPres=self.forPres,
            pValDict=pVals,
            mean_semDict=mean_semVals,
        )

    def plot_eventRateScatter_eOPN3(self, eventRate_byGroup: dict) -> None:
        """
        Function that plots the event rate scatter for eOPN3.

        Parameters:
            eventRate_byGroup (dict): A dictionary containing the event rate by group.
        """

        fig, axes = self.fig_tools.create_plt_subplots(
            ncols=int(self.numSess2use4means / 2), figsize=self.figsize
        )
        axes = axes.flatten()

        arrays2plot = eventRate_byGroup["eOPN3"]["ALL"]

        jitter_amt = 0.0001

        for count, arr_idx in enumerate(range(0, len(arrays2plot), 2)):
            ax2plot = axes[count]
            arrayPre = arrays2plot[arr_idx]
            arrayPost = arrays2plot[arr_idx + 1]

            jitterPre = np.random.normal(0, jitter_amt, size=arrayPre.shape)
            jitterPost = np.random.normal(0, jitter_amt, size=arrayPost.shape)

            arrayPre += jitterPre
            arrayPost += jitterPost

            ax2plot.plot(arrayPre, arrayPost, "o")
            ax2plot.set_xlabel(
                "Pre-eOPN3 mean Event Rate (events/s)", fontsize=self.axis_fs
            )
            ax2plot.set_ylabel(
                "Post-eOPN3 mean Event Rate (events/s)", fontsize=self.axis_fs
            )
            ax2plot.set_title(f"Session {count + 1}", fontsize=self.axis_fs)

            ax2plot.set_xlim(0, 0.006)
            ax2plot.set_ylim(0, 0.008)

        fig.suptitle("Event Rate Scatter eOPN3 - ALL CELLS", fontsize=self.title_fs)

        self.fig_tools.save_figure(
            plt_figure=fig,
            fig_name="EventRateScatter_eOPN3",
            figure_save_path=f"{self.fig_save_path}",
            forPres=self.forPres,
        )

    def plot_OptoSingleSubj(self, optoArr2plot: dict) -> None:
        """
        Function that plots the opto single subject.

        Parameters:
            optoArr2plot (dict): A dictionary containing the opto array to plot.
        """

        fig, axes = self.fig_tools.create_plt_subplots(figsize=self.figsize)

        cues4opto = list(optoArr2plot[0].keys())

        max_val = 0
        outlier = [[], 0]

        for idx, optoArr in enumerate(optoArr2plot):
            cue_vals = optoArr[cues4opto[0]]
            cwo_vals = optoArr["CUEwOPTO"]

            for cval, cwval in zip(cue_vals, cwo_vals):
                cell, camp, cstd = cval
                _, cwamp, cwstd = cwval

                cell_num = cell.split("_")[-1]

                if camp > 100 or cwamp > 100:
                    outlier[-1] += 1
                    outlier[0].append(cell)
                    continue

                max_val = max(max_val, camp, cwamp)

                axes.errorbar(
                    camp,
                    cwamp,
                    xerr=cstd,
                    yerr=cwstd,
                    fmt="o",
                    color=self.session_colors[idx],
                    ecolor=self.session_colors[idx],
                    markersize=10,
                    label=f"S{idx + 1}",
                )

                axes.annotate(
                    cell_num,
                    (camp, cwamp),
                    textcoords="offset points",
                    xytext=(10, 8),
                    ha="center",
                    color=self.session_colors[idx],
                    fontsize=10,
                )
        axes.plot([0, max_val], [0, max_val], linestyle="--", color="gray")
        axes.set_xlabel("CUE Only Max Amplitude", fontsize=self.axis_fs)
        axes.set_ylabel("CUEwOPTO Max Amplitude", fontsize=self.axis_fs)

        self.fig_tools.create_legend_wNO_duplicates(axes)

        if outlier[-1] > 0:
            outliers_str = ", ".join([f"{cell}" for cell in outlier[0]])
            outliers_test = f"Outliers: {outliers_str}"
            self.fig_tools.add_text_box(
                ax=axes,
                text=outliers_test,
                va="top",
            )

        self.fig_tools.save_figure(
            plt_figure=fig,
            fig_name="cueVScueOpto",
            figure_save_path=f"{self.fig_save_path}",
            forPres=self.forPres,
        )

    def plot_OptoAllSubj(self, OptoAmpArrAllSubj: dict, threshold: int = 100) -> None:
        """
        Function that plots the opto for all subjects.

        Parameters:
            OptoAmpArrAllSubj (dict): A dictionary containing the opto amplitude array for all subjects.
            threshold (int, optional): The threshold for the maximum amplitude. Defaults to 100.
        """

        cues4opto = list(OptoAmpArrAllSubj.keys())
        CUE = cues4opto[0]
        CWO = cues4opto[1]

        fig, axes = self.fig_tools.create_plt_subplots(figsize=self.figsize)

        color44_45 = [self.color_dict["orange"], self.color_dict["cyan"]]

        max_val = 0
        outlier = [[], 0]
        for (cue_id, sess, cue_cell, cue_amp, cue_std), (
            cwo_id,
            _,
            cwo_cell,
            cwo_amp,
            cwo_std,
        ) in zip(OptoAmpArrAllSubj[CUE], OptoAmpArrAllSubj[CWO]):
            if cue_id in ["CA3DG44", "CA3DG45"]:
                color2use = color44_45[sess]
                label2use = f"S{sess + 2} (44-45)"
                if sess == 1:
                    continue
            else:
                color2use = self.session_colors[sess]
                label2use = f"S{sess + 1}"

            if cue_amp > threshold or cwo_amp > threshold:
                outlier[-1] += 1
                outlier[0].append([cue_id, sess, cue_cell])
                continue

            max_val = max(max_val, cue_amp, cwo_amp)
            axes.errorbar(
                cue_amp,
                cwo_amp,
                xerr=cue_std,
                yerr=cwo_std,
                fmt="o",
                color=color2use,
                ecolor=color2use,
                markersize=10,
                label=label2use,
            )
        axes.plot([0, max_val], [0, max_val], linestyle="--", color="gray")
        axes.set_xlabel("CUE Only Max Amplitude", fontsize=self.axis_fs)
        axes.set_ylabel("CUEwOPTO Max Amplitude", fontsize=self.axis_fs)

        self.fig_tools.create_legend_wNO_duplicates(axes)

        if outlier[-1] > 0:
            outliers_str = ", ".join(
                [
                    f"[{cue_id}, S{sess + 1}, {cell}]"
                    for cue_id, sess, cell in outlier[0]
                ]
            )
            outliers_test = f"Outliers: {outliers_str}"
            self.fig_tools.add_text_box(
                ax=axes,
                text=outliers_test,
                va="top",
                bbox=dict(
                    boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"
                ),
            )

        self.fig_tools.save_figure(
            plt_figure=fig,
            fig_name="cueVScueOpto",
            figure_save_path=f"{self.fig_save_path}",
            forPres=self.forPres,
        )

    def _create_bold_yLabel(self, ax: object, ylabel: str) -> None:
        """
        Sets the y-label of the given axis with the specified font size and weight.

        Parameters:
            ax (matplotlib.axes.Axes): The axis object to set the y-label for.
            ylabel (str): The label to be set as the y-label.
        """

        ax.set_ylabel(
            f"{ylabel}", fontsize=self.fontsize_ylabel, fontweight=self.fontweight
        )

    def _printf_Session(self, idx: int, base_zero: bool = True) -> str:
        """
        Returns a formatted string representing the session number.

        Parameters:
            idx (int): The index of the session.

        Returns:
            str: A formatted string representing the session number.
        """

        return f"Session {idx + 1}" if base_zero else f"Session {idx}"

    def _create_sep_legend(self) -> None:
        """
        Creates a separate legend for the plot.

        This method creates a legend for the plot based on the color mapping
        defined by the `colors` attribute. The legend elements are created
        based on the `cue_types_set` and `desired_order` attributes.
        """

        color_map = {
            key: self.colors[key]
            for key in (self.cueTypes_set & set(self.desired_order))
            if key in self.colors
        }
        fig_legend = self.fig_tools.create_separate_legend(color_map=color_map)
        self.fig_tools.save_figure(
            plt_figure=fig_legend,
            fig_name="_legend.png",
            figure_save_path=self.fig_save_path,
            forPres=self.forPres,
        )

    def numSessChecker(self, sessFocus: int = None) -> int:
        """
        Checks the number of sessions to use.

        Parameters:
            sessFocus (int, optional): The number of sessions to use. Defaults to None.

        Returns:
            int: The number of sessions to use.

        Notes:
            If sessFocus is None, the number of sessions to use is the number of sessions in the object.
        """

        numSess2use = self.numSess if sessFocus is None else sessFocus
        if numSess2use > self.numSess:
            numSess2use = self.numSess
        return numSess2use

    def numSessByIDChecker(self, sessFocus: int = None) -> int:
        """
        Checks the number of sessions to use by ID.

        Parameters:
            sessFocus (int, optional): The number of sessions to use. Defaults to None.

        Returns:
            int: The number of sessions to use.

        Notes:
            If all session numbers are the same, the number of sessions to use is the number of sessions in the object.
            Otherwise, the number of sessions to use is the maximum session number across subjects. If sessFocus is not None, the number of sessions to use is sessFocus.
        """

        if all(value == self.numSessByID[0] for value in self.numSessByID):
            numSess2use4means = self.numSess2use
        else:
            numSess2use4means = int(max(self.numSessByID))

        if sessFocus is not None:
            if sessFocus > numSess2use4means:
                numSess2use4means = numSess2use4means
            else:
                numSess2use4means = sessFocus

        return numSess2use4means

    def total_Cell_bySession_textBox(
        self,
        title: str,
        numCellDict_byGroup: dict,
        axis2plot: object,
        fontsize: int | None = None,
        byGroup: bool = True,
    ) -> None:
        """
        Function that plots textbox of cell counts by group.

        Parameters:
            title (str): The title of the textbox.
            numCellDict_byGroup (dict): A dictionary containing the number of cells by group.
            axis2plot (object): The axis to plot the textbox on.
            fontsize (int | None, optional): The fontsize of the textbox. Defaults to None.
            byGroup (bool, optional): Whether to plot the cell counts by group. Defaults to True.
        """

        fontsize = self.fs_tk_lbl if fontsize is None else fontsize
        text_content = [title]
        if byGroup:
            for group, counts in numCellDict_byGroup.items():
                if isinstance(counts, dict):
                    text_content.append(f"{group:<4}:")
                    for ct_key, ct_val in counts.items():
                        sess_strings = [
                            f"S{i + 1:<2}: {val[0]:6.1f} ± {val[1]:6.1f}"
                            if isinstance(val, tuple)
                            else f"S{i + 1:<2}: {val:>2}"
                            for i, val in enumerate(ct_val)
                        ]
                        sessions = "  |  ".join(sess_strings)
                        text_content.append(" " * 2 + f"{ct_key:<3}: {sessions}")
                else:
                    sess_strings = [
                        f"S{i + 1:<2}: {count:>2}" for i, count in enumerate(counts)
                    ]
                    sessions = " | ".join(sess_strings)
                    text_content.append(f"{group:<4}: {sessions}")
        else:
            for i, count in enumerate(numCellDict_byGroup):
                text_content.append(f"S{i + 1:<2}: {count:>2}")
        text_content = self.utils.create_multiline_string(text_content)
        self.fig_tools.add_text_box(
            ax=axis2plot,
            text=text_content,
            fontsize=fontsize,
            xpos=0.02,
            ypos=0.98,
            va="top",
        )

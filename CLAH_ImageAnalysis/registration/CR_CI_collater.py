import numpy as np
import pandas as pd
from scipy.stats import sem

from CLAH_ImageAnalysis.core import BaseClass as BC
from CLAH_ImageAnalysis.core import run_CLAH_script
from CLAH_ImageAnalysis.registration import CRwROI_enum


class CR_CI_collater(BC):
    def __init__(self, path: str | list, sess2process: list, forPres: bool) -> None:
        self.program_name = "CIC"
        self.class_type = "manager"
        BC.__init__(self, self.program_name, mode=self.class_type)

        self.static_class_var_init(path, sess2process, forPres)

    def process_order(self) -> None:
        """
        Process the order by executing the necessary steps in the correct order.

        This method performs the following steps:
        1. Extracts CI (Confidence Interval) data.
        2. Creates metadata columns.
        3. Calculates ratios.
        4. Collates the data.

        This method does not return any value.
        """
        self._extract_CI()
        self._create_metadata_columns()
        self._calc_Ratios()
        self._collate2all_data()

    def static_class_var_init(
        self, path: str | list, sess2process: list, forPres: bool
    ) -> None:
        """
        Initializes the static class variables for the CR_CI_collater class.

        Parameters:
            path (str): The folder path.
            sess2process (str): The session to process.

        Returns:
            None
        """
        BC.static_class_var_init(
            self,
            folder_path=path,
            file_of_interest=self.text_lib["selector"]["tags"]["CI"],
            selection_made=sess2process,
        )
        self.forPres = forPres
        self.CRRkey = self.enum2dict(CRwROI_enum.Txt)
        self.all_data = pd.DataFrame()

        self.fig_tools = self.utils.fig_tools
        self.colors4plot = [
            self.color_dict["blue"],
            self.color_dict["red"],
            self.color_dict["green"],
        ]

        self.title_fs = 18
        self.title_fs_small = 12
        self.label_fs = 18
        self.axis_fs = 16
        self.fs_medium = 14

        self.groupFig_path = f"{self.dayPath}/~GroupData"

    def _extract_CI(self) -> None:
        """
        Extracts Cluster Info from the latest saved file.

        This method extracts the Cluster Info from the latest saved file, which is a JSON file containing the cluster information.
        The extracted data is stored in the `df2input` attribute of the class.

        Parameters:
            None

        Returns:
            None
        """
        self.print_wFrm("Extracting Cluster Info")
        latest_CI = self.findLatest([self.CRRkey["C_INFO_SAVE"], self.file_tag["JSON"]])
        self.df2input = pd.read_json(latest_CI, typ="series").to_frame().transpose()

    def _create_metadata_columns(self) -> None:
        """
        Adds metadata columns to the dataframe.

        This method adds the following columns to the dataframe:
        - 'subject_id': The ID of the subject.
        - 'aged': Indicates whether the subject is aged or not.
        - 'Group': The group to which the subject belongs ('WT' or 'EXP').
        """
        self.print_wFrm(f"Adding metadata for {self.ID}")
        self.df2input["subject_id"] = self.ID

        self.optoCheck = False

        if "Ag_vs_nonAg" in self.dayPath and "NIAMossy" not in self.dayPath:
            if self.ID.startswith("aDk"):
                self.df2input["Group"] = "AGED"
            elif self.ID.startswith("Dk"):
                self.df2input["Group"] = "DK"
            elif self.ID.startswith("Wt"):
                self.df2input["Group"] = "WT"
        elif "Ag_vs_nonAg" in self.dayPath and "NIAMossy" in self.dayPath:
            if any(substring in self.ID for substring in ["C"]):
                self.df2input["Group"] = "NONAGED"
            else:
                self.df2input["Group"] = "AGED"
        elif "AD" in self.dayPath:
            if any(substring in self.ID for substring in ["21", "22", "23", "24"]):
                self.df2input["Group"] = "AD"
            elif any(substring in self.ID for substring in ["25", "26"]):
                self.df2input["Group"] = "CTL"
        elif "OPTO" in self.dayPath or "eOPN3" in self.dayPath:
            self.optoCheck = True
            self.df2input["Group"] = "OPTO"

    def _calc_Ratios(self) -> None:
        """
        Calculates ratios for each column in the input dataframe.

        Returns:
            None
        """
        self.print_wFrm("Calculating ratios")

        cols = ["subject_id", "Group"] + [
            col for col in self.df2input.columns if col not in ["subject_id", "Group"]
        ]
        self.df2input = self.df2input[cols]
        self.df2input.set_index("subject_id", inplace=True)

        col2calcRatio = [
            col
            for col in self.df2input.columns
            if col
            not in [
                "subject_id",
                "Group",
                "accepted_clusters",
            ]
            and not col.startswith("trackedROIs_")
            and not col.startswith("nontracked_")
            and not col.startswith("underlying_")
            and not col.startswith("tracked_")
            and not col.startswith("discarded")
        ]

        col2calcTrackedRatio = [
            col for col in self.df2input.columns if col.startswith("trackedROIs_")
        ]

        col4total = [
            col for col in self.df2input.columns if col.startswith("underlying_")
        ]
        col4tracked = [
            col for col in self.df2input.columns if col.startswith("tracked_")
        ] + ["accepted_clusters"]

        for col in col2calcRatio:
            self.df2input[col + "_UC_ratio"] = (
                self.df2input[col] / self.df2input["underlying_cells"]
            )
            self.df2input[col + "_AC_ratio"] = (
                self.df2input[col] / self.df2input["accepted_clusters"]
            )

        cellTypes = ["CueCells", "TunedCells"]
        for col in col2calcTrackedRatio:
            for ct in cellTypes:
                if ct in col:
                    base_col = col.replace(f"_{ct}", "")
                    self.df2input[col + "_ratio"] = (
                        self.df2input[col] / self.df2input[base_col]
                    )

        col4total.sort()
        col4tracked.sort()

        for tkd, tot in zip(col4tracked, col4total):
            name = f"{tkd}/{tot}"
            self.df2input[name] = self.df2input[tkd] / self.df2input[tot]

    def _collate2all_data(self) -> None:
        """
        Collates the data from df2input into the all_data table.

        This method appends the data from the df2input DataFrame to the all_data table.
        It uses the pd.concat() function to concatenate the two DataFrames.
        """
        self.print_wFrm("Collating to all_data table")
        self.all_data = pd.concat([self.all_data, self.df2input])
        self.print_done_small_proc(new_line=True)

    def post_proc_run(self) -> None:
        """
        Perform post-session-by-session processing.

        This method is responsible for performing post-processing tasks after session-by-session processing is complete.
        It changes the current working directory to the ~GroupData folder within the dayPath directory, creates a means table,
        exports tables, and prints a message indicating that the processing is done.
        """
        print("Post-session-by-session processing:")
        self.folder_tools.chdir_check_folder(f"{self.dayPath}/~GroupData", create=True)
        self._create_means_table()
        self._create_table4plotting()
        if not self.optoCheck:
            self._plot_meanTrackedCellRatios_bySession()
            self._plot_trackedProbabilities()
            self._plot_trackedRatios_over_underlying()
        self._export_tables()
        self.print_done_small_proc(new_line=True)

    def _create_means_table(self) -> None:
        """
        Creates a table of means and standard deviations for different groups.

        This method extracts groups from the `all_data` DataFrame based on the "Group" and "aged" columns.
        It calculates the mean and standard deviation for each group using numeric columns.
        The results are stored in the `means_table` attribute.
        """
        self.print_wFrm("Extracting groups")
        self.groupsCat = np.unique(self.all_data["Group"])

        desired_order = ["CTL", "DK", "NONAGED", "AD", "AGED"]

        # Filter the desired order to include only the groups present in groupsCat
        filtered_order = [group for group in desired_order if group in self.groupsCat]

        # Sort groupsCat based on the filtered order
        self.groupsCat = sorted(
            self.groupsCat,
            key=lambda x: filtered_order.index(x) if x in filtered_order else np.nan,
        )

        gr = self.groupsCat
        cellTypes = ["CueCells", "TunedCells"]
        self.all_data_byGroup = {
            g: self.all_data[self.all_data["Group"] == g] for g in self.groupsCat
        }

        self.trackdDict = {g: {ct: None for ct in cellTypes} for g in gr}
        for g in gr:
            self.trackdDict[g]["TOTAL"] = None
            for ct in cellTypes:
                self.trackdDict[g][ct] = None

        # Initialize an empty DataFrame to store the results
        results = pd.DataFrame()

        self.trackdWK = "trackedROIs_weekly_S2vsS3"

        self.print_wFrm("Calculating means & standard deviations")
        # Calculate the mean and standard deviation for each group
        # use np.number to select only numeric columns
        for group_name, group_data in self.all_data_byGroup.items():
            numeric_cols = group_data.select_dtypes(include=[np.number]).columns
            mean = group_data[numeric_cols].mean()
            mean.name = group_name + "_mean"
            sem_values = group_data[numeric_cols].apply(
                lambda x: sem(x, nan_policy="omit")
            )
            sem_values.name = group_name + "_sem"
            results = pd.concat([results, mean, sem_values], axis=1)

            total_trkd = group_data[self.trackdWK].sum()
            self.trackdDict[group_name]["TOTAL"] = total_trkd
            for ct in cellTypes:
                total_trkdCT = group_data[f"{self.trackdWK}_{ct}"].sum()
                self.trackdDict[group_name][ct] = total_trkdCT / total_trkd

        # # Transpose the results DataFrame
        # results = results.transpose()
        self.print_wFrm("Creating means_table")
        self.means_table = results

    def _create_table4plotting(self) -> None:
        """
        Creates a table for plotting.
        """

        def containsWexclude(entry: str, include: list, exclude: list) -> bool:
            """
            Checks if an entry contains all of the include strings and none of the exclude strings.
            """
            return all(sub in entry for sub in include) and not any(
                sub in entry for sub in exclude
            )

        cc_strings = [["CueCells", "_ratio"], ["QC", "trackedROIs", "TC_ratio"]]
        tc_strings = [["TunedCells", "_ratio"], ["QC", "trackedROIs"]]

        CC_means_rows = [
            row
            for row in self.means_table.index
            if containsWexclude(row, cc_strings[0], cc_strings[1])
        ]

        CC_all_cols = [
            col
            for col in self.all_data_byGroup[self.groupsCat[0]].columns
            if containsWexclude(col, cc_strings[0], cc_strings[1])
        ]

        TC_means_rows = [
            row
            for row in self.means_table.index
            if containsWexclude(row, tc_strings[0], tc_strings[1])
        ]

        TC_all_cols = [
            col
            for col in self.all_data_byGroup[self.groupsCat[0]].columns
            if containsWexclude(col, tc_strings[0], tc_strings[1])
        ]

        trkd_ratio_rows = [
            row for row in self.means_table.index if containsWexclude(row, ["/"], [])
        ]

        trkd_total_all_cols = (
            [
                col
                for col in self.all_data_byGroup[self.groupsCat[0]].columns
                if containsWexclude(col, ["tracked_"], ["/"])
            ]
            + ["accepted_clusters"]
            + ["underlying_cells"]
        )

        self.dict2plot = {
            "CC/ALL": self.means_table.loc[CC_means_rows],
            # "CC/TC": self.means_table.loc[CC_TC_rows],
            "TC/ALL": self.means_table.loc[TC_means_rows],
        }

        self.dict2plot_ind = {
            "CC/ALL": {
                gr: self.all_data_byGroup[gr][CC_all_cols] for gr in self.groupsCat
            },
            "TC/ALL": {
                gr: self.all_data_byGroup[gr][TC_all_cols] for gr in self.groupsCat
            },
        }

        self.trkdRatio2plot = self.means_table.loc[trkd_ratio_rows]
        self.trkdTotal_ind = {
            gr: self.all_data_byGroup[gr][trkd_total_all_cols] for gr in self.groupsCat
        }

    def _plot_meanTrackedCellRatios_bySession(self) -> None:
        """
        Plots the mean tracked cell ratios by session.
        """
        totalTypes = ["AC", "UC"]
        width = 0.3
        if len(self.groupsCat) == 2:
            width = 0.2
        elif len(self.groupsCat) == 3:
            width = 0.1

        colors2use = self.colors4plot

        fig, ax = self.fig_tools.create_plt_subplots(nrows=len(totalTypes))
        ax = ax.flatten()

        n_sessions = int(len(self.dict2plot["CC/ALL"].index) / len(totalTypes))
        index = np.arange(1, n_sessions + 1)

        ratios = self.dict2plot.keys()

        pVals = {
            t: {
                r: {f"S{idx + 1}": None for idx in range(n_sessions)}
                for r in ["Cue", "Tuned"]
            }
            for t in ["accepted_clusters", "underlying_cells"]
        }
        mean_semVals = {
            t: {
                gr: {
                    f"S{idx + 1}": {"MEAN": np.nan, "SEM": np.nan}
                    for idx in range(n_sessions)
                }
                for gr in self.groupsCat
            }
            for t in totalTypes
        }
        for t_idx, t in enumerate(totalTypes):
            ax2plot = ax[t_idx]
            if t == "AC":
                total2use = "all tracked cells"
                total2usekey = "accepted_clusters"
            elif t == "UC":
                total2use = "all underlying cells (tracked + untracked)"
                total2usekey = "underlying_cells"

            for r_idx, (ratioType, table) in enumerate(self.dict2plot.items()):
                if ratioType == "CC/ALL":
                    pVal_rkey = "Cue"
                    hatch2use = "//"
                elif ratioType == "TC/ALL":
                    pVal_rkey = "Tuned"
                    hatch2use = None

                arr4sig = []
                index2use = [row for row in table.index if f"{t}_ratio" in row]
                filtered_table = table.loc[index2use]

                for i, group in enumerate(self.groupsCat):
                    ind_table = self.dict2plot_ind[ratioType][group]
                    col2use4ind = [
                        col for col in ind_table.columns if f"{t}_ratio" in col
                    ]
                    filt_ind_table = ind_table[col2use4ind].T
                    arr4sig.append(filt_ind_table.values)

                    meanVal = filtered_table[f"{group}_mean"]
                    semVal = filtered_table[f"{group}_sem"]
                    for s_idx, idxname in enumerate(index2use):
                        mean_semVals[t][group][f"S{s_idx + 1}"] = {
                            "MEAN": meanVal.loc[idxname],
                            "SEM": semVal.loc[idxname],
                        }

                    bar_pos = self.fig_tools.create_index4grouped_barplot(
                        n_bars=len(self.groupsCat) * len(ratios),
                        index=index,
                        width=width,
                        measureLoop_idx=r_idx,
                        num_groups=len(self.groupsCat),
                        gLoop_idx=i,
                    )

                    self.fig_tools.bar_plot(
                        ax=ax2plot,
                        X=bar_pos,
                        Y=meanVal,
                        width=width,
                        label=group,
                        yerr=semVal,
                        hatch=hatch2use,
                        color=colors2use[i],
                    )
                for s_idx, (arr0, arr1) in enumerate(zip(arr4sig[0], arr4sig[1])):
                    pVal = self.fig_tools.create_sig_2samp_annotate(
                        ax=ax2plot,
                        arr0=arr0,
                        arr1=arr1,
                        coords=(
                            index[s_idx] - width + (width * r_idx * len(ratios)),
                            None,
                        ),
                        fontsize=8,
                        return_Pval=True,
                    )
                    pVals[total2usekey][pVal_rkey][f"S{s_idx + 1}"] = pVal

            total_means = [
                self.means_table[f"{gr}_mean"][total2usekey] for gr in self.groupsCat
            ]
            total_sems = [
                self.means_table[f"{gr}_sem"][total2usekey] for gr in self.groupsCat
            ]

            # Create text for the top left corner
            textstr2print = ["Mean total cells"]
            textstr2print += [
                f"{gr}: {mean:.2f} ± {serr:.2f}"
                for gr, mean, serr in zip(self.groupsCat, total_means, total_sems)
            ]
            textstr2print = self.utils.create_multiline_string(textstr2print)

            # Add text box to the top left corner
            ax2plot.text(
                0.02,
                0.98,
                textstr2print,
                transform=ax2plot.transAxes,
                fontsize=8,
                verticalalignment="top",
                bbox=dict(
                    boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"
                ),
            )
            fc = [colors2use[idx] for idx in range(len(self.groupsCat))] + ["none"] * 2
            legend_handles = self.fig_tools.create_legend_patch_fLoop(
                facecolor=fc,
                label=[gr for gr in self.groupsCat] + ["Cue", "Tuned"],
                hatch=[None] * len(self.groupsCat) + ["//", None],
                edgecolor=[self.color_dict["black"]] * len(fc),
                alpha=[1.0] * len(self.groupsCat) + [None] * 2,
            )

            if t_idx == 0:
                ax2plot.legend(handles=legend_handles, loc="upper right")
            if t_idx == 1:
                ax2plot.set_xlabel("Session", fontsize=self.label_fs)
            ax2plot.set_ylabel("Ratio", fontsize=self.label_fs)
            if t == "AC":
                ax2plot.set_ylim(0, 0.5)
            elif t == "UC":
                ax2plot.set_ylim(0, 0.2)
            ax2plot.set_xticks(index)
            ax2plot.tick_params(axis="both", which="major", labelsize=self.axis_fs)
            ax2plot.set_title(
                f"Mean Ratio of tracked cells by cell type over {total2use}",
                fontsize=self.title_fs_small,
            )

        self.fig_tools.save_figure(
            plt_figure=fig,
            figure_save_path=self.groupFig_path,
            fig_name="ratioTrackedCells",
            forPres=self.forPres,
            pValDict=pVals,
            mean_semDict=mean_semVals,
        )

    def _plot_trackedProbabilities(self) -> None:
        """
        Plots the tracked probabilities.
        """
        fig, ax = self.fig_tools.create_plt_subplots()
        width = 0.2
        colors2use = self.colors4plot
        textstr = ["Cells Tracked"]
        xlabel = ["Cue", "Tuned"]

        mean_semVals = {
            gr: {ct: {"PROBABILITY": np.nan} for ct in ["CueCells", "TunedCells"]}
            for gr in self.groupsCat
        }
        for g_idx, group in enumerate(self.trackdDict.keys()):
            # x = g_idx + 1
            cc_tot = self.trackdDict[group]["CueCells"]
            tc_tot = self.trackdDict[group]["TunedCells"]
            tot = self.trackdDict[group]["TOTAL"]

            mean_semVals[group]["CueCells"] = {"PROBABILITY": cc_tot}
            mean_semVals[group]["TunedCells"] = {"PROBABILITY": tc_tot}

            cc_x = 1 + (g_idx - (len(self.trackdDict) - 1) / 2) * (width)
            tc_x = 2 + (g_idx - (len(self.trackdDict) - 1) / 2) * (width)

            self.fig_tools.bar_plot(
                ax,
                X=cc_x,
                Y=cc_tot,
                width=width,
                label="CueCells",
                color=colors2use[g_idx],
                hatch="//",
            )

            self.fig_tools.bar_plot(
                ax,
                X=tc_x,
                Y=tc_tot,
                width=width,
                label="TunedCells",
                color=colors2use[g_idx],
            )
            textstr.append(f"{group}: {tot}")

        textstr = self.utils.create_multiline_string(textstr)
        self.fig_tools.add_text_box(
            ax=ax, text=textstr, fontsize=self.fs_medium, va="top", xpos=0.02, ypos=0.98
        )

        ax.set_xticks(range(1, len(xlabel) + 1))
        ax.set_xticklabels(xlabel)

        ax.tick_params(axis="both", which="major", labelsize=self.axis_fs)

        ax.set_ylabel("Cell Type Probability", fontsize=self.label_fs)
        ax.set_ylim(0, 0.20)
        ax.set_title(
            "Probability of tracked cell types by group (only tracked cells found in Session 2 & 3)",
            fontsize=self.title_fs_small,
        )

        fc = [colors2use[g_idx] for g_idx in range(len(self.groupsCat))]
        legend_handles = self.fig_tools.create_legend_patch_fLoop(
            facecolor=fc,
            label=self.groupsCat,
            edgecolor=[self.color_dict["black"]] * len(fc),
        )
        ax.legend(handles=legend_handles, loc="upper center")

        self.fig_tools.save_figure(
            plt_figure=fig,
            figure_save_path=self.groupFig_path,
            fig_name="trackedProbabilities",
            forPres=self.forPres,
            mean_semDict=mean_semVals,
        )

    def _plot_trackedRatios_over_underlying(self) -> None:
        """
        Plots the tracked ratios over the underlying.
        """
        colors2use = self.colors4plot

        fig, ax = self.fig_tools.create_plt_subplots()
        width = 0.4

        ratios = [ind for ind in self.trkdRatio2plot.index]

        xlabel = []
        for i, r in enumerate(ratios):
            if r.startswith("accepted_clusters"):
                xlabel.append("TrkTot/UndTot")
            if r.startswith("tracked_tune"):
                xlabel.append("TrkTC/UndTC")
            if r.startswith("tracked_cue"):
                xlabel.append("TrkCC/UndCC")

        totals = {
            gr: {
                "underlying_cells": ["ALL", np.nan],
                "underlying_cue_cells": ["CUE", np.nan],
                "underlying_tuned_cells": ["TUNED", np.nan],
            }
            for gr in self.groupsCat
        }

        for group in self.groupsCat:
            for key in totals[group].keys():
                if key == "underlying_cells":
                    totals[group][key][-1] = self.trkdTotal_ind[group][key].sum()
                if key in ["underlying_cue_cells", "underlying_tuned_cells"]:
                    key2use = key.replace("underlying_", "")
                    trkd = self.trkdTotal_ind[group][f"tracked_{key2use}"].sum()
                    utrkd = self.trkdTotal_ind[group][f"nontracked_{key2use}"].sum()
                    totals[group][key][-1] = trkd + utrkd

        text2print = ["Total Underlying Cells"]
        for g_key, tots in totals.items():
            text2print.append(f"{g_key:<3}:")
            for _, tot in tots.items():
                text2print.append(" " * 2 + f"{tot[0]:<5}: {tot[-1]:5.0f}")
        text2print = self.utils.create_multiline_string(text2print)
        self.fig_tools.add_text_box(
            ax=ax, text=text2print, fontsize=10, va="top", xpos=0.02, ypos=0.98
        )

        mean_semVals = {
            r: {gr: {"MEAN": np.nan, "SEM": np.nan} for gr in self.groupsCat}
            for r in ratios
        }
        for r_idx, ratio in enumerate(ratios):
            for g_idx, group in enumerate(self.groupsCat):
                meanVal = self.trkdRatio2plot.loc[ratio, f"{group}_mean"]
                semVal = self.trkdRatio2plot.loc[ratio, f"{group}_sem"]

                mean_semVals[ratio][group] = {"MEAN": meanVal, "SEM": semVal}

                X = r_idx + 1 + (g_idx - (len(self.groupsCat) - 1) / 2) * (width)
                self.fig_tools.bar_plot(
                    ax=ax,
                    X=X,
                    Y=meanVal,
                    width=width,
                    label=group,
                    yerr=semVal,
                    color=colors2use[g_idx],
                )

        ax.set_xticks(range(1, len(xlabel) + 1))
        ax.set_xticklabels(xlabel)
        ax.set_ylabel("Ratio", fontsize=self.label_fs)
        ax.set_ylim(0, 1)

        ax.tick_params(axis="both", which="major", labelsize=self.fs_medium)

        fc = [colors2use[g_idx] for g_idx in range(len(self.groupsCat))]
        legend_handles = self.fig_tools.create_legend_patch_fLoop(
            facecolor=fc,
            label=self.groupsCat,
            edgecolor=[self.color_dict["black"]] * len(fc),
        )
        ax.legend(handles=legend_handles, loc="upper right")

        ax.set_title(
            "Ratio of Tracked Cells over Respective Underlying Population",
            fontsize=self.fs_medium,
        )
        self.fig_tools.save_figure(
            plt_figure=fig,
            figure_save_path=self.groupFig_path,
            fig_name="trackedOVERunderlyingRatios",
            forPres=self.forPres,
            mean_semDict=mean_semVals,
        )

    def _export_tables(self) -> None:
        """
        Export the `means_table` and `all_data` DataFrames to CSV files.

        This method exports the `means_table` and `all_data` DataFrames to separate CSV files.
        The `means_table` DataFrame is exported to a file named "ClusterInfo_means.csv",
        and the `all_data` DataFrame is exported to a file named "ClusterInfo_all.csv".
        """
        self.print_wFrm("Exporting all_data and means_table to CSV")
        self.means_table.to_csv("ClusterInfo_means.csv")
        self.all_data.to_csv("ClusterInfo_all.csv")
        # self.utils.saveNloadUtils.savedict2file(
        #     dict_to_save=self.trackdDict,
        #     dict_name="trackedProbabilities",
        #     filename="trackedProbabilities",
        #     filetype_to_save=self.file_tag["JSON"],
        # )

    @staticmethod
    def calculate_ratio(row: pd.Series) -> float:
        """
        Calculates the ratio of a given row to the underlying cells value.

        Parameters:
            row: The row to calculate the ratio for.

        Returns:
            The calculated ratio.
        """
        underlying_cells_value = row["underlying_cells"]
        return row / underlying_cells_value


if "__main__" == __name__:
    from CLAH_ImageAnalysis.registration import CRwROI_enum

    run_CLAH_script(CR_CI_collater, parser_enum=CRwROI_enum.Parser4CI)

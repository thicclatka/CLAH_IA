import itertools

import numpy as np
import pandas as pd
from scipy import stats

from CLAH_ImageAnalysis.core import BaseClass as BC
from CLAH_ImageAnalysis.unitAnalysis import CCF_Dep


class CCF_StatTesting(BC):
    def __init__(self, cueTrigSig: dict, evTimes: dict, cues: list, ind: list) -> None:
        """
        Initialize the CCF_StatTesting class.

        Parameters:
            cueTrigSig (dict): The cue-triggered signals.
            evTimes (dict): The event times.
            cues (list): The cues.
            ind (list): The indices.
        """
        self.program_name = "CCF"
        self.class_type = "utils"
        BC.__init__(self, self.program_name, mode=self.class_type)

        self.cueTrigSig = cueTrigSig
        self.evTimes = evTimes
        self.cues = cues
        self.ind = ind

    def _apply_StatTest_on_cueTrigSig(self) -> None:
        """
        Performs statistical tests on cue-triggered signals.

        This method calculates statistical tests on cue-triggered signals and stores the results in dictionaries.
        It initializes the `RankSumDict` and `TTestDict` dictionaries to store the results of rank sum and t-tests, respectively.
        It creates a `comp_cue_list` that includes the main cue and other cues from `evTimes` dictionary.
        It initializes the `cueAmp` dictionary to store the amplitude values of cue-triggered signals.
        It iterates over each cell and cue type to calculate the amplitude values and store them in the `cueAmp` dictionary.
        After filling the `cueAmp` dictionary, it performs statistical tests between the main cue and other cues.
        Finally, it plots the cue amplitude boxplots and exports the statistical test results to CSV files.
        """

        self.RankSumDict = {key.upper(): {} for key in self.cues}
        self.TTestDict = {key.upper(): {} for key in self.cues}

        array_to_init = np.full((len(self.cueTrigSig), 10), np.nan)

        # create comp_cue_list that has main_cue as first entry/ies
        comp_cue_list = self.cues + [
            key for key in self.evTimes.keys() if key not in self.cues
        ]
        # create statdicts to be filled after finding cueAmp
        for main_cue, comp_cue in itertools.product(self.cues, comp_cue_list):
            # main_cue = main_cue.upper()
            # comp_cue = comp_cue.upper()
            if main_cue != comp_cue and self.cues.index(main_cue) < comp_cue_list.index(
                comp_cue
            ):
                self.RankSumDict[main_cue][comp_cue] = array_to_init.copy()
                self.TTestDict[main_cue][comp_cue] = array_to_init.copy()
        # init cueAmp dict to be filled
        self.cueAmp = {
            cell: {key: [] for key in self.evTimes} for cell in self.cueTrigSig
        }
        for cell_idx, cell in enumerate(self.cueTrigSig):
            for ct_key in self.cueTrigSig[cell]:
                self.cueAmp[cell][ct_key] = np.full(
                    self.cueTrigSig[cell][ct_key].shape[-1], np.nan
                )
                for cue_idx in range(self.cueTrigSig[cell][ct_key].shape[-1]):
                    # self.cueAmp[cell][ct_key][cue_idx] = CCF_Dep.findMax_fromTrigSig(
                    #     TrigSig=self.cueTrigSig[cell][ct_key][:, cue_idx], ind=self.ind
                    # )
                    self.cueAmp[cell][ct_key][cue_idx] = CCF_Dep.findAUC_fromTrigSig(
                        TrigSig=self.cueTrigSig[cell][ct_key][:, cue_idx], ind=self.ind
                    )
                # if ct_key == "CUE1" or ct_key == "OMITBOTH":
                #     print(f"{cell}-{ct_key}")
                #     print(f"Mean: {np.nanmean(self.cueAmp[cell][ct_key])}")
                #     print(f"Median: {np.nanmedian(self.cueAmp[cell][ct_key])}")
                #     print(f"STD: {np.nanstd(self.cueAmp[cell][ct_key])}")
                #     print("")
        # with a filled cueAmp, do stat tests
        for main_cue in self.cues:
            main_cue = main_cue.upper()
            for cell_idx, cell in enumerate(self.cueAmp):
                for cue_key in self.cueAmp[cell]:
                    if cue_key != main_cue and self.cues.index(
                        main_cue
                    ) < comp_cue_list.index(cue_key):
                        self._perform_tests_and_update_dicts(
                            main_cue, cue_key, cell, cell_idx
                        )
        self._export_stats_csv(self.RankSumDict, "_results_rankSum")
        self._export_stats_csv(self.TTestDict, "_results_TTest")

    def _perform_tests_and_update_dicts(
        self, main_cue: str, comp_cue: str, cell: str, cell_idx: int
    ) -> None:
        """
        Perform statistical tests and update the RankSumDict and TTestDict dictionaries.

        Parameters:
            main_cue (str): The main cue for comparison.
            comp_cue (str): The comparison cue.
            cell (str): The cell identifier.
            cell_idx (int): The index of the cell.
        """

        u_stat, rs_p_val = stats.mannwhitneyu(
            self.cueAmp[cell][main_cue], self.cueAmp[cell][comp_cue]
        )
        t_stat, t_p_val = stats.ttest_ind(
            self.cueAmp[cell][main_cue], self.cueAmp[cell][comp_cue]
        )

        mean_ampMain = np.mean(self.cueAmp[cell][main_cue])
        mean_ampComp = np.mean(self.cueAmp[cell][comp_cue])

        med_ampMain = np.median(self.cueAmp[cell][main_cue])
        med_ampComp = np.median(self.cueAmp[cell][comp_cue])

        std_ampMain = np.std(self.cueAmp[cell][main_cue])
        std_ampComp = np.std(self.cueAmp[cell][comp_cue])

        direction = int(
            1
            if np.mean(self.cueAmp[cell][main_cue])
            > np.mean(self.cueAmp[cell][comp_cue])
            else -1
        )
        self.RankSumDict = self._fill_stat_dict(
            stat_dict=self.RankSumDict,
            test_stat=u_stat,
            p_val=rs_p_val,
            main_cue=main_cue,
            cell_idx=cell_idx,
            comp_cue=comp_cue,
            direction=direction,
            means=(mean_ampMain, mean_ampComp),
            meds=(med_ampMain, med_ampComp),
            stds=(std_ampMain, std_ampComp),
        )
        self.TTestDict = self._fill_stat_dict(
            stat_dict=self.TTestDict,
            test_stat=t_stat,
            p_val=t_p_val,
            main_cue=main_cue,
            cell_idx=cell_idx,
            comp_cue=comp_cue,
            direction=direction,
            means=(mean_ampMain, mean_ampComp),
            meds=(med_ampMain, med_ampComp),
            stds=(std_ampMain, std_ampComp),
        )

    def _fill_stat_dict(
        self,
        stat_dict: dict,
        test_stat: float,
        p_val: float,
        main_cue: str,
        cell_idx: int,
        comp_cue: str,
        direction: int,
        means: tuple,
        meds: tuple,
        stds: tuple,
    ) -> dict:
        """
        Fill the statistical dictionary with the given values.

        Parameters:
            stat_dict (dict): The statistical dictionary to be filled.
            test_stat (float): The test statistic value.
            p_val (float): The p-value.
            main_cue (str): The main cue.
            cell_idx (int): The cell index.
            comp_cue (str): The comparison cue.
            direction (int): The direction. 1 means main cue > comparison cue, -1 means main cue < comparison cue.
            means (tuple): The means of the main and comparison cues.
            meds (tuple): The medians of the main and comparison cues.
            stds (tuple): The standard deviations of the main and comparison cues.

        Returns:
            dict: The updated statistical dictionary.
        """
        sig = 1 if p_val < 0.05 else 0
        stat_dict[main_cue][comp_cue][cell_idx, :] = [
            test_stat,
            p_val,
            sig,
            direction,
            means[0],
            means[1],
            meds[0],
            meds[1],
            stds[0],
            stds[1],
        ]
        return stat_dict

    def _export_stats_csv(self, dict_to_use: dict, fname: str) -> None:
        """
        Export statistics to a CSV file.

        Parameters:
            dict_to_use (dict): The dictionary containing the statistics data.
            fname (str): The filename for the CSV file.
        """

        cells = list(self.cueTrigSig.keys())
        data = []
        self.stat_idx = 0
        self.p_idx = 1
        self.sig_idx = 2
        self.dir_idx = 3
        mean_main_idx = 4
        mean_comp_idx = 5
        med_main_idx = 6
        med_comp_idx = 7
        std_main_idx = 8
        std_comp_idx = 9
        for cue in dict_to_use:
            for test in dict_to_use[cue]:
                for idx, stat_p in enumerate(dict_to_use[cue][test]):
                    cell_idx = cells[idx]
                    data.append(
                        [
                            cell_idx,
                            cue,
                            test,
                            stat_p[self.stat_idx],  # stat value
                            stat_p[self.p_idx],  # p value
                            stat_p[self.sig_idx],  # 1 = sig; 0 = not sig
                            stat_p[self.dir_idx],  # dir
                            stat_p[mean_main_idx],  # mean main cue
                            stat_p[mean_comp_idx],  # mean comp cue
                            stat_p[med_main_idx],  # median main cue
                            stat_p[med_comp_idx],  # median comp cue
                            stat_p[std_main_idx],  # std main cue
                            stat_p[std_comp_idx],  # std comp cue
                        ]
                    )

        # Sig =1 if p < 0.05
        # Dir = 1 if mean cueAmp of CUE is greater than comparison CUE, otherwise -1
        df = pd.DataFrame(
            data,
            columns=[
                "Cell",
                "Cue",
                "Comparison",
                "Stat Value",
                "P Value",
                "Significance",
                "Dir",
                "Mean Main",
                "Mean Comp",
                "Median Main",
                "Median Comp",
                "STD Main",
                "STD Comp",
            ],
        )
        df.to_csv(f"{fname}.csv", index=False)

    def _return_statsNcueAmp(self) -> tuple[dict, dict, dict]:
        """
        Return the statistical test results and cue amplitudes.

        Returns:
            tuple: The statistical test results and cue amplitudes.
        """
        return self.RankSumDict, self.TTestDict, self.cueAmp

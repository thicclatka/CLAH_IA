import numpy as np

from CLAH_ImageAnalysis.core import BaseClass as BC
from CLAH_ImageAnalysis.core import run_CLAH_script
from CLAH_ImageAnalysis.registration import CRwROI_enum
from CLAH_ImageAnalysis.unitAnalysis import PCR_CCF_Plotter, UA_enum


class PostCR_CueCellFinder(BC):
    def __init__(
        self,
        path: str | list,
        sess2process: list,
        outlier_ts: int,
        sessFocus: int,
        forPres: bool,
        plotIndTrigSig: bool,
        concatCheck: bool,
    ) -> None:
        """
        Initializes the PostCR_CueCellFinder class.

        Parameters:
            path (str | list): The path to the directory containing the data.
            sess2process (list): The sessions to process.
            outlier_ts (int): The number of outlier time steps to remove.
            sessFocus (int): The session to focus on.
            forPres (bool): Whether to process for presentation.
            plotIndTrigSig (bool): Whether to plot individual trigger signals.
            concatCheck (bool): Set this to True if you are working with sessions that were motion corrected in a concatenated manner. THIS ONLY WORKS IF MS CONTAINS ONLY 2 SESSIONS AKA THE SPLIT OUTPUT OF 1 CONCATENATED SESSION. Default is False
        """

        self.program_name = "PCR_CFF"
        self.class_type = "manager"
        BC.__init__(self, self.program_name, mode=self.class_type)

        self.static_class_var_init(
            path,
            sess2process,
            outlier_ts,
            sessFocus,
            forPres,
            plotIndTrigSig,
            concatCheck,
        )

    def process_order(self) -> None:
        """
        Executes the processing steps in a specific order.

        This method loads the MSSnROICaT, initializes variables, organizes cluster indices,
        organizes trigger signals by cluster, calculates mean by cluster, plots trigger signals,
        collates two dictionaries by group, and exports the data.
        """

        self.load_MSSnROICaT()
        self.post_load_var_init()
        self.organize_cluster_idx()
        self.organize_TrigSigbyCluster()
        self.CalcMean_byCluster()
        if self.plotIndTrigSig:
            self.plot_TrigSig()
        if not self.optoCheck:
            self.find_SpatialShift()
        # if self.optoCheck:
        #     self.plot_OptoScatter()
        self.collate2byGroupDicts()
        self.export_data()

    def static_class_var_init(
        self,
        path: str | list,
        sess2process: list,
        outlier_ts: int,
        sessFocus: int,
        forPres: bool,
        plotIndTrigSig: bool,
        concatCheck: bool,
    ) -> None:
        """
        Initializes the static class variables for the PostCR_CueCellFinder class.

        Parameters:
            path (str | list): The path to the directory containing the data.
            sess2process (list): The sessions to process.
            outlier_ts (int): The number of outlier time steps to remove.
            sessFocus (int): The session to focus on.
            forPres (bool): Whether to process for presentation.
            plotIndTrigSig (bool): Whether to plot individual trigger signals.
            concatCheck (bool): Set this to True if you are working with sessions that were motion corrected in a concatenated manner. THIS ONLY WORKS IF MS CONTAINS ONLY 2 SESSIONS AKA THE SPLIT OUTPUT OF 1 CONCATENATED SESSION. Default is False
        Returns:
            None
        """
        BC.static_class_var_init(
            self,
            folder_path=path,
            file_of_interest=self.text_lib["selector"]["tags"]["MSS"],
            selection_made=sess2process,
        )
        self.outlier_ts = outlier_ts
        self.sessFocus = sessFocus
        self.forPres = forPres
        self.plotIndTrigSig = plotIndTrigSig
        self.concatCheck = concatCheck
        self.CRRkey = self.enum2dict(CRwROI_enum.Txt)
        self.CCFkey = self.enum2dict(UA_enum.CCF)
        self.PCRkey = self.enum2dict(UA_enum.PCR_CFF)
        self.ind = self.CCFkey["IND"]
        self.ind_len = self.ind[1] - self.ind[0]
        self.cc_type = ["CUE", "NONCUE"]

        # if OPTO is in directory name, set optoCheck to True
        self.optoCheck = False
        if "OPTO" in self.dayPath or "eOPN3" in self.dayPath:
            self.optoCheck = True
            self.groups = ["OPTO"]
            self.OptoAmpArrAllSubj = {}

        self.eOPN3Check = False
        if "eOPN3" in self.dayPath:
            self.optoCheck = True
            self.eOPN3Check = True
            self.groups = ["eOPN3"]

        self.ADCheck = False
        if "AD" in self.dayPath:
            self.ADCheck = True
            self.groups = ["CTL", "AD"]

        self.AGEDCheck = False
        if "Ag_vs_nonAg" in self.dayPath and "NIAMossy" not in self.dayPath:
            self.AGEDCheck = True
            self.groups = ["WT", "DK", "AGED"]

        self.KETCheck = False
        if "KETA" in self.dayPath:
            self.KETCheck = True
            self.groups = ["KETA"]

        self.miniscopeCheck = False
        if "miniscope" in self.dayPath:
            self.miniscopeCheck = True
            self.groups = ["MINISCOPE"]

        self.AGEDMossyCheck = False
        if "Ag_vs_nonAg" in self.dayPath and "NIAMossy" in self.dayPath:
            self.AGEDMossyCheck = True
            self.groups = ["AGED", "NONAGED"]

        self.ATC_keys = ["ALL", "TC", "CC"]

        self.numSessByID = []

        # TrigSig
        self.TrigSig_byGroup = {group: {} for group in self.groups}
        self.meanTS_byGroup = {
            cc: {group: {} for group in self.groups} for cc in self.cc_type
        }
        # MaxVal from TrigSig
        self.maxVal_byGroup = {group: {} for group in self.groups}
        self.meanMV_byGroup = {
            cc: {group: {} for group in self.groups} for cc in self.cc_type
        }

        self.posRates_byGroup = {group: {} for group in self.groups}
        self.posRates_byGroup_CC = {group: {} for group in self.groups}

        self.eventRate_byGroup = {group: {} for group in self.groups}

        self.cellTotals_byGroup = {group: {} for group in self.groups}

        self.IPSpk_byGroup = {group: {} for group in self.groups}
        self.IPSec_byGroup = {group: {} for group in self.groups}

    def forLoop_var_init(self, sess_idx: int, sess_num: int) -> None:
        """
        Initializes variables for the for loop in the PostCR_CueCellFinder class.

        Parameters:
            sess_idx (int): The index of the current session.
            sess_num (int): The total number of sessions.
        """

        super().forLoop_var_init(sess_idx, sess_num)
        self.multSessSegStruc = None
        self.RC_results = None
        self.subj_sessions = None
        self.numSess = None

        if (
            not self.optoCheck
            and not self.ADCheck
            and not self.KETCheck
            and not self.miniscopeCheck
            and not self.AGEDMossyCheck
        ):
            if self.ID.startswith("aDk"):
                self.group = "AGED"
            elif "WT" in self.ID.upper():
                self.group = "WT"
            else:
                self.group = "DK"
        elif self.ADCheck:
            if any(substring in self.ID for substring in ["21", "22", "23", "24"]):
                self.group = "AD"
            elif any(substring in self.ID for substring in ["25", "26"]):
                self.group = "CTL"
        elif self.AGEDMossyCheck:
            if any(substring in self.ID for substring in ["C"]):
                self.group = "NONAGED"
            else:
                self.group = "AGED"
        else:
            self.group = self.groups[0]

    def load_MSSnROICaT(self) -> None:
        """
        Loads multSessSegStruc and Results from ROICaT for the subject.

        This method loads the multSessSegStruc and Results files from ROICaT for the subject.
        It finds the latest versions of the files using the `findLatest` method and loads them using the `load_file` method.
        It also retrieves the sessions for the subject and stores them in `subj_sessions`.
        """

        self.rprint(
            f"Loading multSessSegStruc & Results from ROICaT for Subject {self.ID}:"
        )
        lastest_MSS = self.findLatest([self.file_tag["MSS"], self.file_tag["PKL"]])
        latest_RRC = self.findLatest(
            [self.CRRkey["RESULTS_SAVE"], self.file_tag["PKL"]]
        )

        self.multSessSegStruc = self.load_file(lastest_MSS)
        self.print_wFrm(f"Loaded: {lastest_MSS}")

        if not self.concatCheck:
            self.RC_results = self.load_file(latest_RRC)
            self.print_wFrm(f"Loaded: {latest_RRC}\n")

        # get the sessions for the subject
        self.subj_sessions = list(self.multSessSegStruc.keys())
        self.numSess = len(self.subj_sessions)

        self.numSessByID.append(self.numSess)

        if self.sessFocus is not None:
            if self.numSess < self.sessFocus:
                self.numSess = self.numSess
            else:
                self.numSess = self.sessFocus

    def post_load_var_init(self) -> None:
        """
        Initializes variables after loading data.

        This method reorganizes data into lists by session and extracts relevant information from the multSessSegStruc dictionary.
        """

        print("Reorganizing Data into lists by session", end="", flush=True)
        # find refLapType per session
        self.refLapType = []
        self.lapTypeName = []
        self.ATS = []
        self.CC_IDX = []
        self.TC_IDX = []
        self.TOTAL_CELLS = []
        self.cellTotals = []
        self.evTimes = []
        self.C_Temporal = []
        self.OIDX = []
        self.OAMP = []
        self.OSTD = []
        # self.pksBySess = [[] for _ in range(self.numSess)]
        self.eventRateBySess = {ATC: [] for ATC in self.ATC_keys}
        self.posRates_bySess = {}
        self.posRates_bySess_TC = {}
        self.posRates_bySess_CC = {}
        self.InfoPerSpk = {ATC: [] for ATC in self.ATC_keys}
        self.InfoPerSec = {ATC: [] for ATC in self.ATC_keys}
        for idx, sess in enumerate(self.multSessSegStruc):
            curr_MSS = self.multSessSegStruc[sess]
            curr_CSS = curr_MSS[self.dict_name["CSS"]]
            curr_CCF = curr_MSS[self.dict_name["CCF"]]
            curr_lapCue_lap = curr_CSS["lapCue"]["lap"]

            # laptypes
            self.refLapType.append(curr_lapCue_lap["refLapType"])
            self.lapTypeName.append(curr_lapCue_lap["lapTypeName"])

            # TrigSig & evTimes
            self.ATS.append(curr_CCF["allTrigSig"])
            self.evTimes.append(curr_CCF["evTimes"])

            # segDict C Temp data
            self.C_Temporal.append(curr_MSS["C_Temporal"])

            # Cue Cell index
            self.CC_IDX.append(curr_CCF["CueCellTable"]["CUE_IDX"].astype(int))
            self.TOTAL_CELLS.append(curr_CCF["CueCellTable"]["TOTAL"])

            # Opto
            self.OIDX.append(curr_CCF["OptoIdx"])
            self.OAMP.append(curr_CCF["OptoAmp"])
            self.OSTD.append(curr_CCF["OptoSTD"])

            # obtain PosRatesRef and PosRatesRef_PC
            curr_refLap = f"lapType{self.refLapType[idx] + 1}"
            self.TC_IDX.append(
                np.where(curr_CSS["PCLappedSess"][curr_refLap]["Shuff"]["isPC"] == 1)[0]
            )

            curr_sessTime = self.C_Temporal[idx].shape[-1]
            pks2organize = curr_CSS["pks"]
            curr_eventRate = []
            for pk_vals in pks2organize.values():
                pks = pk_vals["pks"]
                # self.pksBySess[idx].append(pks)
                curr_eventRate.append(pks.shape[0] / curr_sessTime)

            curr_eventRate = np.array(curr_eventRate)
            self.eventRateBySess["ALL"].append(curr_eventRate)
            self.eventRateBySess["TC"].append(curr_eventRate[self.TC_IDX[idx]])
            self.eventRateBySess["CC"].append(curr_eventRate[self.CC_IDX[idx]])

            curr_IPSpk = curr_CSS["PCLappedSess"][curr_refLap]["InfoPerSpk"]
            self.InfoPerSpk["ALL"].append(curr_IPSpk)
            self.InfoPerSpk["TC"].append(curr_IPSpk[self.TC_IDX[idx]])
            self.InfoPerSpk["CC"].append(curr_IPSpk[self.CC_IDX[idx]])

            curr_IPSec = curr_CSS["PCLappedSess"][curr_refLap]["InfoPerSec"]
            self.InfoPerSec["ALL"].append(curr_IPSec)
            self.InfoPerSec["TC"].append(curr_IPSec[self.TC_IDX[idx]])
            self.InfoPerSec["CC"].append(curr_IPSec[self.CC_IDX[idx]])

            num_CC = len(self.CC_IDX[idx])
            num_TC = len(self.TC_IDX[idx])
            num_TOT = self.TOTAL_CELLS[idx]

            self.cellTotals.append(
                {
                    "CC/TC": num_CC / num_TC if num_TC != 0 else np.nan,
                    # # "CC/TOT": num_CC / num_TOT if num_TOT != 0 else np.nan,
                    "TC/TOT": num_TC / num_TOT if num_TOT != 0 else np.nan,
                    "ALL": num_TOT,
                    "CC": num_CC,
                    "TC": num_TC,
                }
            )

            for lt_idx, lt in enumerate(self.lapTypeName[idx]):
                lapType = f"lapType{lt_idx + 1}"
                if lt not in self.posRates_bySess.keys():
                    self.posRates_bySess[lt] = []
                if lt not in self.posRates_bySess_TC.keys():
                    self.posRates_bySess_TC[lt] = []
                if lt not in self.posRates_bySess_CC.keys():
                    self.posRates_bySess_CC[lt] = []
                curr_lt_posRate = curr_CSS["PCLappedSess"][lapType]["posRates"]
                self.posRates_bySess[lt].append(curr_lt_posRate)
                self.posRates_bySess_TC[lt].append(curr_lt_posRate[self.TC_IDX[idx]])
                self.posRates_bySess_CC[lt].append(curr_lt_posRate[self.CC_IDX[idx]])

        self.labelBySess = []
        if self.RC_results is not None:
            for labels in self.RC_results["clusters"]["labels_bySession"]:
                self.labelBySess.append(labels)
        else:
            self.labelBySess = [
                list(range(self.TOTAL_CELLS[idx])) for idx in range(self.numSess)
            ]

        # find untracked cells that are Cue Cells and assign them a unique cluster ID
        for sess, label in enumerate(self.labelBySess):
            for i, cell in enumerate(label):
                if cell == -1 and i in self.CC_IDX[sess]:
                    self.labelBySess[sess][i] = (sess * 1) + (i * 100)

        self.print_done_small_proc(new_line=True)

    def organize_cluster_idx(self) -> None:
        """
        Organizes cluster indices across sessions.

        This method organizes the cluster indices by session and cell number. It creates a list of tuples, where each tuple
        contains the session number, cell number, and cluster ID. If the cluster ID is -1, it is replaced with NaN.

        The method also applies a filter to the cluster IDs based on the `isPC` attribute. It creates a new list of tuples
        with filtered cluster IDs.

        Finally, it finds the cluster indices for both the original and filtered lists and stores them in the `cluster_idx`
        and `cluster_idx_postCCfilter` attributes respectively. It also creates a list of cluster keys.
        """

        print("Organizing cluster indices across sessions:")
        self.clusterBySessByCellNum = [
            (sess, cell, int(cid) if cid != -1 else np.nan)
            for sess, cluster_id in enumerate(self.labelBySess)
            for cell, cid in enumerate(cluster_id)
        ]

        self.clusterBySessByCellNum_postCCfilter = self._applyFilter2clusterIDs(
            self.clusterBySessByCellNum, filter_arr_diff_size=self.CC_IDX
        )

        self.cluster_idx = self._find_cluster_idx(
            self.clusterBySessByCellNum, self.labelBySess
        )
        self.cluster_idx_postCCfilter = self._find_cluster_idx(
            self.clusterBySessByCellNum_postCCfilter, self.labelBySess
        )

        self.cluster_keys = list(self.cluster_idx.keys())
        self.print_wFrm(f"Found {len(self.cluster_keys)} clusters in total\n")

    @staticmethod
    def _applyFilter2clusterIDs(
        cluster_tuple: list[tuple],
        filter_arr_same_size: list = [],
        filter_arr_diff_size: list = [],
    ) -> list:
        """
        Apply filters to cluster IDs based on the provided filter arrays.

        Parameters:
            cluster_tuple (list): List of tuples containing session, cell, and cluster ID.
            filter_arr_same_size (list, optional): 2D array representing the filter for same-sized clusters. Defaults to [].
            filter_arr_diff_size (list, optional): 2D array representing the filter for different-sized clusters. Defaults to [].

        Returns:
            list: Filtered list of tuples containing session, cell, and cluster ID.
        """

        filtered_cluster = []
        for ctuple in cluster_tuple:
            sess, cell, cid = ctuple
            if filter_arr_same_size and filter_arr_same_size[sess][cell] == 0:
                cid = np.nan
            elif (
                filter_arr_same_size
                and filter_arr_same_size[sess][cell] == 1
                and np.isnan(cid)
            ):
                cid = (sess * 1) + (cell * 100)
            elif filter_arr_diff_size and cell not in list(filter_arr_diff_size[sess]):
                cid = np.nan
            filtered_cluster.append((sess, cell, cid))
        return filtered_cluster

    @staticmethod
    def _find_cluster_idx(cluster_ids: list[tuple], labelBySess: list) -> dict:
        """
        Find the indices of clusters in each session.

        Parameters:
            cluster_ids (list): A list of tuples containing cluster information.
            labelBySess (list): A list of session labels.

        Returns:
            dict: A dictionary where the keys are cluster names and the values are tuples of indices.
        """

        cluster_idx = {}
        for sess, cell, id in cluster_ids:
            if not np.isnan(id):
                key = f"clst{id}"
                if key not in cluster_idx:
                    cluster_idx[key] = [np.nan] * len(labelBySess)
                cluster_idx[key][int(sess)] = cell

        return cluster_idx

    def organize_TrigSigbyCluster(self) -> None:
        """
        Organizes TrigSig by Cluster.

        This method initializes empty dictionaries to fill with TrigSig data organized by cluster and cue.
        It appends TrigSig data to the dictionaries based on the cluster and session.
        It also ensures that all arrays are the same shape for later exporting.
        """

        print("Organizing TrigSig by Cluster:")
        # get cue list from keys of evTimes (can use first session)
        self.print_wFrm("Initializing empty dictionaries to fill")
        self.cue_list = self.evTimes[0].keys()

        self.TrigSig_byCluster = {
            cluster_id: {cue: [] for cue in self.cue_list}
            for cluster_id in self.cluster_keys
        }
        self.CueCell_byTrigSig_byCluster = {
            cluster_id: [0] * self.numSess for cluster_id in self.cluster_keys
        }

        self.print_wFrm("Appending TrigSig by Cluster", end="", flush=True)
        for cluster, cell_idx in self.cluster_idx.items():
            for sess, cell in enumerate(cell_idx):
                self._append2TrigSig_byCluster(cluster, cell, sess)
                self._organize_CueCell_byTrigSig_byCluster(cluster, sess)
        self.print_done_small_proc(new_line=False)

        self.print_wFrm(
            "Ensuring all arrays are the same shape for later exporting",
            end="",
            flush=True,
        )
        # ensure all arrays are the same shape
        # this is for later exporting to MAT
        self.TrigSig_byCluster = self._ensure_each_entry_same_shape(
            dict2pad=self.TrigSig_byCluster,
            main_fl_keys=self.cluster_keys,
            keylist2pad=self.cue_list,
        )
        self.print_done_small_proc()

    def _append2TrigSig_byCluster(self, cluster: str, cell: int, sess: int) -> None:
        """
        Append TrigSig to TrigSig_byCluster accordingly.

        Parameters:
            cluster (str): The cluster index.
            cell (int): The cell index.
            sess (int): The session index.
        """

        # append TrigSig to TrigSig_byCluster accordingly
        for cue in self.cue_list:
            if np.isnan(cell):
                self.TrigSig_byCluster[cluster][cue].append(np.nan)
            else:
                cell_num = f"Cell_{cell}"
                self.TrigSig_byCluster[cluster][cue].append(
                    self.ATS[sess][cell_num][cue]
                )

        # turn sole NaNs into arrays of NaNs
        for cluster, cue, i, item in [
            (cluster, cue, i, item)
            for cluster, cues in self.TrigSig_byCluster.items()
            for cue, items in cues.items()
            for i, item in enumerate(items)
        ]:
            if not isinstance(item, np.ndarray) and np.isnan(item):
                # Get the number of trials from the first non-nan item in the list
                num_trials = next(
                    (
                        x.shape[1]
                        for x in self.TrigSig_byCluster[cluster][cue]
                        if isinstance(x, np.ndarray) and not np.all(np.isnan(x))
                    ),
                    1,
                )

                self.TrigSig_byCluster[cluster][cue][i] = np.full(
                    (self.ind_len, num_trials), np.nan
                )

    def _ensure_each_entry_same_shape(
        self, dict2pad: dict, main_fl_keys: list, keylist2pad: object | list
    ) -> dict:
        """
        Ensures that each entry in the given dictionary has the same shape by padding arrays with NaN values.

        Parameters:
            dict2pad (dict): The dictionary containing arrays to be padded.
            main_fl_keys (list): The list of keys to iterate over.
            keylist2pad (obj | list): The list of subkeys whose respective arrays need to be padded.

        Returns:
            dict: The dictionary with padded arrays.
        """

        for key in main_fl_keys:
            for key2pad in keylist2pad:
                if isinstance(dict2pad[key][key2pad][0], np.ndarray):
                    max_shape = max(arr.shape for arr in dict2pad[key][key2pad])
                    padded_arrays = [
                        np.pad(
                            arr,
                            [(0, max_shape[i] - arr.shape[i]) for i in range(arr.ndim)],
                            constant_values=np.nan,
                        )
                        for arr in dict2pad[key][key2pad]
                    ]
                    dict2pad[key][key2pad] = padded_arrays
                elif isinstance(dict2pad[key][key2pad][0], list):
                    for idx, arrs in enumerate(dict2pad[key][key2pad]):
                        max_shape = max(arr.shape for arr in arrs)
                        padded_arrays = [
                            np.pad(
                                arr,
                                [
                                    (0, max_shape[i] - arr.shape[i])
                                    for i in range(arr.ndim)
                                ],
                                constant_values=np.nan,
                            )
                            for arr in arrs
                        ]
                        dict2pad[key][key2pad][idx] = padded_arrays
        return dict2pad

    def _organize_CueCell_byTrigSig_byCluster(self, cluster: str, sess: int) -> None:
        """
        Organizes cue cells by trigger signal and cluster.

        Parameters:
            cluster (str): The cluster index.
            sess (int): The session index.
        """

        if cluster in self.cluster_idx_postCCfilter.keys():
            if not np.isnan(self.cluster_idx_postCCfilter[cluster][sess]):
                self.CueCell_byTrigSig_byCluster[cluster][sess] = 1

    def CalcMean_byCluster(self) -> None:
        """
        Calculates the mean trigger signal across clusters by Cue vs Non-Cue Classification.
        """

        print("Finding Mean Trig Sig across clusters by Cue vs Non-Cue Classification:")
        meanTS_byCC_byCueType = {
            CC: {cue: [[] for _ in range(self.numSess)] for cue in self.cue_list}
            for CC in self.cc_type
        }
        self.print_wFrm("Calculating means")
        for cluster, cueTypes in self.TrigSig_byCluster.items():
            for cue, trigSig in cueTypes.items():
                for sess, tsigarr in enumerate(trigSig):
                    if isinstance(tsigarr, np.ndarray) and not np.isnan(tsigarr).all():
                        # convert infs to nans
                        tsigarr = np.nan_to_num(tsigarr, nan=np.nan)
                        mean_result = np.nanmean(tsigarr, axis=1)
                        if self.CueCell_byTrigSig_byCluster[cluster][sess]:
                            meanTS_byCC_byCueType["CUE"][cue][sess].append(mean_result)
                        else:
                            meanTS_byCC_byCueType["NONCUE"][cue][sess].append(
                                mean_result
                            )

        self.print_wFrm(
            "Reorienting arrays & ensuring all arrays are the same shape for later exporting"
        )
        # turn into arrays & tranpose
        for CC, cueTypes in meanTS_byCC_byCueType.items():
            for cue, mean_arrs in cueTypes.items():
                for sess, marr in enumerate(mean_arrs):
                    meanTS_byCC_byCueType[CC][cue][sess] = np.array(marr).T

        # ensure all arrays are the same shape
        # this is for later exporting to MAT
        self.meanTS_byCC_byCueType = self._ensure_each_entry_same_shape(
            meanTS_byCC_byCueType, main_fl_keys=self.cc_type, keylist2pad=self.cue_list
        )
        self.print_done_small_proc()

    def plot_TrigSig(self) -> None:
        """
        Plots the trigger signal for cue cells.

        This method initializes a PCR_CCF_Plotter object and uses it to plot the trigger signal
        for cue cells. It first plots the 1PDF of the trigger signal for each cluster, and then
        plots the mean trigger signal for each cue type.
        """

        self.PCRCFFplotter = PCR_CCF_Plotter(
            numSess=self.numSess,
            cue_list=self.cue_list,
            groups=self.groups,
            forPres=self.forPres,
            sessFocus=self.sessFocus,
            numSessByID=self.numSessByID,
        )
        # print("Plotting 1PDF:")
        # self.PCRCFFplotter.plot_cueTrigSig_1PDF(
        #     self.TrigSig_byCluster, self.CueCell_byTrigSig_byCluster, self.numSess
        # )
        # self.print_done_small_proc()

        print("Plotting Mean Trig Sig:")
        self.PCRCFFplotter.plot_meanTrigSig(self.meanTS_byCC_byCueType)
        self.print_done_small_proc()

    def plot_OptoScatter(self) -> None:
        """
        Plots the opto scatter plot.
        """

        def cell_check(listOfTuples: list, cell2check: str):
            """
            Checks the cell number in the list of tuples.

            Parameters:
                listOfTuples (list): The list of tuples to check.
                cell2check (str): The cell number to check.

            Returns:
                float: The value of the cell number.
            """
            for c, value in listOfTuples:
                if c == cell2check:
                    return value

        print("Plotting Opto Scatter")
        self.OptoAmpArr = [{} for _ in range(self.numSess)]
        for sess in range(self.numSess):
            for optokeys in self.OIDX[sess]:
                for cell, ctype in self.OIDX[sess][optokeys]:
                    if ctype == "CUE":
                        for cue in self.OAMP[sess][optokeys].keys():
                            if cue not in self.OptoAmpArr[sess].keys():
                                self.OptoAmpArr[sess][cue] = []
                            amp = cell_check(self.OAMP[sess][optokeys][cue], cell)
                            std = cell_check(self.OSTD[sess][optokeys][cue], cell)
                            self.OptoAmpArr[sess][cue].append((cell, amp, std))

        self.PCRCFFplotter.plot_OptoSingleSubj(self.OptoAmpArr)
        self.print_done_small_proc()

    def find_SpatialShift(self) -> None:
        """
        Finds the spatial shift of the trigger signal.
        """

        cues4shift = ["CUE1", "CUE1_SWITCH", "OMITCUE1"]

        if len(self.cue_list) == 2:
            cues4shift = ["CUE1", "OMITCUE1"]

        self.MaxValTrigSig_byCluster = {
            clst: {cue: [] for cue in cues4shift} for clst in self.cluster_keys
        }
        meanMaxVal_byCC_byCueType = {
            CC: {cue: [[] for _ in range(self.numSess)] for cue in cues4shift}
            for CC in self.cc_type
        }

        for clst in self.cluster_keys:
            for cue in cues4shift:
                if cue in self.TrigSig_byCluster[clst].keys():
                    for sess, ts in enumerate(self.TrigSig_byCluster[clst][cue]):
                        maxVal = np.nanmax(ts[30:, :], axis=0)
                        meanMV = np.nanmean(maxVal)
                        self.MaxValTrigSig_byCluster[clst][cue].append(maxVal)
                        if self.CueCell_byTrigSig_byCluster[clst][sess]:
                            meanMaxVal_byCC_byCueType["CUE"][cue][sess].append(meanMV)
                        else:
                            meanMaxVal_byCC_byCueType["NONCUE"][cue][sess].append(
                                meanMV
                            )
        self.meanMaxVal_byCC_byCueType = meanMaxVal_byCC_byCueType
        # ensure all arrays are the same shape
        # self.meanMaxVal_byCC_byCueType = self._ensure_each_entry_same_shape(
        #     meanMaxVal_byCC_byCueType, main_fl_keys=self.cc_type, keylist2pad=cues4shift
        # )

    def collate2byGroupDicts(self) -> None:
        """
        Collates the data from different sessions and clusters into dictionaries based on group and cue type.
        """

        def _create_emptyDictNfill(dict2fill, key2check, mean_arrs=None, sess=None):
            if key2check not in dict2fill.keys():
                dict2fill[key2check] = [[] for _ in range(self.numSess)]

            if mean_arrs is not None:
                dict2fill[key2check] = _sessSizeCheck(dict2fill[key2check], sess)
                for sess_idx, marr in enumerate(mean_arrs):
                    if isinstance(marr, np.ndarray) and marr.size > 0:
                        # remove columns with all nans
                        marr = marr[:, ~np.all(np.isnan(marr), axis=0)].T
                        dict2fill[key2check][sess_idx].extend(marr)
                    elif isinstance(marr, list):
                        dict2fill[key2check][sess_idx].extend(marr)

            if sess is not None:
                dict2fill[key2check] = _sessSizeCheck(dict2fill[key2check], sess)

            return dict2fill

        def _sessSizeCheck(dict2check, sess):
            if len(dict2check) < sess + 1:
                dict2check.append([])
            return dict2check

        def fill_byGroupDicts(dict2look, dictByGroup2fill, sess, ByLT=False):
            if not ByLT:
                for key2check, val in dict2look.items():
                    dictByGroup2fill = _create_emptyDictNfill(
                        dictByGroup2fill, key2check, sess=sess
                    )
                    if not isinstance(val, (float, int)):
                        dictByGroup2fill[key2check][sess].extend(val[sess])
                    else:
                        dictByGroup2fill[key2check][sess].append(val)
            else:
                for lt in self.lapTypeName[sess]:
                    dictByGroup2fill = _create_emptyDictNfill(
                        dictByGroup2fill, lt, sess=sess
                    )
                    dictByGroup2fill[lt][sess].extend(dict2look[lt][sess])

            return dictByGroup2fill

        for sess in range(self.numSess):
            self.eventRate_byGroup[self.group] = fill_byGroupDicts(
                self.eventRateBySess, self.eventRate_byGroup[self.group], sess
            )
            self.IPSpk_byGroup[self.group] = fill_byGroupDicts(
                self.InfoPerSpk, self.IPSpk_byGroup[self.group], sess
            )
            self.IPSec_byGroup[self.group] = fill_byGroupDicts(
                self.InfoPerSec, self.IPSec_byGroup[self.group], sess
            )
            self.cellTotals_byGroup[self.group] = fill_byGroupDicts(
                self.cellTotals[sess], self.cellTotals_byGroup[self.group], sess
            )
            self.posRates_byGroup[self.group] = fill_byGroupDicts(
                self.posRates_bySess_TC,
                self.posRates_byGroup[self.group],
                sess,
                ByLT=True,
            )
            self.posRates_byGroup_CC[self.group] = fill_byGroupDicts(
                self.posRates_bySess_CC,
                self.posRates_byGroup_CC[self.group],
                sess,
                ByLT=True,
            )

        sess4means_byGroup = sess
        self.TrigSig_byGroup[self.group][self.ID] = self.TrigSig_byCluster
        for CC, cueTypes in self.meanTS_byCC_byCueType.items():
            for cue, mean_arrs in cueTypes.items():
                self.meanTS_byGroup[CC][self.group] = _create_emptyDictNfill(
                    self.meanTS_byGroup[CC][self.group],
                    cue,
                    mean_arrs,
                    sess4means_byGroup,
                )
        # if self.optoCheck:
        #     for idx, sess in enumerate(self.OptoAmpArr):
        #         for cue, amp_vals in sess.items():
        #             for cell, amp, std in amp_vals:
        #                 if cue not in self.OptoAmpArrAllSubj.keys():
        #                     self.OptoAmpArrAllSubj[cue] = []
        #                 self.OptoAmpArrAllSubj[cue].append(
        #                     (self.ID, idx, cell, amp, std)
        #                 )
        if not self.optoCheck:
            self.maxVal_byGroup[self.group][self.ID] = self.MaxValTrigSig_byCluster
            for CC, cueTypes in self.meanMaxVal_byCC_byCueType.items():
                for cue, mean_arrs in cueTypes.items():
                    self.meanMV_byGroup[CC][self.group] = _create_emptyDictNfill(
                        self.meanMV_byGroup[CC][self.group],
                        cue,
                        mean_arrs,
                        sess4means_byGroup,
                    )

    def export_data(self) -> None:
        """
        Export data to a file.

        This method exports the data stored in the `PCRTrigSigDict` attribute to a file.
        The data is saved using the `savedict2file` method from the `saveNloadUtils` class.
        The exported data includes the following dictionaries:
        - `TS_BC`: TrigSig_byCluster
        - `CC_BTS_BC`: CueCell_byTrigSig_byCluster
        - `MEAN`: meanTS_byCC_byCueType
        - `OPTO`: OptoAmpArr if self.optoCheck else None
        - `POS_RATE`: posRates_bySess
        - `TC_IDX`: TC_IDX
        - `CC_IDX`: CC_IDX
        """

        self.PCRTrigSigDict = {
            self.PCRkey["TS_BC"]: self.TrigSig_byCluster,
            self.PCRkey["CC_BTS_BC"]: self.CueCell_byTrigSig_byCluster,
            self.PCRkey["MEAN_TS"]: self.meanTS_byCC_byCueType,
            # self.PCRkey["MEAN_MV"]: self.meanMaxVal_byCC_byCueType
            # if not self.optoCheck
            # else None,
            # self.PCRkey["OPTO"]: self.OptoAmpArr if self.optoCheck else None,
            self.PCRkey["POS_RATE"]: self.posRates_bySess,
            # self.PCRkey["POS_RATE_TC"]: self.posRates_bySess_TC,
            # self.PCRkey["POS_RATE_CC"]: self.posRates_bySess_CC,
            self.PCRkey["TC_IDX"]: self.TC_IDX,
            self.PCRkey["CC_IDX"]: self.CC_IDX,
        }
        self.saveNloadUtils.savedict2file(
            dict_to_save=self.PCRTrigSigDict,
            dict_name=self.dict_name[self.program_name],
            filename=self.dict_name[self.program_name],
            filetype_to_save=[self.file_tag["PKL"], self.file_tag["MAT"]],
        )

    def post_proc_run(self) -> None:
        """
        Runs the post-processing steps for the CueCellFinder.

        This method performs the following steps:
        1. Changes to the parent directory.
        2. Retrieves the IDs and cues.
        3. Reorganizes the mean by group.
        4. Plots the mean by group.
        """

        # change to parent dir
        self.folder_tools.chdir_check_folder(self.dayPath, create=False)
        self.get_IDsnCues()
        self.ReorgMean_byGroup()
        self.plot_ByGroupStructs()

    def get_IDsnCues(self) -> None:
        """
        Retrieves the unique IDs and cues from the TrigSig_byGroup and meanTS_byGroup dictionaries.
        """

        self.ID_set = set()
        self.cue_set = set()
        for groups, ids in self.TrigSig_byGroup.items():
            for ID in ids.keys():
                self.ID_set.add(ID)

        for _, groups in self.meanTS_byGroup.items():
            for _, cues in groups.items():
                for cue in cues.keys():
                    self.cue_set.add(cue)

    def ReorgMean_byGroup(self) -> None:
        """
        Reorganizes the mean time series data by group, cue, and session.

        This method iterates over the meanTS_byGroup dictionary and reorganizes the time series data
        by group, cue, and session. It converts each session's data into a numpy array and updates
        the meanTS_byGroup dictionary with the reorganized data.
        """

        def _convert2array(byGroupdict, sess2use):
            """
            Converts the data in the byGroupdict to numpy arrays.

            Parameters:
                byGroupdict (dict): The dictionary containing the data to convert.
                sess2use (int): The number of sessions to use.

            Returns:
                dict: The dictionary with the data converted to numpy arrays.
            """

            for group in byGroupdict.keys():
                for key in byGroupdict[group].keys():
                    for sess in range(sess2use):
                        byGroupdict[group][key][sess] = np.array(
                            byGroupdict[group][key][sess]
                        )
            return byGroupdict

        for CC, groups in self.meanTS_byGroup.items():
            for group, cues in groups.items():
                for cue, sessions in cues.items():
                    for sess_idx, sess in enumerate(sessions):
                        arr = np.array(sess).T
                        if arr.size > 0:
                            for col_idx in range(arr.shape[1]):
                                if np.mean(arr[:, col_idx]) > self.outlier_ts:
                                    arr[:, col_idx] = np.nan
                            self.meanTS_byGroup[CC][group][cue][sess_idx] = arr
                        else:
                            self.meanTS_byGroup[CC][group][cue][sess_idx] = np.array(
                                np.nan
                            )

        sess2use = max(self.numSessByID)

        self.posRates_byGroup = _convert2array(self.posRates_byGroup, sess2use=sess2use)
        self.posRates_byGroup_CC = _convert2array(
            self.posRates_byGroup_CC, sess2use=sess2use
        )
        self.IPSpk_byGroup = _convert2array(self.IPSpk_byGroup, sess2use=sess2use)
        self.eventRate_byGroup = _convert2array(
            self.eventRate_byGroup, sess2use=sess2use
        )
        self.cellTotals_byGroup = _convert2array(
            self.cellTotals_byGroup, sess2use=sess2use
        )

    def plot_ByGroupStructs(self) -> None:
        """
        Plots the by group structures.
        """

        print("Plotting by Group Structures:")
        # reinintialize the PCR_CCF_Plotter object to save in GROUP folder
        self.PCRCFFplotter = PCR_CCF_Plotter(
            numSess=self.numSess,
            cue_list=self.cue_list,
            groups=self.groups,
            sessFocus=self.sessFocus,
            numSessByID=self.numSessByID,
            forPres=self.forPres,
            fig_fname=self.text_lib["FIGSAVE"]["GROUP"],
        )

        self.print_wFrm("Mean TrigSig")
        self.PCRCFFplotter.plot_meanTS_byGroup(self.meanTS_byGroup)
        if not self.optoCheck:
            self.print_wFrm("Mean Peak Amplitude")
            self.PCRCFFplotter.plot_meanMV_byGroup(self.meanMV_byGroup)
            self.print_wFrm("Mean Peak Amplitude (Focusing on CUE1)")
            self.PCRCFFplotter.plot_meanMV_cueFocus(self.meanMV_byGroup, yLim=22)

        if self.optoCheck and not self.eOPN3Check:
            self.print_wFrm("Opto Amplitude Scatter")
            self.PCRCFFplotter.plot_OptoAllSubj(self.OptoAmpArrAllSubj)

        self.print_wFrm("posRates")
        self.PCRCFFplotter.plot_PCR_Tuning(
            self.posRates_byGroup, self.posRates_byGroup_CC
        )

        self.print_wFrm("Cell Proportions (Cue vs Tuned Cells)")
        self.PCRCFFplotter.plot_cellProps_byGroup(self.cellTotals_byGroup)

        # For plotting general rates by group:
        # Info Per Spike, Info Per Second, Event Rate

        gRbyGroup = [self.IPSpk_byGroup, self.IPSec_byGroup, self.eventRate_byGroup]
        ylabels = [
            "Info Per Spike",
            "Info Per Second",
            "Event Rate over whole session (events/s)",
        ]
        fname_pre = ["IPSpk", "IPSec", "EventRate"]
        # yLims = [4, 0.1, 0.0060]
        yLims = [4, 0.1, None]

        for gR, yLab, fnamePre, yLim in zip(gRbyGroup, ylabels, fname_pre, yLims):
            self.print_wFrm(yLab)
            self.PCRCFFplotter.plot_genRatesbyGroup(
                gR, ylabel=yLab, fname_pre=fnamePre, yLim=yLim
            )

        if self.eOPN3Check:
            self.PCRCFFplotter.plot_eventRateScatter_eOPN3(self.eventRate_byGroup)

        self.print_done_small_proc()


if "__main__" == __name__:
    from CLAH_ImageAnalysis.unitAnalysis import UA_enum

    run_CLAH_script(PostCR_CueCellFinder, parser_enum=UA_enum.Parser4PCRCFF)

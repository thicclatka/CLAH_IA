import copy

# from typing import List
import numpy as np
import scipy
from scipy.sparse import csr_matrix, issparse

# from collections import OrderedDict
from CLAH_ImageAnalysis.core import BaseClass as BC


class CRwROI_utils(BC):
    def __init__(
        self,
        program_name: str,
        numSessFound: int,
        numSess2use: int,
        CRkey: dict,
        CRprt: dict,
    ) -> None:
        self.class_type = "utils"
        BC.__init__(self, program_name=program_name, mode=self.class_type)
        self.numSessFound = numSessFound
        self.numSess = numSess2use
        self.CRkey = CRkey
        self.CRprt = CRprt
        self.gm_tls = self.dep.geometric_tools

    def find_centroidNbounds_fromROIsparse(
        self, ROIsparse: np.ndarray, sparse_dims: tuple, min_dist_adjustor: float = 0.7
    ) -> tuple:
        """
        Finds the centroid and bounds from a sparse ROI.

        Parameters:
            ROIsparse (ndarray): Sparse ROI represented as a 2D array.
            min_dist_adjustor (float, optional): Adjustment factor for the minimum distance. Defaults to 0.7.

        Returns:
            tuple: A tuple containing the centroid, bounds, and minimum distance.

        """
        ROIsparse = ROIsparse.reshape(sparse_dims[0], sparse_dims[1])
        maxVal = np.max(sparse_dims)

        # Find non-zero elements
        rows, cols = ROIsparse.nonzero()
        # Combine rows and cols into a single array of coordinates
        nonzero = np.array(list(zip(rows, cols)))

        # Find core points
        core_points = self.gm_tls.apply_DBScan2find_CorePoints(
            arr_pts=nonzero, eps=3, min_samples=10
        )

        if len(core_points) > 0:
            # Calculate the centroid of the core points
            centroid = self.gm_tls.find_center_of_mass(
                arr_pts=core_points, COM_type="median"
            )

            # get the convex hull points
            # use to find min distance
            hull_points = self.gm_tls.find_convel_hull(core_points)

            min_dist = (
                self.gm_tls.find_min_dist_from_COM(COM=centroid, arr_pts=hull_points)
                * min_dist_adjustor
            )

            # Adjust bounds based on minimum distance
            bounds = self.gm_tls.adjust_bounds_fromCOM_w_min_dist(
                COM=centroid, min_dist=min_dist, max_val_allowed=maxVal
            )

        return centroid, bounds, min_dist

    def find_contours_of_nonzero(
        self,
        ROI_sparse: scipy.sparse.coo_matrix,
        sparse_dims: tuple,
        contour_level: float = 0.45,
    ) -> list:
        """
        Find contours of non-zero elements in a sparse ROI.

        Parameters:
            ROI_sparse (scipy.sparse.coo_matrix): Sparse matrix representing the region of interest (ROI).
            contour_level (float, optional): Contour level used for finding the contours. Default is 0.45.

        Returns:
            list: List of contours found in the ROI.
        """
        ROI_dense = ROI_sparse.toarray().reshape(sparse_dims[0], sparse_dims[1])

        # find contours, by default, arr is normalized before contour detection
        contours = self.gm_tls.find_contours(ROI_dense, contour_level=contour_level)

        return contours

    def print_loaded_mSSS_results(self, ID: str) -> None:
        """
        Prints the loaded file results.

        Parameters:
            ID (str): The subject ID.
            num_sess (int): The number of sessions found within multSessSegStruc.
        """
        self.print_wFrm("Loaded file results:")
        self.print_wFrm(f"Subject: {ID}", frame_num=1)
        self.print_wFrm(
            f"Found {self.numSessFound} sessions w/in multSessSegStruc",
            frame_num=1,
        )
        self.print_wFrm(
            f"Using {self.numSess} sessions for analysis",
            frame_num=1,
            new_line_after=1,
        )

    def organize_cellTypes_fromCCT(self, List_CueCellTable: list) -> dict:
        """
        Handles the cell types from the CueCellTable. Results are a dictionary of lists of boolean arrays. 1 indicates the cell is of that type, 0 indicates it is not.

        Parameters:
            List_CueCellTable (list): The list of CueCellTable.

        Returns:
            dict: The cell types. Each entry is a list of lists of boolean arrays. 1 indicates the cell is of that type, 0 indicates it is not. First layer of lists is for each session, second layer of lists represents the cell types in a boolean manner for each cell for that session.
        """
        CCT = List_CueCellTable
        numSess = len(CCT)

        keys2check = CCT[0].keys()
        keys2check = [key for key in keys2check if "_IDX" in key]
        keys2check = [key for key in keys2check if "NOT" not in key]

        isCell = {key.split("_")[0]: [] for key in keys2check}

        for sess in range(numSess):
            total_cells = CCT[sess]["TOTAL"]
            for key in keys2check:
                key4isCell = key.split("_")[0]
                cellidx = np.array(CCT[sess][key], dtype=int)
                isCell_curr = np.zeros(total_cells, dtype=int)
                isCell_curr[cellidx] = 1
                isCell[key4isCell].append(isCell_curr)

        keys2check4both = [key for key in isCell if "CUE" in key and len(key) > 3]

        if len(keys2check4both) > 1:
            isCell["BOTHCUES"] = [[] for _ in range(numSess)]
            for sess in range(numSess):
                cue1 = isCell["CUE1"][sess]
                cue2 = isCell["CUE2"][sess]
                both = np.logical_and(cue1, cue2)
                newcue1 = np.array(
                    [1 if c1 and not both[i] else 0 for i, c1 in enumerate(cue1)]
                )
                newcue2 = np.array(
                    [1 if c2 and not both[i] else 0 for i, c2 in enumerate(cue2)]
                )
                isCell["CUE1"][sess] = newcue1
                isCell["CUE2"][sess] = newcue2
                isCell["BOTHCUES"][sess] = both

        return isCell

    @staticmethod
    def update_count_dict(
        count_dict: dict,
        check: int | None = None,
        qc_check: int | None = None,
        total_key: str | None = None,
        qc_key: str | None = None,
    ) -> None:
        """
        Update the count dictionary with the given check and qc_check values.

        Parameters:
            count_dict (dict): The count dictionary to be updated.
            check (int or None): The check value to be used for updating the total count.
            qc_check (int or None): The qc_check value to be used for updating the qc count.
            total_key (str or None): The key in the count dictionary for the total count.
            qc_key (str or None): The key in the count dictionary for the qc count.

        Returns:
        None
        """
        if check is not None and qc_check is not None:
            count_dict[total_key] = CRwROI_utils.checkNcounter(
                check, count_dict[total_key]
            )
        if qc_check is not None and qc_key is not None:
            count_dict[qc_key] = CRwROI_utils.checkNcounter(
                qc_check, count_dict[qc_key]
            )

    @staticmethod
    def checkNcounter(check: bool, count: int | list) -> int | list:
        """
        Checks the value of `check` and updates the value of `count` based on the result.

        Parameters:
            check (bool): The value to be checked.
            count (int or list): The counter value to be updated.

        Returns:
            int or list: The updated counter value.

        If `count` is an integer, it will be incremented by 1 if `check` is True, otherwise it will remain unchanged.
        If `count` is a list, True will be appended if `check` is True, otherwise False will be appended.

        Examples:
        >>> count = 0
        >>> checkNcounter(True, count)
        1
        >>> checkNcounter(False, count)
        1
        >>> count = []
        >>> checkNcounter(True, count)
        [True]
        >>> checkNcounter(False, count)
        [True, False]
        """
        if isinstance(count, list):
            return count.append(True if check else False)
        if isinstance(count, int):
            return count + 1 if check else count

    @staticmethod
    def extract_alpha_labels(
        cl_silhouette: np.ndarray,
        cl_silhouette_thresh: float,
        cl_intra_means: np.ndarray,
        cl_intra_means_thresh: float,
    ) -> np.ndarray:
        """
        Extracts alpha labels based on given thresholds.

        Parameters:
            cl_silhouette (numpy.ndarray): The silhouette values.
            cl_silhouette_thresh (float): The threshold for silhouette values.
            cl_intra_means (numpy.ndarray): The intra-means values.
            cl_intra_means_thresh (float): The threshold for intra-means values.

        Returns:
            numpy.ndarray: The alpha labels.

        """
        alpha_labels = (np.array(cl_silhouette) > cl_silhouette_thresh) * (
            np.array(cl_intra_means) > cl_intra_means_thresh
        )
        return alpha_labels

    @staticmethod
    def extract_FOV_heightNwidth(
        MSS_struct: dict, subj_sessions: list, sD_str: dict
    ) -> tuple:
        """
        Extracts the field of view (FOV) height and width for each session.

        Parameters:
            MSS_struct (dict): A dictionary containing session information.
            subj_sessions (list): A list of session IDs.
            sD_str (dict): A dictionary containing string keys for session information.

        Returns:
            FOV_height (list): A list of FOV heights for each session.
            FOV_width (list): A list of FOV widths for each session.
        """
        FOV_height = [int(MSS_struct[sess][sD_str["DY"]]) for sess in subj_sessions]
        FOV_width = [int(MSS_struct[sess][sD_str["DX"]]) for sess in subj_sessions]
        return FOV_height, FOV_width

    @staticmethod
    def reshape_spatial4ROICaT(
        array: np.ndarray | scipy.sparse.coo_matrix,
        out_height_width: tuple[int, int],
        transpose: bool = False,
        threeD: bool = False,
        order: str = "C",
        sparse_output: bool = False,
    ) -> np.ndarray | scipy.sparse.coo_matrix:
        """
        Reshape the array (or sparse matrix) to the specified height and width.

        Parameters:
            array (np.ndarray or sparse matrix): The array to reshape.
            out_height_width (tuple[int, int]): The output height and width as a tuple (height, width).
            transpose (bool, optional): Whether to transpose the array. Defaults to False.
            threeD (bool, optional): Whether the array is 3D. Defaults to False.
            order (str, optional): The order of the elements in the reshaped array. Defaults to "C".
            sparse_output (bool, optional): Whether to return a sparse matrix. Defaults to False.

        Returns:
            np.ndarray or sparse matrix: The reshaped array.
        """
        if issparse(array):
            array = array.toarray()  # Convert sparse matrix to dense array

        if transpose:
            # this is for the spatial Footprints defined by A_Spat
            array = array.T

        if threeD:
            array = array.reshape(
                (-1, out_height_width[0], out_height_width[-1]), order=order
            )
            array = np.transpose(array, (0, 2, 1))
        else:
            array = array.reshape(
                (out_height_width[0], out_height_width[-1]), order=order
            )
            array = array.T

        if sparse_output:
            return csr_matrix(
                array.reshape(array.shape[0], -1)
            )  # Convert back to sparse matrix
        else:
            return array

    @staticmethod
    def create_FOV_images(
        MSS_struct: np.ndarray,
        subj_sessions: list,
        out_height_width: tuple[int, int],
    ) -> np.ndarray:
        """
        Create a field of view (FOV) image from the MSS_struct. This is derived from the IMG key within the MSS_struct. This should be the averaged CaCh image of temporally filtered, downsampled image.

        Parameters:
            MSS_struct (numpy.ndarray): The MSS_struct containing the image data.
            out_height_width (List[int]): The desired output height and width of the FOV image.

        Returns:
            numpy.ndarray: The FOV image.

        """
        FOV_images = np.stack(
            [
                np.transpose(
                    CRwROI_utils.reshape_spatial4ROICaT(
                        # MSS_struct[session][sD_str["B_BACK_SPAT"]][:, 0],
                        MSS_struct[session]["IMG"],
                        out_height_width,
                        order="F",
                    ).astype(np.float32)
                )
                for session_idx, session in enumerate(subj_sessions)
            ]
        )
        FOV_images = FOV_images - FOV_images.min(axis=(1, 2), keepdims=True)
        FOV_images = FOV_images / FOV_images.max()
        return FOV_images

    @staticmethod
    def find_nbinsNsmooth_window(
        s_sf_nnz: int,
        n_bin_div: int = 30000,
        n_bin_max: int = 1000,
        n_bin_min: int = 30,
        sw_div: int = 10,
    ):
        """
        Calculates the number of bins and smooth window size based on the given parameters.

        Parameters:
            s_sf_nnz (int): The number of non-zero elements in the input array.
            n_bin_div (int): The divisor used to calculate the maximum number of bins.
            n_bin_max (int): The maximum number of bins allowed.
            n_bin_min (int): The minimum number of bins allowed.
            sw_div (int): The divisor used to calculate the smooth window size.

        Returns:
            n_bins (int): The calculated number of bins.
            smooth_window (int): The calculated smooth window size.
        """
        n_bins = max(min(s_sf_nnz // n_bin_div, n_bin_max), n_bin_min)
        smooth_window = n_bins // sw_div
        return n_bins, smooth_window

    def d2dNw2w_incQC_counter(
        self,
        cid_bySess: tuple,
        alpha_labels: np.ndarray,
        count_dict: dict,
        rejected_label: int,
        isCell: dict,
        labelBySess: list,
    ) -> dict:
        """
        Increment the count of occurrences for different checks based on cluster IDs.

        Parameters:
            cid_bySess (tuple): A tuple containing cluster IDs for different sessions.
            alpha_labels (list): A list of alpha labels.
            count_dict (dict): A dictionary to store the count of occurrences.
            rejected_label (int): The rejected label.

        Returns:
            dict: The updated count dictionary.
        """

        # TODO: MAKE THIS FUNCTION MORE READABLE & LESS REPETITIVE
        ref_ids = cid_bySess[0]
        day_ids = cid_bySess[1]
        week_ids = cid_bySess[2] if len(cid_bySess) >= 3 else None

        ref_labels = labelBySess[0]
        day_labels = labelBySess[1]
        week_labels = labelBySess[2] if len(labelBySess) >= 3 else None

        ref_TC = isCell["PLACE"][0]
        day_TC = isCell["PLACE"][1]
        week_TC = isCell["PLACE"][2] if len(isCell["PLACE"]) >= 3 else None

        ref_CC = isCell["CUE"][0]
        day_CC = isCell["CUE"][1]
        week_CC = isCell["CUE"][2] if len(isCell["CUE"]) >= 3 else None

        for cid in ref_ids:
            if cid == rejected_label:
                if cid == rejected_label:
                    continue
            # since alpha is 0 based cid + 1 will be index for alpha_labels
            # given -1 is the rejected label at 0th entry
            # alpha_label is ordered by cluster_id across sessions
            # - range: -1 - max cluster ID (some pos #)
            alpha_idx = cid + 1

            isCC_ref = ref_CC[ref_labels == cid]
            isCC_day = day_CC[day_labels == cid]

            isTC_ref = ref_TC[ref_labels == cid]
            isTC_day = day_TC[day_labels == cid]

            day_check = (day_ids == cid).sum()
            qc_day_check = alpha_labels[alpha_idx] * day_check

            dc_cc = day_check * (isCC_ref * isCC_day)
            qc_dc_cc = alpha_labels[alpha_idx] * dc_cc

            tc_cc = day_check * (isTC_ref * isTC_day)
            qc_tc_cc = alpha_labels[alpha_idx] * tc_cc

            if week_ids is not None:
                week_check = (week_ids == cid).sum()
                qc_week_check = alpha_labels[alpha_idx] * week_check
                qc_allSess_check = alpha_labels[alpha_idx] * day_check * week_check

                isCC_week = week_CC[week_labels == cid]
                isTC_week = week_TC[week_labels == cid]

                wk_cc = week_check * (isCC_ref * isCC_week)
                qc_wk_cc = alpha_labels[alpha_idx] * dc_cc

                wk_tc = week_check * (isTC_ref * isTC_week)
                qc_wk_tc = alpha_labels[alpha_idx] * tc_cc
            else:
                week_check = 0
                qc_week_check = 0
                qc_allSess_check = 0

                wk_cc = 0
                qc_wk_cc = 0

                wk_tc = 0
                qc_wk_tc = 0

            # determine check & count if check is 1
            CRwROI_utils.update_count_dict(
                count_dict,
                day_check,
                qc_day_check,
                self.CRkey["D2D"],
                self.CRkey["D2D_QC"],
            )
            CRwROI_utils.update_count_dict(
                count_dict,
                dc_cc,
                qc_dc_cc,
                self.CRkey["D2D_CC"],
                self.CRkey["D2D_CC_QC"],
            )
            CRwROI_utils.update_count_dict(
                count_dict,
                tc_cc,
                qc_tc_cc,
                self.CRkey["D2D_TC"],
                self.CRkey["D2D_TC_QC"],
            )
            CRwROI_utils.update_count_dict(
                count_dict,
                week_check,
                qc_week_check,
                self.CRkey["W2W"],
                self.CRkey["W2W_QC"],
            )
            CRwROI_utils.update_count_dict(
                count_dict,
                wk_cc,
                qc_wk_cc,
                self.CRkey["W2W_CC"],
                self.CRkey["W2W_CC_QC"],
            )
            CRwROI_utils.update_count_dict(
                count_dict,
                wk_tc,
                qc_wk_tc,
                self.CRkey["W2W_TC"],
                self.CRkey["W2W_TC_QC"],
            )
            CRwROI_utils.update_count_dict(
                count_dict=count_dict,
                qc_check=qc_allSess_check,
                qc_key=self.CRkey["ALLSESS_QC"],
            )
        for cid in day_ids:
            if cid == rejected_label:
                if cid == rejected_label:
                    continue
            # since alpha is 0 based cid + 1 will be index for alpha_labels
            # given -1 is the rejected label at 0th entry
            # alpha_label is ordered by cluster_id across sessions
            # - range: -1 - max cluster ID (some pos #)
            alpha_idx = cid + 1
            if week_ids is not None:
                week_check = (week_ids == cid).sum()
                qc_week_check = alpha_labels[alpha_idx] * week_check

                isCC_day = day_CC[day_labels == cid]
                isCC_week = week_CC[week_labels == cid]

                isTC_day = day_TC[day_labels == cid]
                isTC_week = week_TC[week_labels == cid]

                wk_cc = week_check * (isCC_day * isCC_week)
                qc_wk_cc = alpha_labels[alpha_idx] * wk_cc

                wk_tc = week_check * (isTC_day * isTC_week)
                qc_wk_tc = alpha_labels[alpha_idx] * wk_tc
            else:
                week_check = 0
                qc_week_check = 0

                wk_cc = 0
                qc_wk_cc = 0

                wk_tc = 0
                qc_wk_tc = 0

            CRwROI_utils.update_count_dict(
                count_dict,
                week_check,
                qc_week_check,
                self.CRkey["W2W2"],
                self.CRkey["W2W2_QC"],
            )
            CRwROI_utils.update_count_dict(
                count_dict,
                wk_cc,
                qc_wk_cc,
                self.CRkey["W2W2_CC"],
                self.CRkey["W2W2_CC_QC"],
            )
            CRwROI_utils.update_count_dict(
                count_dict,
                wk_tc,
                qc_wk_tc,
                self.CRkey["W2W2_TC"],
                self.CRkey["W2W2_TC_QC"],
            )
        return count_dict

    def isTC_incQC_counter(
        self,
        results: dict,
        isCell: dict | None,
        alpha_labels: np.ndarray,
        rejected_label: int,
    ) -> dict:
        """
        Calculate the number of tune cells (TC) and cue cells (CC) values before and after quality control (QC) for each session.

        Parameters:
            results (dict): A dictionary containing the results of the image analysis.
            isCell (dict): A dictionary containing a list of boolean values for each session and for each cell type (TC & CC).
            alpha_labels (list): A list of alpha labels for each cluster.
            rejected_label (int): The label indicating a rejected cluster.

        Returns:
            dict: A dictionary containing the isPC values before and after QC for each session.
                The dictionary has two keys: "PRE_QC" and "POST_QC".
                The value for each key is a numpy array containing the isPC values for each session.
        """
        if isCell is None:
            cellTypes = None
            isCellCount_dict = None
        else:
            cellTypes = isCell.keys()
            isCellCount_dict = {cT: {"PRE_QC": [], "POST_QC": []} for cT in cellTypes}

        if cellTypes is not None:
            for ct in cellTypes:
                for labels, counts in zip(
                    results[self.CRkey["CLUSTERS"]]["labels_bySession"], isCell[ct]
                ):
                    isC_forSess = []
                    isC_forSess_QC = []

                    for cid, cellBool in zip(labels, counts):
                        # each PC_cell is either True or False
                        if cid == rejected_label:
                            # if rejected label, append False
                            isC_forSess.append(False)
                            isC_forSess_QC.append(False)
                            continue
                        # set alpha label similarly to above
                        # see note above for alpha_labels
                        alpha_idx = cid + 1
                        # check if PC cell post QC
                        cellBool_QC = alpha_labels[alpha_idx] * cellBool
                        # determine check & append True if check is 1
                        CRwROI_utils.checkNcounter(cellBool, isC_forSess)
                        CRwROI_utils.checkNcounter(cellBool_QC, isC_forSess_QC)
                    # append to isPC_dict accordingly
                    isCellCount_dict[ct]["PRE_QC"].append(np.array(isC_forSess))
                    isCellCount_dict[ct]["POST_QC"].append(np.array(isC_forSess_QC))

        return isCellCount_dict

    def create_resultsNrun_data(
        self,
        roicat: object,
        labels2use: np.ndarray,
        n_roi: int,
        aligner: object,
        clusterer: object,
        spatialFootprints: np.ndarray,
        FOV_height_width: tuple,
        session_bool: np.ndarray,
        n_session: int,
        folder_path: str,
        kwargs_mcdm_tmp: dict,
        self_sdict: dict,
        blurrer_sdict: dict,
        roinet_sdict: dict,
        swt_sdict: dict,
        sim_sdict: dict,
    ) -> tuple:
        """
        Creates the results and run_data dictionaries for later saving.

        Parameters:
            roicat (object): The roicat object.
            labels2use (list): The list of labels to use.
            n_roi (int): The number of ROIs.
            aligner (object): The aligner object.
            clusterer (object): The clusterer object.
            spatialFootprints (np.ndarray): The spatial footprints.
            FOV_height_width (tuple): The height and width of the FOV.
            session_bool (np.ndarray): The session boolean array.
            n_session (int): The number of sessions.
            folder_path (str): The folder path.
            kwargs_mcdm_tmp (dict): The optimal parameters.
            self_sdict (dict): The self dictionary.
            blurrer_sdict (dict): The blurrer dictionary.
            roinet_sdict (dict): The roinet dictionary.
            swt_sdict (dict): The swt dictionary.
            sim_sdict (dict): The sim dictionary.

        Returns:
            tuple: A tuple containing the results and run_data dictionaries.
        """

        # label keys fits order of output for make_label_variants
        label_keys = ["squeezed", "bySession", "bool", "bool_bySession", "dict"]
        # fill in labels dict
        label_tuple = roicat.tracking.clustering.make_label_variants(
            labels=labels2use, n_roi_bySession=n_roi
        )
        # unload tuple into dict
        labels = {}
        labels = dict(zip(label_keys, label_tuple))
        # create results dict for later saving
        results = {
            self.CRkey["CLUSTERS"]: {
                "labels" if key == "squeezed" else f"labels_{key}": labels[key]
                for key in labels.keys()
            },
            self.CRkey["ROIS"]: {
                "ROIs_aligned": aligner.ROIs_aligned,
                "ROIs_raw": spatialFootprints,
                "frame_height": FOV_height_width[0],
                "frame_width": FOV_height_width[1],
                "idx_roi_session": np.where(session_bool)[1],
                self.CRkey["N_SESS"]: n_session,
            },
            "input_data": {
                "path": folder_path,
            },
            self.CRkey["QM"]: (
                clusterer.quality_metrics
                if hasattr(clusterer, self.CRkey["QM"])
                else None
            ),
            "OPTIMAL_PARAMS": kwargs_mcdm_tmp,
        }
        # create run_data dict
        run_data = copy.deepcopy(
            {
                "data": self_sdict,
                "aligner": aligner.serializable_dict,
                "blurrer": blurrer_sdict,
                "roinet": roinet_sdict,
                "swt": swt_sdict,
                "sim": sim_sdict,
                "clusterer": clusterer.serializable_dict,
            }
        )

        return results, run_data

    def create_cluster_info_dict(
        self,
        cluster_id: np.ndarray,
        allROIS: np.ndarray,
        alpha_labels: np.ndarray,
        count_dict: dict,
        isCell_pre_cluster: dict,
        isCell_post_cluster: dict,
        rejected_label: int,
        verbose: bool = False,  # TODO: FIX THIS OUTPUT, PRINTS DOESNT MATCH CLUSTER_INFO
    ) -> dict:
        """
        Create a dictionary containing information about the clusters.

        Parameters:
            cluster_id (numpy.ndarray): Array containing cluster IDs.
            allROIS (numpy.ndarray): Array containing all ROIs.
            alpha_labels (numpy.ndarray): Array containing alpha labels.
            count_dict (dict): Dictionary containing count information.
            isCell (dict): Dictionary containing info on whether cell is a cue cell or tuned cell.
            rejected_label (int): The label for rejected clusters.

        Returns:
            dict: A dictionary containing the following information:
                - CRwROItxt.UCELL: Total number of cells.
                - CRwROItxt.ACLUSTERS: Total number of accepted clusters.
                - CRwROItxt.ACLUSTERS_QC: Total number of accepted clusters after QC.
                - CRwROItxt.DISCARD: Total number of discarded clusters.
                - CRwROItxt.DISCARD_QC: Total number of discarded clusters after QC.
                - CRwROItxt.TR_D: Count of cells tracked between S1 and S2 (one day apart).
                - CRwROItxt.TR_W: Count of cells tracked between S1 and S3 (one week apart).
                - CRwROItxt.TR_AS: Count of cells tracked across all sessions (S1-3).
                - CRwROItxt.TR_D_QC: Count of cells tracked between S1 and S2 (one day apart) after QC.
                - CRwROItxt.TR_W_QC: Count of cells tracked between S1 and S3 (one week apart) after QC.
                - CRwROItxt.TR_AS_QC: Count of cells tracked across all sessions (S1-3) after QC.
        """

        def find_sumWrejected_label(
            label_arr, rejected_label, accepted=False, rejected=False
        ):
            if isinstance(label_arr, list):
                label_arr = np.array(label_arr)
            if accepted:
                return (label_arr != rejected_label).sum()
            if rejected:
                return (label_arr == rejected_label).sum()

        def find_percNprint(count, total):
            perc = count / total * 100
            return f"{count} ({perc:.2f}%)"

        UCELL = self.CRkey["UCELL"]

        # init some cluster info vars for legibility
        # ucell = tracked cells (accepted cluster or acluster) across sessions + discarded cells
        acluster = find_sumWrejected_label(cluster_id, rejected_label, accepted=True)
        discard = find_sumWrejected_label(allROIS, rejected_label, rejected=True)
        ucell = acluster + discard

        # Discarded QC is total discarded + (total clusters - 1) - total accepted post QC
        # cluster - 1 to account for the rejected label
        discard_qc = discard + ((alpha_labels.shape[0] - 1) - alpha_labels.sum())

        cluster_info = {
            UCELL: ucell,
            self.CRkey["ACLUSTERS"]: acluster,
            self.CRkey["ACLUSTERS_QC"]: alpha_labels.sum(),
            self.CRkey["DISCARD"]: discard,
            self.CRkey["DISCARD_QC"]: discard_qc,
        }

        # iterate through each session to count cells that are tracked by cell type and add to dict accordingly
        if isCell_post_cluster is not None:
            for key in isCell_post_cluster.keys():
                cluster_info[f"UNTRACKED_{key}"] = 0
                cluster_info[f"UNDERLYING_{key}"] = 0
                cluster_info[f"TRACKED_{key}"] = 0
            for cellType in isCell_post_cluster.keys():
                for i, (pre, post) in enumerate(
                    zip(
                        isCell_post_cluster[cellType]["PRE_QC"],
                        isCell_post_cluster[cellType]["POST_QC"],
                    )
                ):
                    cluster_info[f"{cellType}_S{i + 1}"] = pre.sum()
                    cluster_info[f"{cellType}_S{i + 1}_QC"] = post.sum()

            for cellType in isCell_post_cluster.keys():
                tracked_cells = 0
                for sess in range(self.numSess):
                    isCell2check = isCell_pre_cluster[cellType][sess]
                    isCell2check_clust = isCell_post_cluster[cellType]["PRE_QC"][sess]
                    trkd = cluster_info[f"{cellType}_S{sess + 1}"]

                    # find cells not tracked between sessions
                    cluster_info[f"UNTRACKED_{cellType}"] += sum(
                        [
                            1 if pre == 1 and post == 0 else 0
                            for pre, post in zip(isCell2check, isCell2check_clust)
                        ]
                    )
                    # sum cells tracked between sessions
                    tracked_cells += trkd

                untracked_cells = cluster_info[f"UNTRACKED_{cellType}"]
                underlying_cells = untracked_cells + tracked_cells
                cluster_info[f"TRACKED_{cellType}"] = tracked_cells
                cluster_info[f"UNDERLYING_{cellType}"] = underlying_cells

        total_cells = cluster_info[UCELL]

        # patterns_to_remove = ["CC_SX", "CC_SX_QC", "TC_SX", "TC_SX_QC"]

        # Create a new version of self.CRprt without the specified patterns
        # CRprt4verbosePrint = {
        #     key: value
        #     for key, value in self.CRprt.items()
        #     if not any(key.startswith(pattern) for pattern in patterns_to_remove)
        # }

        # ordered_cluster_info = OrderedDict()
        # for key4CI in self.CRkey.values():
        #     if key4CI in cluster_info:
        #         ordered_cluster_info[key4CI] = cluster_info[key4CI]

        # TODO: need to fix this
        # if verbose:
        #     print("Clustering count results:")
        #     for cluster_key, print_key in zip(cluster_info, CRprt4verbosePrint):
        #         if cluster_key in [UCELL]:
        #             self.print_wFrm(
        #                 f"{self.CRprt[print_key]} {cluster_info[cluster_key]}"
        #             )
        #         else:
        #             self.print_wFrm(
        #                 f"{self.CRprt[print_key]} {find_percNprint(cluster_info[cluster_key], total_cells)}"
        #             )
        #     print(
        #         "Note: All percentages are based on dividing by underlying cell count"
        #     )

        return cluster_info

import os
import shutil
import caiman as cm
import matplotlib.pyplot as plt
import numpy as np
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.components_evaluation import estimate_components_quality_auto
from enum import Enum

from CLAH_ImageAnalysis.core import BaseClass as BC
from CLAH_ImageAnalysis.tifStackFunc import TSF_enum


class CNMF_Utils(BC):
    """
    A utility class for performing CNMF (Constrained Non-negative Matrix Factorization)
    on calcium imaging data.

    This class provides methods for initializing CNMF, refining component patches,
    evaluating components, plotting contourimg_as_uint()s, extracting DF/F values, and exporting
    segmentation data.

    Attributes:
        basename (str): Base name for output files.
        Ca_Array (ndarray): The calcium imaging data array.
        dview (object): Parallel processing object (e.g., ipyparallel view).
        dx (int): Displacement along the x-axis.
        dy (int): Displacement along the y-axis.
        dims (tuple): Dimensions of the imaging data.
        n_processes (int): Number of processes to use in CNMF.
        k (int): Expected number of components (neurons).
        merge_thresh (float): Threshold for merging components.
        p (int): Order of autoregressive model (1 or 2) used in CNMF.
        gnb (int): Number of background components.
        gSig (tuple): Expected half-size of the neurons.
        alpha_snmf (float): Sparsity parameter for NMF.
        only_init_patch (bool): Only initialize patches without CNMF fitting.
        memory_fact (float): Memory factor for reducing data size in patches.
        method_init (str): Method for initialization ('greedy', 'sparse_nmf', etc.).
        frames (int): Number of frames in the data (used in some evaluations).
        decay_time (float): Decay time of the indicator.
        min_SNR (float): Minimum signal-to-noise ratio for accepting a component.
        r_values_min (float): Minimum R-value for accepting a component.
        use_cnn (bool): Whether to use CNN for component classification.
        thresh_cnn_min (float): Threshold for CNN classification.
        threshold (float): Threshold for component evaluation.
        vmax (float): Maximum value for contour plotting.
        method_deconvolution (str): Method used for spike deconvolution.
        check_nan (bool): Check for NaNs in the data.
        quantileMin (float): Minimum quantile for DF/F extraction.
        frames_window (int): Number of frames for smoothing in DF/F extraction.

    Methods:
        find_initial_patches: Performs the initial CNMF factorization to find patches.
        refine_patches: Refines the patches found in the initial CNMF factorization.
        comp_evaluator: Evaluates the components obtained from CNMF.
        comp_contour_plotter: Plots the contours of the components.
        fill_in_accepted_patches: Fills in the accepted patches after component evaluation.
        calc_F_dff: Calculates the DF/F (Delta F over F) values.
        createNexport_segDict: Creates and exports a dictionary with segmentation data.
    """

    def __init__(
        self,
        basename: str,
        Ca_Array: np.ndarray,
        dview: object,
        dx: int,
        dy: int,
        dims: tuple,
        n_processes: int,
        params: dict | None = None,
        onePhotonCheck: bool = False,
        extract_OR_detrend: str = "extract",
    ) -> None:
        """
        Initializes the CNMF_Utils class.

        Parameters:
            basename (str): Base name for output files.
            Ca_Array (np.ndarray): The calcium imaging data array.
            dview (object): Parallel processing object (e.g., ipyparallel view).
            dx (int): Displacement along the x-axis.
            dy (int): Displacement along the y-axis.
            dims (tuple): Dimensions of the imaging data.
            n_processes (int): Number of processes to use in CNMF.
            params (dict | None, optional): Parameters for CNMF. Defaults to None.
            onePhotonCheck (bool, optional): Whether the session is a one-photon session. Defaults to False.
            extract_OR_detrend (str, optional): Whether to extract or detrend the data. Defaults to "extract".
        """
        self.program_name = "CNMF"
        self.class_type = "utils"
        BC.__init__(self, program_name=self.program_name, mode=self.class_type)

        CNMFPRS = self.enum2dict(TSF_enum.CNMF_PARAMS)
        par_idx = 0 if not onePhotonCheck else 1
        self.CNMFpar = {k: v[par_idx] for k, v in CNMFPRS.items()}

        if params is None:
            params = {}

        alpha_snmf = params.get("ALPHA_SNMF", None)
        check_nan = params.get("CHECK_NAN", None)
        center_psf = params.get("CENTER_PSF", None)
        decay_time = params.get("DECAY", None)
        frames = params.get("FPS", None)
        frames_window = params.get("FRAME_WINDOW", None)
        gSig = params.get("GSIG", None)
        gSiz = params.get("GSIZ", None)
        gnb = params.get("GNB", None)
        k = params.get("K", None)
        low_rank_background = params.get("LOW_RANK_BACKGROUND", None)
        memory_fact = params.get("MEMORY_FACT", None)
        merge_thresh = params.get("MERGE_THRESH", None)
        method_deconvolution = params.get("METH_DECONV", None)
        method_init = params.get("METH_INIT", None)
        min_pnr = params.get("MIN_PNR", None)
        min_SNR = params.get("MIN_SNR", None)
        min_corr = params.get("MIN_CORR", None)
        nb_patch = params.get("NB_PATCH", None)
        only_init_patch = params.get("ONLY_INIT_PATCH", None)
        p = params.get("P", None)
        quantileMin = params.get("QUANTILE_MIN", None)
        r_values_min = params.get("RVAL_THR", None)
        s_min = params.get("SPIKE_MIN", None)
        thresh_cnn_min = params.get("CNN_THR", None)
        threshold = params.get("CE_THRESH", None)
        use_cnn = params.get("USE_CNN", None)
        vmax = params.get("CE_VMAX", None)

        # for rf and stride
        rf = params.get("RF", None)
        stride = params.get("STRIDE", None)

        # printing initiat of CNMF
        print("Initializing CNMF funcs for cell segmentation")

        # non-default parameters
        self.basename = basename
        self.Ca_Array = Ca_Array
        self.dview = dview
        self.dx = dx
        self.dy = dy
        self.dims = dims
        self.n_processes = n_processes
        self.extract_OR_detrend = extract_OR_detrend

        # CNMF parameters use TSF_enum.CNMF_Params if not provided by user
        self.alpha_snmf = (
            int(self.CNMFpar["ALPHA_SNMF"]) if alpha_snmf is None else alpha_snmf
        )
        self.check_nan = (
            bool(self.CNMFpar["CHECK_NAN"]) if check_nan is None else check_nan
        )
        self.center_psf = (
            bool(self.CNMFpar["CENTER_PSF"]) if center_psf is None else center_psf
        )
        self.decay_time = self.CNMFpar["DECAY"] if decay_time is None else decay_time
        self.frames = self.CNMFpar["FPS"] if frames is None else frames
        self.frames_window = (
            self.CNMFpar["FRAME_WINDOW"] if frames_window is None else frames_window
        )
        self.gSig = self.CNMFpar["GSIG"] if gSig is None else gSig
        self.gSiz = self.CNMFpar["GSIZ"] if gSiz is None else gSiz
        self.gnb = int(self.CNMFpar["GNB"]) if gnb is None else gnb
        self.k = self.CNMFpar["K"] if k is None else k
        self.low_rank_background = (
            bool(self.CNMFpar["LOW_RANK_BACKGROUND"])
            if low_rank_background is None
            else low_rank_background
        )
        self.memory_fact = (
            int(self.CNMFpar["MEMORY_FACT"]) if memory_fact is None else memory_fact
        )
        self.merge_thresh = (
            self.CNMFpar["MERGE_THRESH"] if merge_thresh is None else merge_thresh
        )
        self.method_deconvolution = (
            self.CNMFpar["METH_DECONV"]
            if method_deconvolution is None
            else method_deconvolution
        )
        self.method_init = (
            self.CNMFpar["METHOD_INIT"] if method_init is None else method_init
        )
        self.min_pnr = self.CNMFpar["MIN_PNR"] if min_pnr is None else min_pnr
        self.min_SNR = self.CNMFpar["MIN_SNR"] if min_SNR is None else min_SNR
        self.min_corr = self.CNMFpar["MIN_CORR"] if min_corr is None else min_corr
        self.nb_patch = int(self.CNMFpar["NB_PATCH"]) if nb_patch is None else nb_patch
        self.only_init_patch = (
            bool(self.CNMFpar["ONLY_INIT_PATCH"])
            if only_init_patch is None
            else only_init_patch
        )
        self.p = int(self.CNMFpar["P"]) if p is None else p
        self.quantileMin = (
            self.CNMFpar["QUANTILE_MIN"] if quantileMin is None else quantileMin
        )
        self.r_values_min = (
            self.CNMFpar["RVAL_THR"] if r_values_min is None else r_values_min
        )
        self.s_min = self.CNMFpar["SPIKE_MIN"] if s_min is None else s_min
        self.thresh_cnn_min = (
            self.CNMFpar["CNN_THR"] if thresh_cnn_min is None else thresh_cnn_min
        )
        self.threshold = self.CNMFpar["CE_THRESH"] if threshold is None else threshold
        self.use_cnn = bool(self.CNMFpar["USE_CNN"]) if use_cnn is None else use_cnn
        self.vmax = self.CNMFpar["CE_VMAX"] if vmax is None else vmax

        # for rf and stride
        rf = self.CNMFpar["RF"] if rf is None else rf
        stride = self.CNMFpar["STRIDE"] if stride is None else stride

        # parameters that need to be tuples
        self.gSig = (self.gSig, self.gSig)

        # see TSF_enum.CNMF_Params for defaults
        paramsDict = {
            "ALPHA_SNMF": self.alpha_snmf,
            "CENTER_PSF": bool(self.center_psf),
            "CE_THRESH": self.threshold,
            "CE_VMAX": self.vmax,
            "CHECK_NAN": bool(self.check_nan),
            "CNN_THR": self.thresh_cnn_min,
            "DECAY": self.decay_time,
            "FRAME_WINDOW": self.frames_window,
            "GNB": int(self.gnb),
            "GSIG": self.gSig,
            "GSIZ": self.gSiz,
            "K": int(self.k),
            "LOW_RANK_BACKGROUND": bool(self.low_rank_background),
            "MEMORY_FACT": int(self.memory_fact),
            "MERGE_THRESH": self.merge_thresh,
            "METH_DECONV": self.method_deconvolution,
            "METH_INIT": self.method_init,
            "MIN_CORR": self.min_corr,
            "MIN_PNR": self.min_pnr,
            "MIN_SNR": self.min_SNR,
            "NB_PATCH": int(self.nb_patch),
            "ONLY_INIT_PATCH": self.only_init_patch,
            "P": int(self.p),
            "QUANTILE_MIN": self.quantileMin,
            "RF": int(rf),
            "RVAL_THR": self.r_values_min,
            "SPIKE_MIN": self.s_min,
            "STRIDE": int(stride),
            "USE_CNN": self.use_cnn,
        }

        CNMF_enum = Enum("CNMF_Params", paramsDict)
        parFile = "CNMF" + ("_onePhoton" if onePhotonCheck else "")

        TSF_enum.export_settings2file(CNMF_enum, parFile)

        wCH = None
        if "Ch2" in self.basename:
            wCH = "Ch2"
        elif "Ch1" in self.basename:
            wCH = "Ch1"

        # DS image fname
        self.DS_image = self.utils.image_utils.get_DSImage_filename(wCH=wCH)

        # strucs to fill
        self._init_strucs_to_fill()

    def _init_strucs_to_fill(self) -> None:
        """
        Initializes the data structures to be filled.

        This method initializes the following attributes to None:
        - A_in
        - C_in
        - b_in
        - f_in
        - F_dff
        - NonNegMatrix_post_refining
        - idx_components
        - idx_components_bad
        - SNR_comp
        - r_values
        - cnn_preds
        - NonNegMatrix
        """
        self.A_in = None
        self.C_in = None
        self.b_in = None
        self.f_in = None
        self.F_dff = None
        self.NonNegMatrix_post_refining = None
        self.idx_components = None
        self.idx_components_bad = None
        self.SNR_comp = None
        self.r_values = None
        self.cnn_preds = None
        self.NonNegMatrix = None

    def find_initial_patches(self, rf: int, stride: int) -> None:
        """
        Find initial patches using CNMF algorithm.

        Parameters:
            rf (int): The size of the receptive field.
            stride (int): The stride used for patch extraction.

        Returns:
            None
        """
        with self.StatusPrinter.output_btw_dots(
            pre_msg="Initializing 1st factorialization",
            post_msg="1st factorialization complete",
            timekeep=True,
            timekeep_msg="Factorialization",
        ):
            NonNegMatrix = cnmf.CNMF(
                n_processes=self.n_processes,
                k=self.k,
                gSig=self.gSig,
                gSiz=self.gSiz,
                merge_thresh=self.merge_thresh,
                p=self.p,
                dview=self.dview,
                rf=rf,
                stride=stride,
                memory_fact=self.memory_fact,
                method_init=self.method_init,
                alpha_snmf=self.alpha_snmf,
                only_init_patch=self.only_init_patch,
                gnb=self.gnb,
                low_rank_background=self.low_rank_background,
                nb_patch=self.nb_patch,
                min_corr=self.min_corr,
                min_pnr=self.min_pnr,
                min_SNR=self.min_SNR,
                s_min=self.s_min,
                center_psf=self.center_psf,
            )
            # fit CNMF model to image stack
            self.NonNegMatrix = NonNegMatrix.fit(self.Ca_Array)

    def refine_patches(self, rf: int, stride: int) -> None:
        """
        Refines the patches using the accepted patches from the first factorialization.

        Parameters:
            rf (int): The patch size.
            stride (int): The stride used for patch extraction.

        Returns:
            None
        """
        with self.StatusPrinter.output_btw_dots(
            pre_msg="Initializing 2nd factorialization using accepted patches from 1st to refine",
            post_msg="2nd factorialization complete",
            timekeep=True,
            timekeep_msg="Factorialization",
        ):
            NonNegMatrix_post_refining = cnmf.CNMF(
                n_processes=self.n_processes,
                k=self.A_in.shape[-1],
                gSig=self.gSig,
                gSiz=self.gSiz,
                p=self.p,
                dview=self.dview,
                merge_thresh=self.merge_thresh,
                Ain=self.A_in,
                Cin=self.C_in,
                b_in=self.b_in,
                f_in=self.f_in,
                rf=rf,
                stride=stride,
                gnb=self.gnb,
                method_deconvolution=self.method_deconvolution,
                check_nan=self.check_nan,
                center_psf=self.center_psf,
                low_rank_background=self.low_rank_background,
                nb_patch=self.nb_patch,
                min_corr=self.min_corr,
                min_pnr=self.min_pnr,
                min_SNR=self.min_SNR,
                s_min=self.s_min,
            )
            # fit CNMF model to image stack
            self.NonNegMatrix_post_refining = NonNegMatrix_post_refining.fit(
                self.Ca_Array
            )

        # evaluate components after 2nd factorialization
        self.comp_evaluator_post2ndFactor()

        # clear NonNegMatrix for memory
        with self.StatusPrinter.garbage_collector():
            self.NonNegMatrix = None
            self.A_in, self.C_in, self.b_in, self.f_in = None, None, None, None

    def comp_evaluator(self) -> None:
        """
        Evaluate the quality of components.

        This method evaluates the quality of components by estimating various metrics such as
        signal-to-noise ratio (SNR), r-values, and CNN predictions. It prints the progress
        of the evaluation and stores the results in instance variables.
        """

        with self.StatusPrinter.output_btw_dots(
            pre_msg="Evaluating components", post_msg="Evaluation completed"
        ):
            (
                self.idx_components,
                self.idx_components_bad,
                self.SNR_comp,
                self.r_values,
                self.cnn_preds,
            ) = estimate_components_quality_auto(
                Y=self.Ca_Array,
                A=self.NonNegMatrix.estimates.A,
                C=self.NonNegMatrix.estimates.C,
                b=self.NonNegMatrix.estimates.b,
                f=self.NonNegMatrix.estimates.f,
                YrA=self.NonNegMatrix.estimates.YrA,
                frate=self.frames,
                decay_time=self.decay_time,
                gSig=self.gSig,
                dims=self.dims,
                dview=self.dview,
                min_SNR=self.min_SNR,
                r_values_min=self.r_values_min,
                use_cnn=self.use_cnn,
                thresh_cnn_min=self.thresh_cnn_min,
            )

    def comp_evaluator_post2ndFactor(self) -> None:
        """
        Evaluate the quality of components after the second factorialization.
        """

        with self.StatusPrinter.output_btw_dots(
            pre_msg="Evaluating components", post_msg="Evaluation completed"
        ):
            (
                self.idx_components_2ndFactor,
                self.idx_components_bad_2ndFactor,
                self.SNR_comp_2ndFactor,
                self.r_values_2ndFactor,
                self.cnn_preds_2ndFactor,
            ) = estimate_components_quality_auto(
                Y=self.Ca_Array,
                A=self.NonNegMatrix_post_refining.estimates.A,
                C=self.NonNegMatrix_post_refining.estimates.C,
                b=self.NonNegMatrix_post_refining.estimates.b,
                f=self.NonNegMatrix_post_refining.estimates.f,
                YrA=self.NonNegMatrix_post_refining.estimates.YrA,
                frate=self.frames,
                decay_time=self.decay_time,
                gSig=self.gSig,
                dims=self.dims,
                dview=self.dview,
                min_SNR=self.min_SNR,
                r_values_min=self.r_values_min,
                use_cnn=self.use_cnn,
                thresh_cnn_min=self.thresh_cnn_min,
            )

    def comp_contour_plotter(self, folder_path: str | None = None) -> None:
        """
        Plot the contours of components.

        This method plots the contours of accepted and rejected components based on the provided parameters.
        """
        print("Plotting contours of components")
        Cn = cm.local_correlations(self.Ca_Array.transpose(2, 1, 0))
        Cn[np.isnan(Cn)] = 0
        plt.figure()
        # Accepted components
        plt.subplot(121)
        crd_good = cm.utils.visualization.plot_contours(
            self.NonNegMatrix.estimates.A[:, self.idx_components],
            Cn,
            thr=self.threshold,
            vmax=self.vmax,
        )
        plt.title("Accepted")

        # Rejected components
        plt.subplot(122)
        crd_bad = cm.utils.visualization.plot_contours(
            self.NonNegMatrix.estimates.A[:, self.idx_components_bad],
            Cn,
            thr=self.threshold,
            vmax=self.vmax,
        )
        plt.title("Rejected")

        savefig_name = f"ContourPlot_CompEval{self.file_tag['PNG']}"

        if folder_path is not None:
            savefig_name = f"{folder_path}/" + savefig_name

        plt.savefig(savefig_name)
        plt.close()

        # clear for memory
        with self.StatusPrinter.garbage_collector():
            Cn, crd_good, crd_bad = None, None, None

        self.print_done_small_proc()

    def fill_in_accepted_patches(self) -> None:
        """
        Fills in the accepted patches in the NonNegMatrix object.

        This method updates the `A_in`, `C_in`, `b_in`, and `f_in` attributes of the NonNegMatrix object
        with the accepted patches specified by the `idx_components` attribute.
        """
        self.A_in = self.NonNegMatrix.estimates.A[:, self.idx_components]
        self.C_in = self.NonNegMatrix.estimates.C[self.idx_components, :]
        self.b_in = self.NonNegMatrix.estimates.b
        self.f_in = self.NonNegMatrix.estimates.f

    def calc_F_dff(self, Yr: np.ndarray) -> None:
        """
        Calculates the df/f values.

        Parameters:
            Yr (np.ndarray): The raw fluorescence data.
        """
        try:
            with self.StatusPrinter.output_btw_dots(
                pre_msg="Extracting df/f values",
                done_msg=True,
                timekeep=True,
                timekeep_msg="df/f extraction",
            ):
                import caiman.source_extraction.cnmf.utilities as cnmf_tools

                if self.extract_OR_detrend == "extract":
                    self.F_dff = cnmf_tools.extract_DF_F(
                        Yr=Yr,
                        A=self.NonNegMatrix_post_refining.estimates.A,
                        C=self.NonNegMatrix_post_refining.estimates.C,
                        bl=self.NonNegMatrix_post_refining.estimates.bl,
                        quantileMin=self.quantileMin,
                        frames_window=self.frames_window,
                        dview=self.dview,
                    )

                elif self.extract_OR_detrend == "detrend":
                    self.F_dff = cnmf_tools.detrend_df_f(
                        self.NonNegMatrix_post_refining.estimates.A,
                        self.NonNegMatrix_post_refining.estimates.b,
                        self.NonNegMatrix_post_refining.estimates.C,
                        self.NonNegMatrix_post_refining.estimates.f,
                        YrA=self.NonNegMatrix_post_refining.estimates.YrA,
                        quantileMin=self.quantileMin,
                        frames_window=self.frames_window,
                    )
        except Exception as e:
            print(f"Error in calc_F_dff: {e}")
            print("Setting F_dff to None\n")
            self.F_dff = None

    def createNexport_segDict(
        self,
        concatCheck: bool = False,
        folder_path_concat: str = None,
        folder_path_subj: list = [],
    ):
        """
        Creates and exports a segmentation dictionary (segDict).
        This dictionary contains C, A, DFF, DX, DY, and S:
            - C = temporal components
            - A = spatial components
            - b = background components
            - DFF = DF/F values
            - DX = displacement along x-axis
            - DY = displacement along y-axis
            - S = deconvolved signals

        Parameters:
            concatCheck (bool, optional): Whether to concatenate data. Defaults to False.
            folder_path_concat (str, optional): Path to the folder for concatenated data. Defaults to None.
            folder_path_subj (list, optional): List of paths to the folders for each subject. Defaults to [].
        """

        def _split_array(array: np.ndarray, split: int) -> tuple:
            """
            Splits an array into two parts.

            Parameters:
                array (np.ndarray): The array to split.
                split (int): The index to split the array at.

            Returns:
                tuple: A tuple containing two arrays, split at the given index.
            """
            arrayA = array[:, :split]
            arrayB = array[:, split:]
            return arrayA, arrayB

        dict_name = self.text_lib["dict_name"]

        segDict = {}
        segFilename = self.basename + self.file_tag["SD"]
        filetype_to_save = [
            self.file_tag["H5"],
            self.file_tag["PKL"],
            self.file_tag["MAT"],
        ]
        if concatCheck:
            segFilename = f"{folder_path_concat}/" + segFilename

        SDkey = self.enum2dict(TSF_enum.segDict_Txt)

        segDict = {
            SDkey["C_TEMPORAL"]: self.NonNegMatrix_post_refining.estimates.C,
            SDkey["A_SPATIAL"]: self.NonNegMatrix_post_refining.estimates.A,
            SDkey["B_BACK_SPAT"]: self.NonNegMatrix_post_refining.estimates.b,
            SDkey["F_BACK_TEMP"]: self.NonNegMatrix_post_refining.estimates.f,
            SDkey["DFF"]: self.F_dff,
            SDkey["DX"]: self.dx,
            SDkey["DY"]: self.dy,
            SDkey["S_DECONV"]: self.NonNegMatrix_post_refining.estimates.S,
            SDkey["YRA"]: self.NonNegMatrix_post_refining.estimates.YrA,
            SDkey["RSR"]: self.NonNegMatrix_post_refining.estimates.R,
            SDkey["IDX_GODO"]: self.idx_components_2ndFactor.astype(int),
            SDkey["IDX_BAD"]: self.idx_components_bad_2ndFactor.astype(int),
            SDkey["SNR_COMP"]: self.SNR_comp_2ndFactor,
            SDkey["R_VALUES"]: self.r_values_2ndFactor,
            SDkey["CNN_PREDS"]: self.cnn_preds_2ndFactor,
        }

        self.saveNloadUtils.savedict2file(
            dict_to_save=segDict,
            dict_name=dict_name["SD"],
            filename=segFilename,
            date=True,
            filetype_to_save=filetype_to_save,
        )

        print("Plotting C Temporal Results", end="", flush=True)
        self._plot_CTemp(CTemp=segDict[SDkey["C_TEMPORAL"]])
        self.print_done_small_proc(new_line=False)
        print("Plotting A Spatial Results", end="", flush=True)
        self._plot_A_Spat(A_Spat=segDict[SDkey["A_SPATIAL"]])
        self.print_done_small_proc(new_line=True)

        if concatCheck:
            print(
                "Splitting concatenated data into pre and post segments & saving into respective session folders:"
            )
            total_frames = self.Ca_Array.shape[0]
            splitIdx = int(total_frames / 2)
            C_A, C_B = _split_array(segDict[SDkey["C_TEMPORAL"]], splitIdx)
            f_A, f_B = _split_array(segDict[SDkey["F_BACK_TEMP"]], splitIdx)
            S_A, S_B = _split_array(segDict[SDkey["S_DECONV"]], splitIdx)

            list2save = [("pre", C_A, f_A, S_A), ("post", C_B, f_B, S_B)]

            for idx, (name, C, f, S) in enumerate(list2save):
                self.print_wFrm(name)
                basename2use = self.basename.replace("-", f"{name}-")
                if name == "post":
                    basename2use = basename2use.replace("001", "002")

                segDict = {
                    SDkey["C_TEMPORAL"]: C,
                    SDkey["A_SPATIAL"]: segDict[SDkey["A_SPATIAL"]],
                    SDkey["B_BACK_SPAT"]: segDict[SDkey["B_BACK_SPAT"]],
                    SDkey["F_BACK_TEMP"]: f,
                    SDkey["DFF"]: segDict[SDkey["DFF"]],
                    SDkey["DX"]: self.dx,
                    SDkey["DY"]: self.dy,
                    SDkey["S_DECONV"]: S,
                }
                segFname = (
                    f"{folder_path_subj[idx]}/"
                    + basename2use
                    + self.file_tag["COMP_SDFNAME"]
                )

                self.saveNloadUtils.savedict2file(
                    dict_to_save=segDict,
                    dict_name=f"{dict_name['SD']}_{name}",
                    filename=segFname,
                    date=True,
                    filetype_to_save=filetype_to_save,
                )

        # clear for memory
        with self.StatusPrinter.garbage_collector():
            segDict = None

    def _plot_CTemp(self, CTemp: np.ndarray) -> None:
        numCells = CTemp.shape[0]
        fig = self.fig_tools.create_plotly_subplots(
            rows=numCells, cols=1, shared_xaxes=True, vertical_spacing=0.002
        )

        for i in range(numCells):
            self.fig_tools.add_plotly_trace(
                fig=fig,
                x=np.arange(CTemp.shape[1]),
                y=CTemp[i, :],
                name=f"Cell {i + 1}",
                mode="lines",
                row=i + 1,
                col=1,
            )
            # Update both x and y axes for each subplot
            fig.update_yaxes(
                title_text=f"C{i + 1}",
                row=i + 1,
                col=1,
            )
            # fig.update_xaxes(
            #     title_text="Time (frames)",
            #     row=i + 1,
            #     col=1,
            # )

        fig.update_layout(
            title=f"Temporal Components for {self.basename}",
            height=100 * numCells,
            showlegend=False,
        )
        self.fig_tools.save_plotly(plotly_fig=fig, fig_name=f"{self.basename}_CTemp")

    def _plot_A_Spat(self, A_Spat: np.ndarray) -> None:
        DS_image = self.image_utils.read_image(self.findLatest(self.DS_image))

        fig, ax = self.fig_tools.create_plt_subplots()
        ax.imshow(DS_image, cmap="grey", aspect="equal")

        new_shape = (A_Spat.shape[1], DS_image.shape[0], DS_image.shape[1])
        ASpat2use = A_Spat.toarray()
        ASpat2use = np.transpose(ASpat2use)
        ASpat2use = ASpat2use.reshape(new_shape)

        RGB_color = self.fig_tools.hex_to_rgba(self.color_dict["green"], wAlpha=False)
        cmap = self.fig_tools.make_segmented_colormap(
            cmap_name="DSImage", hex_color=RGB_color, from_white=True
        )

        cell_num = ASpat2use.shape[0]

        for cell in range(cell_num):
            data_2d = ASpat2use[cell, :, :]
            self.fig_tools.plot_imshow(
                fig=fig,
                axis=ax,
                data2plot=data_2d.T,
                cmap=cmap,
                alpha=1,
                vmax=data_2d.max(),
            )
            self.fig_tools.label_cellNum_overDSImage(
                axis=ax,
                data=data_2d,
                cell_str=f"Cell_{cell}",
                color=self.color_dict["red"],
                fontsize=10,
            )

        self.fig_tools.save_figure(
            plt_figure=fig,
            fig_name=f"{self.basename}_{self.file_tag['A']}",
        )

        # ASP_fname = self.text_lib["Folders"]["ASPAT_CHECK"]
        # log_file = f"{ASP_fname}/ASpat_log.txt"
        # if os.path.exists(log_file):
        #     os.remove(log_file)
        # if os.path.exists(ASP_fname):
        #     shutil.rmtree(ASP_fname)

        # for i in range(A_Spat.shape[1]):
        #     import tifffile as tif
        #     from skimage.util import img_as_ubyte

        #     comp = A_Spat[:, i]
        #     comp = comp.toarray()
        #     comp = np.transpose(comp.reshape(self.dims))
        #     comp = comp / np.max(comp)
        #     comp = img_as_ubyte(comp)

        #     os.makedirs(ASP_fname, exist_ok=True)

        #     tif.imwrite(
        #         f"{ASP_fname}/Component_{i:03}.tif",
        #         comp,
        #     )

        #     non_zero = np.where(comp > 0)
        #     if len(non_zero[0]) == 0:
        #         continue

        #     # Calculate bounding box
        #     min_y, max_y = np.min(non_zero[0]), np.max(non_zero[0])
        #     min_x, max_x = np.min(non_zero[1]), np.max(non_zero[1])

        #     region = comp[min_y:max_y, min_x:max_x]
        #     # Check corners
        #     corners = [(0, 0), (0, -1), (-1, 0), (-1, -1)]
        #     for y, x in corners:
        #         if region[y, x] == 0:
        #             # Check if surrounded by non-zeros
        #             if y == 0:  # top row
        #                 below = region[1, x] > 0
        #             else:  # bottom row
        #                 below = region[-2, x] > 0

        #             if x == 0:  # left column
        #                 right = region[y, 1] > 0
        #             else:  # right column
        #                 right = region[y, -2] > 0

        #             if below and right:
        #                 region[y, x] = np.max(region)

        #     height = max_y - min_y + 1
        #     width = max_x - min_x + 1

        #     # region = comp[min_y:max_y, min_x:max_x]
        #     # is_perfect = np.all(region > 0)

        #     aspect_ratio = max(height / width, width / height)
        #     msg = f"Component {i:03}: Aspect ratio: {aspect_ratio}"
        #     # print(msg)
        #     with open(log_file, "a") as f:
        #         f.write(msg + "\n")
        #     # if is_perfect:
        #     #     msg = f"|-- Perfect Component {i:03}: {is_perfect}"
        #     #     # print(msg)
        #     #     with open(log_file, "a") as f:
        #     #         f.write(msg + "\n")

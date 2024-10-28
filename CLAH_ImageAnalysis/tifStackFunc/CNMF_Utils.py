import caiman as cm
import matplotlib.pyplot as plt
import numpy as np
from caiman.source_extraction.cnmf import cnmf as cnmf

from CLAH_ImageAnalysis.core import BaseClass as BC
from CLAH_ImageAnalysis.tifStackFunc import TSF_enum


class CNMF_Utils(BC):
    """
    A utility class for performing CNMF (Constrained Non-negative Matrix Factorization)
    on calcium imaging data.

    This class provides methods for initializing CNMF, refining component patches,
    evaluating components, plotting contours, extracting DF/F values, and exporting
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
        method_init: str | None = None,
        meth_deconv: str | None = None,
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
            method_init (str | None, optional): Method for initialization ('greedy', 'sparse_nmf', etc.). Defaults to None.
            meth_deconv (str | None, optional): Method used for spike deconvolution. Defaults to None.
            extract_OR_detrend (str, optional): Whether to extract or detrend the data. Defaults to "extract".
        """
        self.program_name = "CNMF"
        self.class_type = "utils"
        BC.__init__(self, program_name=self.program_name, mode=self.class_type)

        self.CNMFpar = self.enum2dict(TSF_enum.CNMF_Params)
        # printing initiat of CNMF
        print("Initializing CNMF funcs for cell segmentation")
        TSF_enum.export_settings2file(TSF_enum.CNMF_Params, "CNMF")

        # non-default parameters
        self.basename = basename
        self.Ca_Array = Ca_Array
        self.dview = dview
        self.dx = dx
        self.dy = dy
        self.dims = dims
        self.n_processes = n_processes
        self.extract_OR_detrend = extract_OR_detrend

        # CNMF parameters vsee TSF_enum.CNMF_Params
        self.k = self.CNMFpar["K"]
        self.merge_thresh = self.CNMFpar["MERGE_THRESH"]
        self.p = int(self.CNMFpar["P"])
        self.gnb = self.CNMFpar["GNB"]
        self.gSig = self.CNMFpar["GSIG"]
        self.alpha_snmf = int(self.CNMFpar["ALPHA_SNMF"])
        self.only_init_patch = bool(self.CNMFpar["ONLY_INIT_PATCH"])
        self.memory_fact = int(self.CNMFpar["MEMORY_FACT"])
        self.frames = self.CNMFpar["FPS"]
        self.decay_time = self.CNMFpar["DECAY"]
        self.min_SNR = self.CNMFpar["MIN_SNR"]
        self.r_values_min = self.CNMFpar["RVAL_THR"]
        self.use_cnn = bool(self.CNMFpar["USE_CNN"])
        self.thresh_cnn_min = self.CNMFpar["CNN_THR"]
        self.threshold = self.CNMFpar["CE_THRESH"]
        self.vmax = self.CNMFpar["CE_VMAX"]
        self.method_deconvolution = self.CNMFpar["METH_DECONV"]
        self.check_nan = bool(self.CNMFpar["CHECK_NAN"])
        self.quantileMin = self.CNMFpar["QUANTILE_MIN"]
        self.frames_window = self.CNMFpar["FRAME_WINDOW"]

        # CNMF params that can be input via parser
        # see TSF_enum.CNMF_Params for defaults
        self.method_init = (
            self.CNMFpar["METHOD_INIT"] if method_init is None else method_init
        )
        self.method_deconvolution = (
            self.CNMFpar["METH_DECONV"] if meth_deconv is None else meth_deconv
        )

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
            )
            # fit CNMF model to image stack
            self.NonNegMatrix_post_refining = NonNegMatrix_post_refining.fit(
                self.Ca_Array
            )
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
        from caiman.components_evaluation import estimate_components_quality_auto

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

    def createNexport_segDict(
        self,
        concatCheck: bool = False,
        folder_path_concat: str = None,
        folder_path_subj: list = [],
        prev_sd_varnames: bool = False,
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
            prev_sd_varnames (bool, optional): Whether to use previous variable names. Defaults to False.
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
        segFilename = self.basename + self.file_tag["COMP_SDFNAME"]
        filetype_to_save = [
            self.file_tag["H5"],
            self.file_tag["PKL"],
            self.file_tag["MAT"],
        ]
        if concatCheck:
            segFilename = f"{folder_path_concat}/" + segFilename

        SDkey = self.enum2dict(TSF_enum.segDict_Txt)

        if prev_sd_varnames:
            segFilename = (
                self.basename + "_prevNameVar_" + self.file_tag["COMP_SDFNAME"]
            )
            filetype_to_save = [self.file_tag["MAT"]]
            SDkey["A_SPATIAL"] = "A"
            SDkey["C_TEMPORAL"] = "C"
            SDkey["B_BACK_SPAT"] = "b"
            SDkey["F_BACK_TEMP"] = "f"
            SDkey["DFF"] = "dff"
            SDkey["DX"] = "d1"
            SDkey["DY"] = "d2"
            SDkey["S_DECONV"] = "S"

        segDict = {
            SDkey["C_TEMPORAL"]: self.NonNegMatrix_post_refining.estimates.C,
            SDkey["A_SPATIAL"]: self.NonNegMatrix_post_refining.estimates.A,
            SDkey["B_BACK_SPAT"]: self.NonNegMatrix_post_refining.estimates.b,
            SDkey["F_BACK_TEMP"]: self.NonNegMatrix_post_refining.estimates.f,
            SDkey["DFF"]: self.F_dff,
            SDkey["DX"]: self.dx,
            SDkey["DY"]: self.dy,
            SDkey["S_DECONV"]: self.NonNegMatrix_post_refining.estimates.S,
        }
        self.saveNloadUtils.savedict2file(
            dict_to_save=segDict,
            dict_name=dict_name["SD"],
            filename=segFilename,
            date=True,
            filetype_to_save=filetype_to_save,
        )

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
                    + ("_prevNameVar_" if prev_sd_varnames else "")
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

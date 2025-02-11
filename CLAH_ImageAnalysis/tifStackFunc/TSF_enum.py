import os
from enum import Enum
from CLAH_ImageAnalysis import utils


class MOCO_Params(Enum):
    """
    Constants for motion correction:

    - MAX_SHIFTS:
        Maximum allowed rigid shift in pixels during motion correction.
        This determines the range of the search space for alignment between frames
        and restricts how far the algorithm can move the entire image in x- and y-directions.

    - NITER_RIG:
        Number of iterations to perform during rigid motion correction.
        More iterations may improve alignment but increase computation time.

    - SPLITS_RIG:
        Number of chunks to split the movie into for parallel processing during rigid motion correction.
        Splitting the movie allows efficient processing on multi-core systems
        by dividing the workload into manageable segments.

    - STRIDES:
        Distance in pixels to start a new path for patch-wise rigid motion correction.
        This defines the spacing between patches where motion correction is computed.

    - OVERLAPS:
        Overlap in pixels between adjacent patches during patch-wise motion correction.
        The total path size for a patch is the sum of STRIDES and OVERLAPS.
        Higher overlaps can improve accuracy at the cost of increased computation time.

    - SPLITS_ELS:
        Number of chunks to split the movie into for parallel processing during elastic non-rigid motion correction.
        Similar to SPLITS_RIG, this enables efficient handling of non-rigid correction by dividing the movie.

    - UPSAMPLE_FACTOR:
        Factor to upsample frames by during motion correction to improve alignment precision.
        A higher value reduces the chance of smearing when merging corrected patches.

    - MAX_DEV_RIG:
        Maximum deviation in pixels allowed for a patch with respect to rigid shifts.
        This acts as a constraint to prevent patches from moving unrealistically
        far during patch-wise rigid correction.

    - SHIFTS_OPENCV:
        Whether to apply shifts using OpenCV instead of traditional methods.
        OpenCV implementations are typically faster but may result in smoother
        (and potentially less accurate) motion correction results.

    - NONNEG:
        Whether to ensure the saved movie and template remain mostly nonnegative.
        This is achieved by subtracting the minimum value in the movie,
        which is useful for avoiding negative pixel values that can arise
        during filtering or preprocessing.

    - PW_RIGID:
        Whether to use patch-wise rigid registration for motion correction.
        Patch-wise correction divides the frame into smaller patches
        and performs rigid alignment on each, allowing for more localized corrections.

    - USE_CUDA:
        Whether to leverage CUDA (GPU acceleration) for motion correction.
        CUDA can significantly speed up the process on compatible hardware.

    - GSIG_FILT:
        Standard deviation (filter size) for the high-pass filter used in motion correction.
        High-pass filtering helps remove low-frequency noise or background variations,
        enhancing the ability to detect and correct motion.
    """

    MAX_SHIFTS = 30  # maximum allowed rigid shift
    NITER_RIG = 1  # number of iterations for rigid motion correction
    SPLITS_RIG = 50  # for parallelization splits movies in num_splits chunks over time (rigid motion correction)
    STRIDES = 48  # start new path for pw-rigid motion correction every X pxls
    OVERLAPS = 32  # overlap btw paths (size of path strides + overlaps)
    SPLITS_ELS = 56  # for parallelization split movies in num_split chunks across time (elastic non-rigid motion correction)
    UPSAMPLE_FACTOR = 50  # upsample factor to avoid smearing when merging patches
    MAX_DEV_RIG = 3  # max deviation allowed for patch wrt rigid shifts
    SHIFTS_OPENCV = True  # applying shifts in a fast way (but smoothing results)
    NONNEG = True  # make saved movie & template mostly nonnegative by removing min_mov from movie
    BORDER_NAN = True  # how to handle NaNs at the border
    PW_RIGID = True  # use patch-wise rigid registration
    USE_CUDA = True  # use cuda for registration
    GSIG_FILT = None  # filter size for high-pass filter


class MOCO_Params4OnePhoton(Enum):
    """
    Constants for motion correction for one-photon data:
    """

    MAX_SHIFTS = 20
    NITER_RIG = 1
    SPLITS_RIG = 50
    SPLITS_ELS = 56
    STRIDES = 50
    OVERLAPS = 28
    UPSAMPLE_FACTOR = 50
    MAX_DEV_RIG = 10
    SHIFTS_OPENCV = True
    NONNEG = True
    BORDER_NAN = True
    PW_RIGID = True
    USE_CUDA = True
    GSIG_FILT = None  # filter size for high-pass filter


# GLOBAL VAR of OPTS for CNMF_Params
CNMF_OPTS = {
    "METHOD_INIT": ["greedy_roi", "sparse_nmf", "pca_ica", "corr_pnr"],
    "DECONV_METHOD": ["oasis", "cvxpy", "mcmc"],
}


class CNMF_Params(Enum):
    """
    Constants for CNMF (Constrained Nonnegative Matrix Factorization):
    - P: Order of the autoregressive system.
    - GNB: Number of global background components.
    - GSIG: Expected half-size of neurons in pixels.
    - MERGE_THRESH: Threshold for merging components based on correlation.
    - INIT_METH: Method for initialization ("greedy_roi").
    - ISDENDRITES: Whether the data includes dendrites.
    - ALPHA_SNMF: Weight of the sparsity regularization term.
    - MIN_SNR: Minimum signal-to-noise ratio for a component to be accepted.
    - RVAL_THR: Threshold for the correlation value used in component evaluation.
    - CNN_THR: Threshold for the CNN-based component evaluation.
    - FPS: Imaging rate in frames per second.
    - K: Number of components per patch.
    - RF: Half-size of patches in pixels.
    - STRIDE: Amount of overlap between patches in pixels.
    - MEMORY_FACT: Factor determining how much memory should be used.
    - NPROC: Number of processes to use for parallel computing.
    - DECAY: Time constant for the exponential decay in the autoregressive model.
    - METH_DECONV: Method for deconvolution ("oasis").
    - CHECK_NAN: Whether to check for NaNs in the data.
    - USE_CNN: Whether to use a CNN for component evaluation.
    """

    P = 1  # 2  # order of autoregressive system
    GNB = 2  # 3  # number of global background components
    NB_PATCH = 1  # number of background components per patch
    GSIG = [4, 4]  # expected half size of neurons
    GSIZ = None
    MERGE_THRESH = 0.7  # 0.5  # merging threshold / max correlation allowed
    #! initialization method; options: greedy_roi [0], sparse_nmf [1], pca_ica[2], corr_pnr[3]
    #! see CNMF_OPTS above
    METHOD_INIT = CNMF_OPTS["METHOD_INIT"][0]
    ISDENDRITES = False  # Does data include dendrites?
    ALPHA_SNMF = 0  # Sparsity penalty / weight of sparsity regularization
    MIN_SNR = 4  # 6  # signal to noise ratio for accepting component
    RVAL_THR = 0.8  # threshold for correlation value used in component evaluation
    CNN_THR = 0.7  # 0.8  # threshold used to determine if a component should be kept
    CENTER_PSF = False  # whether to center the PSF

    FPS = 10  # imaging rate in fps
    K = 15  # 10  # 20 # number of components per patch
    RF = 25  # 15  # half size f patches in pxls (ie 25 = 50x50)
    STRIDE = 8  # 6  # amount of overlap btw patches in pxls
    MEMORY_FACT = 1  # how much memory should be used
    DECAY = 4  # time constant for exponential decay in autoregressive model
    #! method for deconvolution; options: oasis[0], cvxpy[1], mcmc[2]
    #! see CNMF_OPTS above
    METH_DECONV = CNMF_OPTS["DECONV_METHOD"][0]
    CHECK_NAN = True  # check for NaNs in the data
    USE_CNN = False  # use CNN for component evaluation
    SPIKE_MIN = None  # minimum spike size
    MIN_CORR = 0.85  # min peak value from correlation image
    LOW_RANK_BACKGROUND = True  # whether to keep background of each patch intact, True performs low-rank approximation if gnb > 0
    MIN_PNR = 20  # minimum peak to noise ratio from PNR image

    ONLY_INIT_PATCH = False  # only run initialization on patches

    CE_THRESH = 0.8  # threshold for component evaluation
    CE_VMAX = 0.75  # max value for component evaluation

    QUANTILE_MIN = 8  # minimum quantile to be used in thrsholding operations
    FRAME_WINDOW = 250  # number of frames to consider in a sliding window


class CNMF_Params_1p(Enum):
    """
    Constants for CNMF for one-photon data:
    """

    CENTER_PSF = False
    CNN_THR = 0.7
    FPS = 10  #! NEED TO CHECK THIS
    GNB = 0
    GSIG = [2, 2]
    GSIZ = None
    K = None
    LOW_RANK_BACKGROUND = False
    MERGE_THRESH = 0.65
    METHOD_INIT = CNMF_OPTS["METHOD_INIT"][0]
    MIN_CORR = 0.95
    MIN_PNR = 15
    MIN_SNR = 20
    NB_PATCH = 0
    RF = 40
    RVAL_THR = 0.85
    SPIKE_MIN = -15


class segDict_Txt(Enum):
    """
    Constants for segmentation dictionary:
    - A_SPATIAL: Key for the spatial footprint of each component.
    - C_TEMPORAL: Key for the temporal trace of each component.
    - B_BACK: Key for the background of each component.
    - DFF: Key for the delta F/F of each component.
    - DX: Key for the x-shifts for motion correction.
    - DY: Key for the y-shifts for motion correction.
    - S_DECONV: Key for the deconvolved spike trains of each component.
    """

    A_SPATIAL = "A_Spatial"
    C_TEMPORAL = "C_Temporal"
    B_BACK_SPAT = "B_Back_Spat"
    F_BACK_TEMP = "F_Back_Temp"
    DFF = "dff"
    DX = "dx"
    DY = "dy"
    S_DECONV = "S_Deconvolution"
    YRA = "YrA_TempResidual"
    RSR = "R_SpatialResidual"


class Parser4M2SD(Enum):
    """
    Enumeration class that defines various parameters for the parser used for Moco2segDict (motion correction & segmentation analysis)

    Attributes:
        PARSER (str): Name of the parser.
        HEADER (str): Header for the analysis.
        PARSER4 (str): What parser is this for.
        ARG_NAME_LIST (List[Tuple[str, str]]): List of argument names and their short forms.
        HELP_TEXT_LIST (List[str]): List of help text descriptions for the arguments.
        DEFAULT_LIST (List[bool]): List of default values for the arguments.
        TYPE_LIST (List[str]): List of types for the arguments.
    """

    PARSER = "Parser"
    HEADER = "Motion Correction & Segmentation Analysis"
    PARSER4 = "TSF"
    PARSER_FN = "Moco2segDict"
    ARG_DICT = {
        ("motion_correct", "mc"): {
            "TYPE": "bool",
            "DEFAULT": False,
            "HELP": "Whether to perform motion correction. Default is false. (e.g. -mc y, -mc yes, -mc true to enable)",
        },
        ("segment", "sg"): {
            "TYPE": "bool",
            "DEFAULT": False,
            "HELP": "Whether to perform segmentation. Default is false. (e.g. -seg y, -seg yes, -seg true to enable)",
        },
        ("n_proc4MOCO", "n4mc"): {
            "TYPE": "int",
            "DEFAULT": 26,
            "HELP": "How many processors to use for motion correction. Default is 26 processes",
        },
        ("n_proc4CNMF", "n4cnmf"): {
            "TYPE": "int",
            "DEFAULT": None,
            "HELP": "How many processors to use for CNMF segmentation. Default is using all available processes.",
        },
        ("concatenate", "cat"): {
            "TYPE": "bool",
            "DEFAULT": False,
            "HELP": "Concatenate H5s into a single H5 before motion correction, but create 2 segDicts. ONLY USE THIS TO COMBINE THE RESULTS FOR THE SAME SUBJECT ID ACROSS 2 SESSIONS.",
        },
        ("prev_sd_varnames", "psv"): {
            "TYPE": "bool",
            "DEFAULT": False,
            "HELP": "Use the old variable names for the segDict (i.e. A, C, S, etc). Default is False, in which names will be A_Spatial, C_Temporal, etc.",
        },
        ("mc_iter", "mci"): {
            "TYPE": "int",
            "DEFAULT": 1,
            "HELP": "Number of iterations for motion correction. Default is 1. WARNING: this is not the same as the number of iterations for rigid motion correction (niter_rig) within caiman and it can add to the total processing time.",
        },
        ("overwrite", "ow"): {
            "TYPE": "bool",
            "DEFAULT": False,
            "HELP": "Overwrite existing files (segDicts, sqz_H5s, tifs, mmaps, etc). Default is False.",
        },
        ("compute_metrics", "cm"): {
            "TYPE": "bool",
            "DEFAULT": False,
            "HELP": "Calculate motion correction metrics. Default is False.",
        },
        ("use_cropper", "crp"): {
            "TYPE": "bool",
            "DEFAULT": False,
            "HELP": "Use the cropping utility for 1photon data (.isxd files). Default is False.",
        },
        ("separate_channels", "sc"): {
            "TYPE": "bool",
            "DEFAULT": False,
            "HELP": "Whether to motion correct channels separately. Only applicable for 2photon data with 2 channels. Default is False.",
        },
    }


def export_settings2file(enum_class: Enum, ptype: str, inFolder: bool = True) -> None:
    """
    Export the parameters from the given enum class to a file.

    Parameters:
        enum_class (Enum): The enum class containing the parameters to be exported.
        ptype (str): The type of parameters to be exported.
    """

    dict_from_enum = utils.enum_utils.enum2dict(enum_class)
    utils.print_wFrame(f"Parameters to be applied for {ptype}")
    file_tag = utils.text_dict()["file_tag"]

    # ftags = [file_tag["JSON"], file_tag["TXT"]]
    ftags = [file_tag["JSON"]]

    if inFolder:
        PARAMS_FOLDER = utils.text_dict()["Folders"]["PARAMS"]
        os.makedirs(PARAMS_FOLDER, exist_ok=True)
    else:
        PARAMS = file_tag["PARAMS"]

    for key, value in dict_from_enum.items():
        utils.print_wFrame(f"{key}: {value}", frame_num=1)

    utils.print_wFrame("Exporting Parameters to file")
    for ftag2save in ftags:
        if inFolder:
            fname2save = f"{PARAMS_FOLDER}/{ptype}"
        else:
            fname2save = f"{PARAMS}_{ptype}"
        utils.enum_utils.export_param_from_enum(
            enum_class, fname2save, file_type=ftag2save
        )
    print()

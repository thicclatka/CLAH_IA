from enum import Enum

from CLAH_ImageAnalysis import utils


class MOCO_Params(Enum):
    """
    Constants for motion correction:
    - MAX_SHIFTS: Maximum allowed rigid shift in pixels.
    - NITER_RIG: Number of iterations for rigid motion correction.
    - SPLITS_RIG: Number of chunks to split the movie into for parallel processing during rigid motion correction.
    - STRIDES: Distance in pixels to start a new path for patch-wise rigid motion correction.
    - OVERLAPS: Overlap in pixels between paths (total path size is STRIDES + OVERLAPS).
    - SPLITS_ELS: Number of chunks to split the movie into for parallel processing during elastic non-rigid motion correction.
    - UPSAMPLE_FACTOR: Factor to upsample by to avoid smearing when merging patches.
    - MAX_DEV_RIG: Maximum deviation in pixels allowed for a patch with respect to rigid shifts.
    - SHIFTS_OPENCV: Whether to apply shifts using OpenCV (faster but may result in smoother results).
    - NONNEG: Whether to make the saved movie and template mostly nonnegative by subtracting the minimum movie value.
    - PW_RIGID: Whether to use patch-wise rigid registration.
    - USE_CUDA: Whether to use CUDA for registration.
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

    MAX_SHIFTS = 30
    NITER_RIG = 1
    SPLITS_RIG = 56
    SPLITS_ELS = 56
    STRIDES = 42
    OVERLAPS = 32
    UPSAMPLE_FACTOR = 50
    MAX_DEV_RIG = 3
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
    GSIG = [4, 4]  # expected half size of neurons
    MERGE_THRESH = 0.7  # 0.5  # merging threshold / max correlation allowed
    #! initialization method; options: greedy_roi [0], sparse_nmf [1], pca_ica[2], corr_pnr[3]
    #! see CNMF_OPTS above
    METHOD_INIT = CNMF_OPTS["METHOD_INIT"][0]
    ISDENDRITES = False  # Does data include dendrites?
    ALPHA_SNMF = 0  # Sparsity penalty / weight of sparsity regularization
    MIN_SNR = 4  # 6  # signal to noise ratio for accepting component
    RVAL_THR = 0.8  # threshold for correlation value used in component evaluation
    CNN_THR = 0.7  # 0.8  # threshold used to determine if a component should be kept
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

    ONLY_INIT_PATCH = False  # only run initialization on patches

    CE_THRESH = 0.8  # threshold for component evaluation
    CE_VMAX = 0.75  # max value for component evaluation

    QUANTILE_MIN = 8  # minimum quantile to be used in thrsholding operations
    FRAME_WINDOW = 250  # number of frames to consider in a sliding window


class CNMF_Params_1p(Enum):
    """
    Constants for CNMF for one-photon data:
    """

    MIN_SNR = 4
    CNN_THR = 0.7


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
    }


def export_settings2file(enum_class: Enum, ptype: str) -> None:
    """
    Export the parameters from the given enum class to a file.

    Parameters:
        enum_class (Enum): The enum class containing the parameters to be exported.
        ptype (str): The type of parameters to be exported.
    """

    dict_from_enum = utils.enum_utils.enum2dict(enum_class)
    utils.print_wFrame(f"Parameters to be applied for {ptype}")
    file_tag = utils.text_dict()["file_tag"]
    for key, value in dict_from_enum.items():
        utils.print_wFrame(f"{key}: {value}", frame_num=1)

    utils.print_wFrame("Exporting Parameters to file")
    for ftag2save in [file_tag["JSON"], file_tag["TXT"]]:
        utils.enum_utils.export_param_from_enum(
            enum_class, f"{ptype}_Parameters", file_type=ftag2save
        )
    print()

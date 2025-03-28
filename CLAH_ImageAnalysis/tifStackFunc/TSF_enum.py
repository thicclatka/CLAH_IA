import os
from enum import Enum
from CLAH_ImageAnalysis import utils

# GLOBAL VAR of OPTS for CNMF_Params
CNMF_OPTS = {
    "METHOD_INIT": ["greedy_roi", "sparse_nmf", "pca_ica", "corr_pnr"],
    "METH_DECONV": ["oasis", "cvxpy", "mcmc"],
}
MOCO_OPTS = {}


class MOCO_PARAMS(Enum):
    """
    Constants for motion correction. Each tuple contains:
    (2p_default, 1p_default, help_string)
    """

    MAX_SHIFTS = (
        30,
        20,
        "Maximum allowed rigid shift in pixels during motion correction. This determines the range of the search space for alignment between frames and restricts how far the algorithm can move the entire image in x- and y-directions",
    )
    NITER_RIG = (
        1,
        1,
        "Number of iterations for rigid motion correction. More iterations may improve alignment but increase computation time",
    )
    SPLITS_RIG = (
        50,
        50,
        "Number of chunks to split movie for parallel processing (rigid). Splitting the movie allows efficient processing on multi-core systems by dividing the workload into manageable segments",
    )
    STRIDES = (
        48,
        50,
        "Distance in pixels between patches for motion correction. This defines the spacing between patches where motion correction is computed",
    )
    OVERLAPS = (
        32,
        28,
        "Overlap in pixels between adjacent patches. The total path size for a patch is the sum of STRIDES and OVERLAPS. Higher overlaps can improve accuracy at the cost of increased computation time",
    )
    SPLITS_ELS = (
        56,
        56,
        "Number of chunks to split movie for elastic correction. Similar to SPLITS_RIG, this enables efficient handling of non-rigid correction by dividing the movie",
    )
    UPSAMPLE_FACTOR = (
        50,
        50,
        "Factor to upsample frames for better precision. A higher value reduces the chance of smearing when merging corrected patches",
    )
    MAX_DEV_RIG = (
        3,
        10,
        "Maximum deviation allowed for patches vs rigid shifts. This acts as a constraint to prevent patches from moving unrealistically far during patch-wise rigid correction",
    )
    SHIFTS_OPENCV = (
        True,
        True,
        "Use OpenCV for faster (but smoother) corrections. OpenCV implementations are typically faster but may result in smoother (and potentially less accurate) motion correction results",
    )
    NONNEG = (
        True,
        True,
        "Ensure movie and template remain mostly nonnegative by subtracting the minimum value in the movie. This is useful for avoiding negative pixel values that can arise during filtering or preprocessing",
    )
    BORDER_NAN = (
        True,
        True,
        "How to handle NaNs at the border. If True, NaNs are replaced with the nearest valid pixel value. If False, NaNs are left unchanged",
    )
    PW_RIGID = (
        True,
        True,
        "Use patch-wise rigid registration. Patch-wise correction divides the frame into smaller patches and performs rigid alignment on each, allowing for more localized corrections",
    )
    USE_CUDA = (True, True, "Use CUDA GPU acceleration. Default is True")
    GSIG_FILT = (
        None,
        None,
        "Standard deviation for high-pass filter (None=disabled). High-pass filtering helps remove low-frequency noise or background variations, enhancing the ability to detect and correct motion",
    )


class CNMF_PARAMS(Enum):
    """
    Constants for CNMF (Constrained Nonnegative Matrix Factorization).
    Each tuple contains:
    (2p_default, 1p_default, help_string)
    """

    P = (
        1,
        1,
        "Order of the autoregressive system. Higher values mean more variables to model with and a higher risk of overfitting",
    )
    GNB = (
        2,
        2,
        "Number of global background components. Higher values mean more background components to model with",
    )
    NB_PATCH = (1, 1, "Number of background components per patch")
    GSIG = (
        4,
        2,
        "Expected half-size of neurons in pixels. Affects the width of a 2D gaussian kernel used to model neuron size",
    )
    GSIZ = (
        None,
        None,
        "Average diameter of neurons in pixels. Default is None, which means the size is determined automatically based on gSig (GSIG)",
    )
    MERGE_THRESH = (
        0.7,
        0.7,
        "Threshold for merging components based on correlation. Higher values mean stricter merging criteria",
    )
    METHOD_INIT = (
        CNMF_OPTS["METHOD_INIT"][0],
        CNMF_OPTS["METHOD_INIT"][0],
        "Method for initialization. Options: greedy_roi - uses a greedy approach; sparse_nmf - uses sparse non-negative matrix factorization; pca_ica - uses PCA and ICA; corr_pnr - uses correlation-based peak-to-noise ratio",
    )
    ISDENDRITES = (False, False, "Whether the data includes dendrites")
    ALPHA_SNMF = (
        0,
        0,
        "Weight of sparsity regularization term. Higher values enforce more sparsity in the components",
    )
    MIN_SNR = (
        4,
        4,
        "Minimum signal-to-noise ratio for accepting a component. Higher values mean more stringent filtering",
    )
    RVAL_THR = (
        0.8,
        0.85,
        "Threshold for correlation value used in component evaluation. Higher values mean more stringent filtering",
    )
    CNN_THR = (
        0.7,
        0.7,
        "Threshold for CNN-based component evaluation. Higher values mean more stringent filtering",
    )
    CENTER_PSF = (
        False,
        False,
        "Whether to center the PSF (Point Spread Function). If True, the PSF is centered on the image",
    )
    FPS = (10, 10, "Imaging rate in frames per second")
    K = (
        15,
        20,
        "Number of components per patch. If you observe a high density of components, you can increase this value",
    )
    RF = (25, 40, "Half-size of patches in pixels (e.g., 25 = 50x50)")
    STRIDE = (8, 20, "Amount of overlap between patches in pixels")
    MEMORY_FACT = (
        1,
        1,
        "Factor determining memory usage. Default is 1, which works for most cases with 16GB RAM",
    )
    DECAY = (4, 4, "Time constant for exponential decay in autoregressive model")
    METH_DECONV = (
        CNMF_OPTS["METH_DECONV"][0],
        CNMF_OPTS["METH_DECONV"][0],
        "Method for deconvolution. Options: oasis - fastest and most accurate; cvxpy - slower but accurate; mcmc - slowest but most accurate",
    )
    CHECK_NAN = (True, True, "Whether to check for NaNs in the data")
    USE_CNN = (False, False, "Whether to use CNN for component evaluation")
    SPIKE_MIN = (None, None, "Minimum spike size threshold")
    MIN_CORR = (
        0.85,
        0.95,
        "Minimum peak value from correlation image. Higher values mean more stringent filtering",
    )
    LOW_RANK_BACKGROUND = (
        True,
        True,
        "Whether to keep background of each patch intact. True performs low-rank approximation if gnb > 0",
    )
    MIN_PNR = (
        20,
        15,
        "Minimum peak to noise ratio from PNR image. Higher values mean more stringent filtering",
    )
    ONLY_INIT_PATCH = (False, False, "Whether to only run initialization on patches")
    CE_THRESH = (
        0.8,
        0.8,
        "Threshold for component evaluation. Higher values mean more stringent filtering",
    )
    CE_VMAX = (
        0.75,
        0.75,
        "Maximum value for component evaluation. Higher values mean more stringent filtering",
    )
    QUANTILE_MIN = (
        8,
        8,
        "Minimum quantile for thresholding operations. Higher values mean more stringent filtering",
    )
    FRAME_WINDOW = (
        250,
        250,
        "Number of frames to consider in a sliding window. Higher values mean more frames are considered",
    )


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

    ACCEPTED_LABELS = "accepted_labels"
    A_SPATIAL = "A"
    A_SPATIAL_ALL = "A_all"
    B_BACK_SPAT = "b"
    C_TEMPORAL = "C"
    C_TEMPORAL_ALL = "C_all"
    DFF = "dff"
    DX = "d1"
    DY = "d2"
    F_BACK_TEMP = "f"
    IDX_BAD = "idx_components_bad"
    IDX_GODO = "idx_components"
    RSR = "R"
    S_DECONV = "S"
    YRA = "YrA"
    SNR_COMP = "SNR_comp"
    R_VALUES = "r_values"
    CNN_PREDS = "cnn_preds"


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
        # ("prev_sd_varnames", "psv"): {
        #     "TYPE": "bool",
        #     "DEFAULT": False,
        #     "HELP": "Use the old variable names for the segDict (i.e. A, C, S, etc). Default is False, in which names will be A_Spatial, C_Temporal, etc.",
        # },
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
        ("export_postseg_residuals", "ers"): {
            "TYPE": "bool",
            "DEFAULT": False,
            "HELP": "Whether to export the post-segmentation residuals as a video file. Default is False.",
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


def generateNexport_user_set_params(
    onePhotonCheck: bool, param_type: str = "MC", **kwargs
) -> None:
    """
    Generate and export parameters for motion correction or CNMF.

    Parameters:
        onePhotonCheck (bool): Whether the session is a one-photon session.
        param_type (str): Type of parameters to generate ("MC" or "CNMF").
        **kwargs: Keyword arguments to override the default values.

    Returns:
        None: Saves the parameters to a JSON file.
    """
    # Select the appropriate enum and filename based on param_type
    if param_type == "MC":
        reference_enum = MOCO_PARAMS
        fname = utils.text_dict()["file_tag"]["USER_MC_PARAMS"]
    elif param_type == "CNMF":
        reference_enum = CNMF_PARAMS
        fname = utils.text_dict()["file_tag"]["USER_CNMF_PARAMS"]
    else:
        raise ValueError("param_type must be either 'MC' or 'CNMF'")

    # Get the dictionary from the enum
    reference_dict = utils.enum_utils.enum2dict(reference_enum)

    param_keys = list(reference_dict.keys())
    param_keys.sort()
    par_idx = 0 if not onePhotonCheck else 1

    # Generate parameters dictionary
    params = {key: kwargs.get(key, reference_dict[key][par_idx]) for key in param_keys}

    # Save to JSON
    utils.enum_utils.create_json_from_enumDict(params, fname, indent=4)
    return

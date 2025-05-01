from enum import Enum


class Parser4TOD(Enum):
    """
    Enumeration class that defines various parameters for the parser used for quickTuning

    Attributes:
        PARSER (str): Name of the parser.
        HEADER (str): Header for the analysis.
        PARSER4 (str): What parser is this for.
        ARG_NAME_LIST (List[Tuple[str, str]]): List of argument names and their short forms.
        HELP_TEXT_LIST (List[str]): List of help text descriptions for the arguments.
        DEFAULT_LIST (List[bool]): List of default values for the arguments.
        TYPE_LIST (List[str]): List of types for the arguments.
    """

    PARSER = "Parser4TOD"
    HEADER = "Two Odor Decoder"
    PARSER4 = "decoder"
    PARSER_FN = "Two Odor Decoder"
    ARG_DICT = {
        ("num_folds", "f"): {
            "TYPE": "int",
            "DEFAULT": 10,
            "HELP": "Number of folds for cross-validation. Default is 10.",
        },
        ("null_repeats", "n"): {
            "TYPE": "int",
            "DEFAULT": 10,
            "HELP": "Number of null repeats for shuffling data. Default is 10.",
        },
        ("useZscore", "z"): {
            "TYPE": "bool",
            "DEFAULT": True,
            "HELP": "Whether to use Z-score normalization on Temporal Data. Default is True.",
        },
        ("decoder_type", "dt"): {
            "TYPE": "str",
            "DEFAULT": None,
            "HELP": "Type of decoder to use. Options are 'SVC', 'GBM', or 'LSTM'. Default is None, which will prompt a user selection. SVC = Support Vector Classifier; GBM = Gradient Boosting Machine; LSTM = Long Short-Term Memory (NN)",
        },
        ("cost_param", "c"): {
            "TYPE": "float",
            "DEFAULT": None,
            "HELP": "Cost parameter for the SVM. Default is None, which prompts hyperparameter tuning to find the best C. (FOR SVC)",
        },
        ("kernel_type", "k"): {
            "TYPE": "str",
            "DEFAULT": "rbf",
            "HELP": "Type of kernel to use for the SVM. Default is 'rbf' or a radial base function. (FOR SVC)",
        },
        ("gamma", "g"): {
            "TYPE": "float|str",
            "DEFAULT": "scale",
            "HELP": "Gamma parameter for the SVM. Options are 'auto', 'scale', or float. Default is 'scale', which is 1 / (n_features * X.var()). 'auto' uses 1 / n_features. For float, must be non-negative. (FOR SVC)",
        },
        ("weight", "w"): {
            "TYPE": "str",
            "DEFAULT": None,
            "HELP": "Weight parameter for SVC; sets the parameter C of class i to class i_weight * C for SVC. Default is None. (FOR SVC)",
        },
        ("n_estimators", "e"): {
            "TYPE": "int",
            "DEFAULT": 100,
            "HELP": "Number of estimators. Default is 100. This is the number of boosting stages to be run. (FOR GBM)",
        },
        ("max_depth", "d"): {
            "TYPE": "int",
            "DEFAULT": 3,
            "HELP": "Maximum depth of the individual regression estimators. Default is 3. (FOR GBM)",
        },
        ("learning_rate", "l"): {
            "TYPE": "float",
            "DEFAULT": 0.1,
            "HELP": "Learning rate; how much to shrink contribution of each tree. Default is 0.1. (FOR GBM)",
        },
    }
    EXCEPTION = {
        "SVC": ["C", "kernel_type", "gamma", "weight"],
        "GBM": ["n_estimators", "max_depth", "learning_rate"],
        "LSTM": [],
    }


class Parser4TOD_DEV(Enum):
    """
    Enumeration class that defines various parameters for the parser used for quickTuning

    Attributes:
        PARSER (str): Name of the parser.
        HEADER (str): Header for the analysis.
        PARSER4 (str): What parser is this for.
        ARG_NAME_LIST (List[Tuple[str, str]]): List of argument names and their short forms.
        HELP_TEXT_LIST (List[str]): List of help text descriptions for the arguments.
        DEFAULT_LIST (List[bool]): List of default values for the arguments.
        TYPE_LIST (List[str]): List of types for the arguments.
    """

    PARSER = "Parser4TOD"
    HEADER = "Two Odor Decoder"
    PARSER4 = "decoder"
    PARSER_FN = "Two Odor Decoder"
    ARG_DICT = {
        ("num_folds", "f"): {
            "TYPE": "int",
            "DEFAULT": 10,
            "HELP": "Number of folds for cross-validation. Default is 10.",
        },
        ("null_repeats", "n"): {
            "TYPE": "int",
            "DEFAULT": 10,
            "HELP": "Number of null repeats for shuffling data. Default is 10.",
        },
        ("cost_param", "c"): {
            "TYPE": "float",
            "DEFAULT": None,
            "HELP": "Cost parameter for the SVM. Default is None, which prompts hyperparameter tuning to find the best C. (FOR SVC)",
        },
        ("kernel_type", "k"): {
            "TYPE": "str",
            "DEFAULT": "rbf",
            "HELP": "Type of kernel to use for the SVM. Default is 'rbf' or a radial base function. (FOR SVC)",
        },
        ("gamma", "g"): {
            "TYPE": "float|str",
            "DEFAULT": "scale",
            "HELP": "Gamma parameter for the SVM. Options are 'auto', 'scale', or float. Default is 'scale', which is 1 / (n_features * X.var()). 'auto' uses 1 / n_features. For float, must be non-negative. (FOR SVC)",
        },
        ("weight", "w"): {
            "TYPE": "str",
            "DEFAULT": "balanced",
            "HELP": "Weight parameter for SVC; sets the parameter C of class i to class i_weight * C for SVC. Default is None. (FOR SVC)",
        },
        ("n_estimators", "e"): {
            "TYPE": "int",
            "DEFAULT": 100,
            "HELP": "Number of estimators. Default is 100. This is the number of boosting stages to be run. (FOR GBM)",
        },
        ("max_depth", "d"): {
            "TYPE": "int",
            "DEFAULT": 3,
            "HELP": "Maximum depth of the individual regression estimators. Default is 3. (FOR GBM)",
        },
        ("learning_rate", "l"): {
            "TYPE": "float",
            "DEFAULT": 0.1,
            "HELP": "Learning rate; how much to shrink contribution of each tree. Default is 0.1. (FOR GBM)",
        },
    }

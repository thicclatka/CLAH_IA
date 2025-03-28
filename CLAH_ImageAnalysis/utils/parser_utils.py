import argparse
import os
from typing import Any
from CLAH_ImageAnalysis.utils import print_header
from CLAH_ImageAnalysis.utils import print_wFrame
from CLAH_ImageAnalysis.utils import text_dict


def _valid_path(path: str, out: bool = False) -> str | list:
    """
    Validate a path and return it if it exists.

    Parameters:
        path (str): The path to validate.
        out (bool, optional): Whether to create the directory if it doesn't exist. Defaults to False.

    Returns:
        str | list: The validated path.
    """

    if not path:
        return []
    if not os.path.exists(path) and not out:
        raise argparse.ArgumentTypeError(f"Path {path} does not exist.")
    if not os.path.exists(path) and out:
        os.makedirs(path)
    if not os.path.isdir(path):
        raise argparse.ArgumentTypeError(f"Path {path} is not a directory.")
    return path


def float_or_str(input_value: str) -> float | str:
    """
    Convert a string to a float if possible, otherwise return the string.

    Parameters:
        input_value (str): The string to convert.

    Returns:
        float | str: The converted value or the original string if conversion fails.
    """
    try:
        return float(input_value)
    except ValueError:
        return input_value


def _init_parser(desc: str) -> argparse.ArgumentParser:
    """
    Create an argument parser with the given description.

    Parameters:
        desc (str): The description for the argument parser.

    Returns:
        argparse.ArgumentParser: The created argument parser.
    """
    parser = argparse.ArgumentParser(description=desc)

    parser.allow_unknown_args = True
    return parser


def _add_argument(
    parser: argparse.ArgumentParser,
    arg_name: str,
    help_text: str,
    default_value: Any,
    type_func: bool = True,
) -> None:
    """
    Add an argument to the given argument parser.

    Parameters:
        parser (argparse.ArgumentParser): The argument parser to add the argument to.
        arg_name (str): The name of the argument.
        help_text (str): The help text for the argument.
        default_value (Any): The default value for the argument.
        type_func (bool, optional): Whether to use a type function for the argument. Defaults to True.
    """

    text_lib = text_dict()

    def tf2add_bool(x: str) -> bool:
        """
        Convert a string to a boolean.

        Parameters:
            x (str): The string to convert.

        Returns:
            bool: The converted boolean value.
        """
        return str(x).lower() in text_lib["YES_RESP"]

    def tf2add_path4out(path: str) -> str | list:
        """
        Validate a path and return it if it exists, creating it if out is True.

        Parameters:
            path (str): The path to validate.

        Returns:
            str | list: The validated path.
        """
        return _valid_path(path, out=True)

    def tf2addList(range_str: str) -> list | str:
        """
        Parses a string containing numbers, ranges, or 'all'/'ALL' into a list of integers or returns 'all'.
        Example input: '[21-45,47]', 'all', 'ALL'
        Example output for '[21-45,47]': [21, 22, ..., 45, 47]
        Example output for 'all' or 'ALL': 'all'

        Parameters:
            range_str (str): The string to parse.

        Returns:
            list | str: The parsed list of integers or 'all'.
        """

        if range_str.lower() == "all":
            return "all"
        else:
            # Remove square brackets and split by comma
            items = range_str.strip("[]").split(",")
            result = []
            for item in items:
                if "-" in item:  # It's a range
                    start, end = item.split("-")
                    result.extend(range(int(start), int(end) + 1))
                else:  # It's a single number
                    result.append(int(item))
            return result

    if type_func == "bool":
        tf2add = tf2add_bool
    elif type_func == "list":
        tf2add = tf2addList
    elif type_func == "path":
        tf2add = _valid_path
    elif type_func == "out_path":
        tf2add = tf2add_path4out
    elif type_func == "int":
        tf2add = int
    elif type_func == "float|str" or type_func == "str|float":
        tf2add = float_or_str
    else:
        tf2add = None
    if isinstance(arg_name, tuple):
        parser.add_argument(
            f"-{arg_name[1]}",
            f"--{arg_name[0]}",
            type=tf2add,
            default=default_value,
            help=help_text,
        )
    else:
        parser.add_argument(
            f"--{arg_name}",
            type=tf2add,
            default=default_value,
            help=help_text,
        )


def createParserNextractArgs(
    parser_enum: object, clear_terminal: bool = True
) -> argparse.Namespace:
    """
    Create a parser and extract arguments from the given parser enum.
    """
    os.system("clear") if clear_terminal else None

    header_txt = parser_enum.HEADER.value
    enum_name = parser_enum.PARSER.value
    parser4 = parser_enum.PARSER4.value
    arg_dict = parser_enum.ARG_DICT.value
    parserFullName = parser_enum.PARSER_FN.value

    arg_name_list = []
    help_text_list = []
    default_list = []
    type_list = []

    for key, val in arg_dict.items():
        arg_name_list.append(key)
        help_text_list.append(val["HELP"])
        default_list.append(val["DEFAULT"])
        type_list.append(val["TYPE"])

    print_header(header_txt)
    parser = _init_parser(desc=f"Run {parserFullName} with command line arguments.")

    # Prepend "path" and "Path to the data directory" to the lists
    arg_name_list.insert(0, ("path", "p"))
    help_text_list.insert(
        0,
        "Path to the data directory (e.g. /path/to/data). Default will prompt user to choose path. NOTE: Pick directory which holds session folders; do not pick/set a session folder.",
    )
    default_list.insert(0, [])
    type_list.insert(0, "path")

    arg_name_list.insert(1, ("sess2process", "s2p"))
    help_text_list.insert(
        1,
        "List of sessions to process. Write in format '1,2,3', '1-3', or '1,2-5' to select by specific session number. Input all or ALL to process all eligible sessions that are available within the set path. Default will prompt user to choose.",
    )
    default_list.insert(1, [])
    type_list.insert(1, "list")

    for arg_name, help_text, default, ptype in zip(
        arg_name_list, help_text_list, default_list, type_list
    ):
        _add_argument(
            parser,
            arg_name=arg_name,
            help_text=help_text,
            default_value=default,
            type_func=ptype,
        )
    args, unknown = parser.parse_known_args()

    if hasattr(args, "decoder_type"):
        ignored_params = []
        if args.decoder_type == "SVC":
            ignored_params = ["n_estimators", "max_depth", "learning_rate"]
        elif args.decoder_type == "GBM":
            ignored_params = ["cost_param", "kernel_type", "gamma", "weight"]
        elif args.decoder_type in ["LSTM", "NB"]:
            ignored_params = [
                "cost_param",
                "kernel_type",
                "gamma",
                "weight",
                "n_estimators",
                "max_depth",
                "learning_rate",
            ]
        if ignored_params:
            for param in ignored_params:
                if hasattr(args, param):
                    setattr(args, param, None)

    print("Parameters inputted for current run:")
    for param in vars(args):
        print_wFrame(f"{param}: {getattr(args, param)}")
        if "n_proc" in param and getattr(args, param) is None:
            print_wFrame(
                f"WARNING: {param} is set to None. This means all available processors will be used. Use/test with caution.",
                frame_num=1,
            )
        if "cnmf" in param and getattr(args, param) is None:
            print_wFrame(
                f"WARNING: {param} is set to None. See TSF_enum.CNMF_Params for default value.",
                frame_num=1,
            )

        if param in ["path", "sess2process"] and not getattr(args, param):
            param2print = "a path" if param == "path" else "which sessions to process"
            print_wFrame(
                f"NOTE: Given empty input, user will be prompted to choose {param2print}.",
                frame_num=1,
            )
        if param in ["cost_param"] and getattr(args, param) is None:
            print_wFrame(
                "NOTE: cost_param is set to None. This means hyperparameter tuning will be performed to find the best C ff SVC is the decoder_type. If not the decoder_type, this will be ignored.",
                frame_num=1,
            )
        if param in ["kernel_type", "gamma", "weight"] and getattr(args, param) is None:
            dt = "SVC"
        elif (
            param in ["n_estimators", "max_depth", "learning_rate"]
            and getattr(args, param) is None
        ):
            dt = "GBM"
        else:
            dt = None
        if dt is not None:
            print_wFrame(
                f"NOTE: Set to None means decoder_type is not {dt}, so this will be ignored.",
                frame_num=1,
            )

    print(
        f"\nCheck {parser4}_enum.{enum_name} for more information and for setting default values.\n\n"
    )

    return args

import re

from CLAH_ImageAnalysis import utils
from CLAH_ImageAnalysis.behavior import TDML2tBD_enum as TDMLE

text_lib = utils.text_dict()
lean_str = text_lib["breaker"]["lean"]
TDMLkey = utils.enum_utils.enum2dict(TDMLE.TXT)
TIME_KEY = TDMLkey["TIME_KEY"]
POS_KEY = TDMLkey["POSITION_KEY"]


def create_event_structure(additional_keys: list = None) -> dict:
    """
    Create an event structure dictionary with predefined keys.

    Args:
        additional_keys (list, optional): Additional keys to include in the structure. Defaults to None.

    Returns:
        dict: Event structure dictionary with predefined keys and optional additional keys.
    """
    structure = {
        TDMLkey["START"]: {key: [] for key in [TIME_KEY, POS_KEY]},
        TDMLkey["STOP"]: {key: [] for key in [TIME_KEY, POS_KEY]},
    }
    if additional_keys:
        for key in additional_keys:
            structure[key] = []
    return structure


def init_empty_keys(keys: list) -> dict:
    """
    Initializes an empty dictionary with the given keys.

    Parameters:
    keys (list): A list of keys to be used in the dictionary.

    Returns:
    dict: An empty dictionary with the given keys as the keys and empty lists as the values.
    """
    return {key: [] for key in keys}


def extract_first_number(s: str) -> int:
    """
    Extracts the first number from a given string.

    Parameters:
    s (str): The input string from which the first number needs to be extracted.

    Returns:
    int: The first number found in the string. If no number is found, returns 0.
    """
    match = re.search(r"\d+", s)
    return int(match.group()) if match else 0


def print_dict_tree(dict_to_print: dict, title: list = []) -> None:
    """
    Prints the structure of a dictionary tree.

    Args:
        dict_to_print (dict): The dictionary to be printed.
        title (list, optional): The title of the dictionary tree. Defaults to [].

    Returns:
        None
    """
    if title:
        print(f"Structure of {title}:")
    print(lean_str)
    sorted_keys = sorted(dict_to_print.keys(), key=lambda x: extract_first_number(x))
    for key in sorted_keys:
        print(key)
        for sub_key in dict_to_print[key]:
            utils.print_wFrame(f"{sub_key} = {dict_to_print[key][sub_key]}")
    print(lean_str)


def print_lapDict_results(lapDict: dict, cues: list) -> None:
    """
    Prints the lap dictionary results and the number of cues found for a session.

    Parameters:
    lapDict (dict): The lap dictionary containing lap types and their corresponding values.
    cues (list): The list of cues found for the session.

    Returns:
    None
    """
    total_cue = len(cues)
    print_dict_tree(lapDict, "Lap Types")
    print(f"{total_cue} cue(s) were found for this session:")
    for idx, cue in enumerate(cues, start=1):
        utils.print_wFrame(f"{idx} - {cue}")
    print()

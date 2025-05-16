import json

from CLAH_ImageAnalysis.utils import (
    debug_utils,
    get_current_date_string,
    text_formatting,
)

# strings for file names & process
file_tag = text_formatting.text_dict()["file_tag"]


def enum2dict(enum_class: object) -> dict:
    """
    Get a dictionary of constant names and their corresponding values from an enum class.

    Parameters:
        enum_class: The enum class to extract constants from.

    Returns:
        A dictionary containing constant names as keys and their corresponding values as values.
    """

    constants_dict = {}
    # Use __members__.items() to get all members, including aliases
    for name, member in enum_class.__members__.items():
        constants_dict[name] = member.value
    return constants_dict


def export_param_from_enum(
    enum_class: object,
    filename: str,
    file_type: str,
    indent: int = 4,
    date: bool = False,
) -> None:
    """
    Export constants from an enum class to a file.

    Parameters:
        enum_class (object): The enum class to export constants from.
        filename (str): The name of the file to save the constants to.
        file_type (str): The type of file to save the constants to (e.g., "JSON", "TXT").
        indent (int): The number of spaces to use for indentation in the JSON file (default is 4).
    """

    if date is True:
        filename = f"{filename}_{get_current_date_string()}"

    debug_utils.raiseVE_wAllowables(
        file_type, set([file_tag["JSON"], file_tag["TXT"]]), "file_type"
    )
    if file_type == file_tag["JSON"]:
        enum_dict = enum2dict(enum_class)
        create_json_from_enumDict(enum_dict, filename, indent)
    elif file_type == file_tag["TXT"]:
        create_txt_from_enum(enum_class, filename)


def create_json_from_enumDict(enum_dict: dict, filename: str, indent: int = 4) -> None:
    """
    Create a JSON file from a dictionary of constants.

    Parameters:
        enum_dict (dict): The dictionary of constants.
        filename (str): The name of the file to save the JSON data to.
        indent (int): The number of spaces to use for indentation in the JSON file (default is 4).
    """

    filename2save = f"{filename}{file_tag['JSON']}"
    with open(filename2save, "w") as file:
        json.dump(enum_dict, file, indent=indent)


def create_txt_from_enum(enum_class: object, filename: str) -> None:
    """
    Create a text file from an enum class.

    Parameters:
        enum_class (object): The enum class to create the text file from.
        filename (str): The name of the file to save the text data to.
    """

    filename2save = f"{filename}{file_tag['TXT']}"
    with open(filename2save, "w") as file:
        for name, member in enum_class.__members__.items():
            file.write(f"{name} = {member.value}\n")

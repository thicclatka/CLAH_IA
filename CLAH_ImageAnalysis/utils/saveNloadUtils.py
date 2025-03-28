import json
import os
import pickle
import h5py
import numpy as np
from rich import print
from scipy.io import loadmat
from scipy.io import savemat
from scipy.sparse import isspmatrix
from CLAH_ImageAnalysis.utils import basename_finder
from CLAH_ImageAnalysis.utils import enum_utils
from CLAH_ImageAnalysis.utils import get_current_date_string
from CLAH_ImageAnalysis.utils import text_formatting
from CLAH_ImageAnalysis.utils import debug_utils


# strings for file names & process
# text_lib = text_dict()
file_tag = text_formatting.text_dict()["file_tag"]
print_wFrame = text_formatting.print_wFrame

acceptable_filetypes = set(
    [
        file_tag["MAT"],
        file_tag["PKL"],
        file_tag["H5"],
        file_tag["JSON"],
        file_tag["TDML"],
    ]
)


def savedict2file(
    dict_to_save: dict,
    dict_name: str,
    filename: str,
    file_tag_to_remove: str = "",
    file_suffix: list = [],
    date: bool = False,
    filetype_to_save: list = [
        file_tag["MAT"],
        file_tag["PKL"],
        file_tag["H5"],
    ],
) -> None:
    """
    Save a dictionary to a file in various formats.

    Parameters:
        dict_to_save (dict): The dictionary to be saved.
        dict_name (str): The name of the dictionary.
        filename (str): The name of the file to save.
        file_tag_to_remove (list, optional): List of file tags to remove from the filename. Defaults to [].
        file_suffix (list, optional): List of file suffixes to append to the filename. Defaults to [].
        date (bool, optional): Whether to append the current date to the filename. Defaults to False.
        filetype_to_save (list, optional): List of file types to save the dictionary as. Defaults to [file_tag["MAT"], file_tag["JSON"], file_tag["H5"]].
    """

    #  sometimes filename will be string that requires no changes
    fname_save = filename
    #  if statements address when filename does needs adjustments
    if file_tag_to_remove:
        fname_save = basename_finder(fname_save, file_tag_to_remove)
    if file_suffix:
        fname_save = f"{fname_save}_{file_suffix}"
    if date:
        fname_save = f"{fname_save}_{get_current_date_string()}"

    # python_use = [file_tag["H5"], file_tag["JSON"], file_tag["PKL"]]
    # matlab_use = [file_tag["MAT"]]

    # check if filetype_to_save is a string for a single file type
    # if so, convert it to a list
    if isinstance(filetype_to_save, str):
        filetype_to_save = [filetype_to_save]

    # save the dictionary in the specified file formats
    for ftag in filetype_to_save:
        if ftag != file_tag["PKL"]:
            # replace None values with NaN values for non-pickle files
            dict_to_save = replace_noneWnan(dict_to_save)
            # dict_to_save = flatten_dict(dict_to_save)
        if ftag == file_tag["MAT"]:
            dict_to_save = convert_types4mat(dict_to_save)
        # in case there is any file extension in the filename
        # remove it & replace with the new file extension to be saved for current iteration
        fname_save = fname_save.split(".")[0] + ftag
        # save accordingly
        saver_func(
            dict_to_save=dict_to_save,
            dict_name=dict_name,
            filename=fname_save,
            file_end=ftag,
        )


def flatten_dict(d: dict, parent_key: str = "", sep: str = "_") -> dict:
    """
    Recursively flattens a nested dictionary into a single-level dictionary.

    Parameters:
        d (dict): The dictionary to be flattened.
        parent_key (str, optional): The parent key to be used for the flattened keys. Defaults to "".
        sep (str, optional): The separator to be used between the parent key and the child key. Defaults to "_".

    Returns:
        dict: The flattened dictionary.

    Raises:
        ValueError: If the value is not a dictionary or a list of arrays.
    """

    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            try:
                # Check if v is a list of arrays with different shapes or contains np.nan
                if isinstance(v, list) and (
                    all(isinstance(i, np.ndarray) for i in v)
                    or any(np.isnan(i) for i in v if isinstance(i, float))
                ):
                    items.append((new_key, v))  # Keep the list of arrays as is
                else:
                    items.append((new_key, np.array(v)))
            except ValueError as e:
                print(f"Error converting value to numpy array for key {new_key}: {e}")
                print(f"Value: {v}")
    return dict(items)


def saver_func(
    dict_to_save: dict, dict_name: str, filename: str, file_end: str
) -> None:
    """
    Save a dictionary to a file based on the specified file format.

    Parameters:
        dict_to_save (dict): The dictionary to be saved.
        dict_name (str): The name of the dictionary.
        filename (str): The name of the file to save.
        file_end (str): The file format to use for saving.

    Raises:
        ValueError: If the file format is not recognized.
    """

    print(f"Saving {dict_name} as {file_end}")

    if "/" in filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)

    print_wFrame(f"Filename: {filename}")
    print_wFrame("Saving", end="", flush=True)
    debug_utils.raiseVE_wAllowables(file_end, acceptable_filetypes, "file type")
    if file_end == file_tag["MAT"]:
        try:
            savemat(filename, dict_to_save)
        except ValueError as e:
            print(f"Error saving dictionary to MAT file: {e}")
    elif file_end == file_tag["PKL"]:
        with open(filename, "wb") as pkl_file:
            pickle.dump(dict_to_save, pkl_file)
    elif file_end == file_tag["H5"]:
        dict_to_save4H5 = convert_dtype_for_h5(dict_to_save)
        with h5py.File(filename, "w") as h5file:
            for key, value in dict_to_save4H5.items():
                # for sparse matrices, save the data, indices, indptr, and shape separately
                if isspmatrix(value):
                    grp = h5file.create_group(key)
                    for attr in ["data", "indices", "indptr", "shape"]:
                        grp.create_dataset(attr, data=np.array(getattr(value, attr)))
                elif isinstance(value, dict):
                    grp = h5file.create_group(key)
                    for k, v in value.items():
                        grp.create_dataset(k, data=v)
                else:
                    h5file.create_dataset(key, data=value)
    elif file_end == file_tag["JSON"]:
        dict_to_save = convert_types4json(dict_to_save)
        with open(filename, "w") as json_file:
            json.dump(dict_to_save, json_file, indent=2, sort_keys=True)
    text_formatting.print_done_small_proc()


def convert_types4json(obj: object) -> object:
    """
    Convert types within an object to a format compatible with JSON.

    Parameters:
        obj (object): The object to be converted.

    Returns:
        object: The converted object.
    """
    if isinstance(obj, np.int64):
        return int(obj)
    if isinstance(obj, dict):
        return {k: convert_types4json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_types4json(item) for item in obj]
    return obj


def convert_dtype_for_h5(obj: object) -> object:
    """
    Converts data within an object to a format compatible with HDF5.
    Specifically handles the case where an object is a numpy array of lists,
    each containing numpy arrays, and converts them as needed.

    Parameters:
        obj (object):   The object to be converted, can be a dict, list, numpy array,
                        or a nested structure of these.

    Returns:
        object: The converted object with data in an HDF5 compatible format.
    """

    if isinstance(obj, np.ndarray):
        try:
            # Try to convert to float64
            obj = obj.astype(np.float64)
        except ValueError:
            # If it fails, convert to string
            obj = obj.astype("S")
    elif isinstance(obj, dict):
        # Recursively apply to values if it's a dictionary
        for key, value in obj.items():
            obj[key] = convert_dtype_for_h5(value)
    elif isinstance(obj, list):
        # Apply to each item if it's a list
        if len(obj) > 1:
            obj = [convert_dtype_for_h5(item) for item in obj]
        else:
            obj = convert_dtype_for_h5(np.array(obj))
    elif isinstance(obj, tuple):
        # Apply to each item if it's a tuple
        obj = tuple(convert_dtype_for_h5(item) for item in obj)
    elif isinstance(obj, str):
        # convert to list & then process accordingly
        obj = [obj]
        obj = convert_dtype_for_h5(obj)

    return obj


def replace_noneWnan(obj: object) -> object:
    """
    Replaces None values with NaN values in a nested dictionary, list, or numpy array.

    Parameters:
        obj: The object to be processed. Can be a dictionary, list, or numpy array.

    Returns:
        The processed object with None values replaced by NaN values.
    """

    if isinstance(obj, dict):
        for key, value in obj.items():
            if value is None:
                obj[key] = np.nan
            else:
                obj[key] = replace_noneWnan(value)
    elif isinstance(obj, list):
        for i in range(len(obj)):
            if obj[i] is None:
                obj[i] = np.nan
            else:
                obj[i] = replace_noneWnan(obj[i])
    elif isinstance(obj, np.ndarray):
        obj = np.where(obj is None, np.nan, obj)
    return obj


def convert_types4mat(obj: object) -> object:
    """
    Convert types within an object to a format compatible with MATLAB.
    Converts simple lists (containing numbers) to numpy arrays, while preserving
    lists of lists or lists of arrays.

    Parameters:
        obj (object): The object to be converted.

    Returns:
        object: The converted object with simple lists transformed to numpy arrays.
    """
    if isinstance(obj, list):
        # Check if it's a list of lists or list of arrays
        if any(isinstance(x, (list, np.ndarray)) for x in obj):
            return [convert_types4mat(item) for item in obj]
        # If it's a simple list, convert to array
        return np.array(obj)
    elif isinstance(obj, dict):
        return {
            f"K{k}"
            if isinstance(k, str) and k.replace(".", "").isdigit()
            else k: convert_types4mat(v)
            for k, v in obj.items()
        }
    return obj


def load_pkl_file(fname: str) -> dict:
    """
    Load a pickle file and return the loaded dictionary.

    Parameters:
        fname (str): The file path of the pickle file to load.

    Returns:
        dict: The loaded dictionary from the pickle file.
    """

    with open(fname, "rb") as file:
        loaded_dict = pickle.load(file)
    return loaded_dict


def load_h5_file(filename: str) -> dict:
    """
    Load data from an HDF5 file.

    Parameters:
        filename (str): The path to the HDF5 file.

    Returns:
        dict: A dictionary containing the data loaded from the HDF5 file.
    """

    def read_item(item):
        """
        Reads an item from the HDF5 file. Converts datasets to NumPy arrays
        and recursively reads groups.
        """
        if isinstance(item, h5py.Dataset):
            # Convert datasets to NumPy arrays
            return item[()]
        elif isinstance(item, h5py.Group):
            # Recursively read groups
            return {subkey: read_item(item[subkey]) for subkey in item}
        else:
            return item

    data = {}
    with h5py.File(filename, "r") as h5_file:
        # Iterate over items in the HDF5 file and read them
        for key, value in h5_file.items():
            data[key] = read_item(value)

    return data


def load_json_file(
    filename: str, list2append: list | None = None, multiobj: bool = False
) -> dict | list | None:
    """
    Load a JSON file and return its contents as a Python object.

    Parameters:
        filename (str): The path to the JSON file.

    Returns:
        dict: The contents of the JSON file as a Python dictionary.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        JSONDecodeError: If the file is not a valid JSON file.

    """
    with open(filename, "r") as json_file:
        if not multiobj and list2append is None:
            try:
                return json.load(json_file)
            except json.JSONDecodeError:
                print(f"Error loading JSON file: {filename}")
            return None
        else:
            for line in json_file:
                try:
                    json_line = json.loads(line)
                    list2append.append(json_line)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON from line: {line}")
            return list2append


def load_file(
    fname: str,
    previous: bool = False,
    multi4json: bool = False,
    list2append4json: list | None = None,
) -> dict:
    """
    Load a file based on the given file name and file tag.

    Parameters:
        fname (str): The file name or path.
        previous (bool, optional): Whether the file is being loaded as part of a previous process. Defaults to False.

    Returns:
        The loaded file object.

    Raises:
        ValueError: If the file extension is missing from the filename.
        ValueError: If the file tag is not recognized.
    """

    if previous:
        print_wFrame(f"Found previous file: {fname}")
        print_wFrame("Loading", end="", flush=True)

    # Extract the file name and extension
    file_name, file_extension = os.path.splitext(fname)

    # Check if the file extension is present
    if not file_extension:
        raise ValueError("File extension is missing from the filename.")

    debug_utils.raiseVE_wAllowables(file_extension, acceptable_filetypes, "file type")
    if file_extension == file_tag["MAT"]:
        file2load = loadmat(fname)
    elif file_extension == file_tag["PKL"]:
        file2load = load_pkl_file(fname)
    elif file_extension == file_tag["H5"]:
        file2load = load_h5_file(fname)
    elif file_extension in [file_tag["JSON"], file_tag["TDML"]]:
        file2load = load_json_file(
            fname, multiobj=multi4json, list2append=list2append4json
        )

    if previous:
        text_formatting.print_done_small_proc()
    return file2load


def load_segDict(
    filename: str,
    all: bool = False,
    keep_A_sparse: bool = False,
    print_prev_bool: bool = True,
    **kwargs,
) -> list | np.ndarray | None:
    """
    Load segmentation dictionary from a file.

    Parameters:
        filename (str): The path to the file containing the segmentation dictionary.
        all (bool, optional): If True, return all data in the order they are defined in the dictionary.
            If False (default), return only the requested data.
        keep_A_sparse (bool, optional): If True, keep the 'A_SPATIAL' data as a sparse matrix.
            If False (default), convert the 'A_SPATIAL' data to a dense ndarray.
        **kwargs: Additional keyword arguments used to filter the output dictionary.

    Returns:
        list or ndarray or None: The requested data from the segmentation dictionary.
            If `all` is True, returns a list of all data in the order they are defined in the dictionary.
            If `all` is False and no data is requested, returns None.
            If `all` is False and only one data is requested, returns that data.
            If `all` is False and multiple data are requested, returns a list of the requested data in the order they are requested.
    """

    def _checkKey_add2output_dict(key: str) -> None:
        key2use = sD_str[key]
        if key2use in unloaded_dict:
            output_dict[key2use] = unloaded_dict[key2use]

    # load module
    # needs to be done here to avoid circular import
    from CLAH_ImageAnalysis.tifStackFunc.TSF_enum import segDict_Txt

    sD_str = enum_utils.enum2dict(segDict_Txt)
    sD_keys = list(sD_str.keys())

    # load segDict
    unloaded_dict = load_file(filename, previous=print_prev_bool)
    output_dict = {}
    for key in sD_keys:
        _checkKey_add2output_dict(key)

    if all:
        # Return all data in the order they are defined in output_dict
        return [output_dict[key] for key in output_dict]

    # Filter the output_dict based on the kwargs
    requested_data = [
        output_dict[key] for key in kwargs if key in output_dict and kwargs[key]
    ]

    # Return the requested data in the order they are requested
    return (
        requested_data
        if len(requested_data) > 1
        else (requested_data[0] if requested_data else None)
    )


def convert_lists_to_arrays(obj: object) -> object:
    """
    Converts lists within an object to numpy arrays recursively.

    Parameters:
        obj (object): The object to be converted.

    Returns:
        object: The converted object with lists replaced by numpy arrays.
    """

    if isinstance(obj, list):
        return np.array(obj)
    elif isinstance(obj, dict):
        return {k: convert_lists_to_arrays(v) for k, v in obj.items()}
    return obj

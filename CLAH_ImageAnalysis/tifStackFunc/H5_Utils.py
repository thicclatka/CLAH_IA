import gc
from collections import namedtuple

import h5py
import numpy as np

from CLAH_ImageAnalysis.core import BaseClass as BC


class H5_Utils(BC):
    """A utility class for working with HDF5 files."""

    def __init__(self) -> None:
        self.program_name = "H5"
        self.class_type = "utils"
        BC.__init__(self, program_name=self.program_name, mode=self.class_type)

    def print_h5_tree(self, name: str, obj: object) -> None:
        """
        Print the HDF5 object and its attributes in a tree format.

        Parameters:
            name (str): The name of the HDF5 object.
            obj (object): The HDF5 object.
        """
        if isinstance(obj, h5py.Dataset):
            self.print_wFrm(f"Dataset: {name} (dtype: {obj.dtype}, shape: {obj.shape})")
        elif isinstance(obj, h5py.Group):
            self.print_wFrm(f"{name}")
        self.print_attrs(name, obj)

    def print_attrs(self, name: str, obj: object) -> None:
        """
        Print the attributes of an HDF5 object.

        Parameters:
            name (str): The name of the HDF5 object.
            obj (object): The HDF5 object.
        """
        attrs = list(obj.attrs.items())
        if attrs:
            for attr in attrs:
                self.print_wFrm(f'Attribute: "{attr[0]}" = "{attr[1]}"', frame_num=1)

    def display_h5_tree(self, file_path: str) -> None:
        """
        Display the tree structure of an HDF5 file.

        Parameters:
            file_path (str): The path to the HDF5 file.
        """
        with h5py.File(file_path, "r") as f:
            f.visititems(self.print_h5_tree)

    def _file_read_utils(self, file_to_read: str) -> str:
        """
        Selects a file to read based on the user's input.

        Parameters:
            file_to_read (str): The path to the file to read.

        Returns:
            str: The path to the selected file.
        """
        file_to_read = self.utils.file_selector(
            file_to_read, default=self.file_tag["H5"]
        )
        return file_to_read

    def _get_element_size(self, hf: h5py.File, hf_key: str = "imaging") -> tuple | None:
        """
        Get the element size in micrometers from the given HDF5 file.

        Parameters:
            hf (h5py.File): The HDF5 file object.

        Returns:
            tuple or None: The element size in micrometers as a tuple (x, y, z) if available,
            or None if not found.
        """
        element_size_um = None
        if hf_key in hf:
            pass
        elif hf_key not in hf:
            hf_key = "/"
        if "element_size_um" in hf[hf_key].attrs:
            element_size_um = hf[hf_key].attrs["element_size_um"]
            if isinstance(element_size_um, str):
                element_size_um = tuple(
                    map(float, element_size_um.strip("()").split(", "))
                )
            elif isinstance(element_size_um, np.ndarray):
                element_size_um = tuple(element_size_um)
        return element_size_um

    def extract_element_size(self, file_to_read: str) -> tuple | None:
        """
        Extract the element size from an HDF5 file.

        This method is a handler for when the user just wants to extract the element size from an HDF5 file.

        Parameters:
            file_to_read (str): The path to the HDF5 file.

        Returns:
            tuple: The element size in micrometers (um) as a tuple (x, y, z).

        """
        file_to_read = self._file_read_utils(file_to_read)
        hf = h5py.File(file_to_read, "r+")
        element_size_um = self._get_element_size(hf)
        hf.close()
        return element_size_um

    def read_file(self, file_to_read: str, verbose: bool = True) -> tuple:
        """
        Reads an h5 file and returns the file object and filename.

        Parameters:
            file_to_read (str): The path to the h5 file to be read.
            verbose (bool, optional): Whether to print verbose output. Defaults to True.

        Returns:
            tuple: A tuple containing the h5 file object and the filename.
        """
        file_to_read = self._file_read_utils(file_to_read)

        if verbose:
            self.rprint("H5 READER")
            self.rprint(self.text_lib["breaker"]["lean"])
        h5filename = file_to_read
        if verbose:
            self.rprint("Reading h5 file at " + h5filename)

        hf = h5py.File(h5filename, "r+")
        return hf, h5filename

    def read_file4MOCO(
        self, file_to_read: str, endFr: int | None = None, hf_key: str = "imaging"
    ) -> tuple:
        """
        Reads a file for motion correction.

        Parameters:
            file_to_read (str): The path of the file to read.
            endFr (int, optional): The end frame number. Defaults to 0.
            hf_key (str, optional): The key to access the imaging data in the HDF5 file. Defaults to "imaging".

        Returns:
            tuple: A tuple containing the following elements:
                - h5filename (str): The name of the HDF5 file.
                - hfsiz (numpy.ndarray): An array containing the dimensions of the image stack.
                - info (tuple): A tuple containing information about the image stack.
                - segCh (int): The index of the selected channel.
                - chan_num (list): A list of channel numbers.
                - element_size_um (float): The size of each image element in micrometers.
        """
        # init vars
        h5filename = None
        hfsiz = None
        info = None
        segCh = None
        chan_num = None
        element_size_um = None

        # read file
        hf, h5filename = self.read_file(file_to_read)
        info = hf[hf_key].shape
        element_size_um = self._get_element_size(hf)
        hf.close()

        Total_frames = info[0] if endFr is None else endFr

        if len(info) == 5:
            numCh = info[1]
            if numCh == 1:
                segCh = 0  # selects green
                chan_num = [1]
            elif numCh > 1:
                segCh = 1  # selects green / red is at Ch1 (0 in py index)
                chan_num = [0, 1]
        else:
            numCh = 1
            segCh = None
            chan_num = [0]

        d = namedtuple("d", ["h", "w"])
        if len(info) == 5:
            d.h = info[-2]  # height of image
            d.w = info[-3]  # width of image
        else:
            d.h = info[1]  # height of image
            d.w = info[2]  # width of image
        hfsiz = np.transpose([d.h, d.w, Total_frames])

        self.display_h5_tree(h5filename)

        if numCh == 1 and segCh is not None:
            self.print_wFrm(f"Found {Total_frames} frames from Ch {chan_num[0] + 1}")
        elif numCh == 1 and segCh is None:
            self.print_wFrm(f"Found {Total_frames} frames from miniscope h5 file")
        else:
            incremented_channels = ", ".join([str(x + 1) for x in chan_num])
            self.print_wFrm(
                f"Found {Total_frames} frames & 2 channels: {incremented_channels}"
            )
        if segCh is not None:
            self.print_wFrm(f"segCh is Ch {chan_num[segCh] + 1}")
        self.rprint(self.text_lib["breaker"]["lean"], "\n")
        return h5filename, hfsiz, info, segCh, chan_num, element_size_um

    def read_file4wrapMultSess(self, file_to_read: str) -> tuple | None:
        """
        Reads a file and returns the element size in micrometers.

        Parameters:
            file_to_read (str): The file to be read.

        Returns:
            tuple | None: The element size in micrometers.
        """
        hf, _ = self.read_file(file_to_read, verbose=False)
        element_size_um = self._get_element_size(hf)
        return element_size_um

    def write_to_file(
        self,
        array_to_write: np.ndarray,
        filename: str,
        chan_idx: list,
        element_size_um: tuple,
        dimension_labels: str,
        hf_group: str = "/",
        hf_key: str = "imaging",
        date: bool = False,
        return_fname: bool = False,
        twoChan: bool = False,
    ):
        """Write data to an HDF5 file.

        Parameters:
            array_to_write (numpy.ndarray): The array to write to the HDF5 file.
            filename (str): The name of the HDF5 file.
            chan_idx (list): The channel indices.
            element_size_um (tuple): The element size in micrometers (um).
            dimension_labels (str): The dimension labels.
            date (bool, optional): Whether to add the current date to the filename. Defaults to False.
            return_fname (bool, optional): Whether to return the filename. Defaults to False.
            twoChan (bool, optional): Whether to add the channel index to the filename. Defaults to False.

        Returns:
            str: The path to the HDF5 file.

        """
        TKEEPER = self.time_utils.TimeKeeper(cst_msg="Writing data to h5")

        # create hf_name which is the group and key concatenated
        # usually will be "/imaging"
        hf_name = f"{hf_group}{hf_key}"
        # add date if specified
        if date:
            filename = filename + "_" + self.time_utils.get_current_date_string()

        if twoChan:
            filename += f"_Ch{chan_idx[0] + 1}"

        # add h5 to end
        filename += self.file_tag["H5"]
        self.print_wFrm(f"Using filename: {filename}")
        self.print_wFrm("Writing data to h5", end="", flush=True)
        hf = h5py.File(filename, "w")
        try:
            hf[hf_group].create_dataset(hf_key, data=array_to_write)
            # after dataset is created use hf_name
            hf[hf_name].attrs["DIMENSION_LABELS"] = dimension_labels
            hf[hf_name].attrs["channel_index"] = [f"Ch{ch + 1}" for ch in chan_idx]
            if element_size_um is not None:
                hf[hf_name].attrs["element_size_um"] = np.array(element_size_um)
            hf.close()
            self.print_done_small_proc(new_line=False)
        except Exception as e:
            hf.close()
            self.utils.debug_utils.raiseVE_SysExit1(f"An error occurred: {e}")
        # set end of timer
        TKEEPER.setEndNprintDuration()
        print()
        # print H5 structure
        self.rprint("Structure of recently written h5 structure")
        self.print_wFrm(f"Dataset Path: {hf_name}")
        self.print_wFrm(f"Filename: {filename}")
        self.display_h5_tree(filename)
        print()
        if return_fname:
            return filename

    def squeeze_fileNwrite(
        self,
        file2read: str,
        chan_idx: list,
        element_size_um: tuple,
        dimension_labels: str,
        array2use: np.ndarray | None = None,
        hf_key="imaging",
        remove_Cycle: bool = False,
        twoChan: bool = False,
        export_sample: bool = False,
        high_pass_applied: bool = False,
        CLAHE_applied: bool = False,
        bandpass_applied: bool = False,
        crop_applied: bool = False,
    ):
        """Squeeze an HDF5 file and write the squeezed data to a new HDF5 file.

        Parameters:
            file_to_read (str): The path to the HDF5 file.
            chan_idx (int): List of the selected channel to use.
            element_size_um (tuple): The element size in micrometers (um).
            dimension_labels (str): The dimension labels.
            array2use (numpy.ndarray, optional): The array to write to the HDF5 file. Defaults to None.
            hf_key (str, optional): The key to access the imaging data in the HDF5 file. Defaults to "imaging".
            remove_Cycle (bool, optional): Whether to remove the cycle tag from the filename. Defaults to False.
            twoChan (bool, optional): Whether to add the channel index to the filename. Defaults to False.
            export_sample (bool, optional): Whether to export sample of the squeezed data. Defaults to False.
            high_pass_applied (bool, optional): Whether high-pass filter was applied to the data. Defaults to False.
            CLAHE_applied (bool, optional): Whether CLAHE was applied to the data. Defaults to False.
            bandpass_applied (bool, optional): Whether bandpass filter was applied to the data. Defaults to False.
            crop_applied (bool, optional): Whether cropping was applied to the data. Defaults to False.
        Returns:
            str: The path to the squeezed HDF5 file.
        Raises:
            ValueError: If the array shape is unexpected.
        """

        applied_strings = [
            ("_BPF", bandpass_applied),
            ("_HPF", high_pass_applied),
            ("_CE", CLAHE_applied),
            ("_CROPPED", crop_applied),
        ]

        h5filename = file2read
        fname_split = h5filename.split(".")[0]
        if remove_Cycle:
            fname_split = self.folder_tools.basename_finder(
                fname_split, self.file_tag["CYCLE"]
            )

        h5fname_sqz = fname_split + self.file_tag["SQZ"]

        if array2use is None:
            try:
                hf = h5py.File(h5filename, "r+")
                hf_loaded = hf[hf_key][()]
            except Exception as e:
                hf.close()
                self.utils.debug_utils.raiseVE_SysExit1(f"An error occurred: {e}")
        else:
            hf_loaded = array2use

        if len(hf_loaded.shape) == 5:
            # dimension order is t,c,y,x,z
            hf_squeeze = np.squeeze(hf_loaded[:, chan_idx[0], :, :, 0])
        elif len(hf_loaded.shape) == 3:
            # if data is 3D, implies miniscope data
            # add keywords to filename pending on what was applied in preprocessing
            # used _applied parameters to determine what was applied

            hf_squeeze = hf_loaded
            filtered_fname_append = ""
            for string, applied_bool in applied_strings:
                if applied_bool:
                    filtered_fname_append += string

            h5fname_sqz = fname_split + filtered_fname_append + self.file_tag["SQZ"]
        else:
            raise ValueError(
                f"Unexpected array shape: {hf_loaded.shape}. "
                "Expected either 5D array (t,1,y,x,c) or 3D array (t,y,x)"
            )

        h5fname_sqz = self.write_to_file(
            array_to_write=hf_squeeze,
            filename=h5fname_sqz,
            chan_idx=chan_idx,
            element_size_um=element_size_um,
            dimension_labels=dimension_labels,
            date=False,
            return_fname=True,
            twoChan=twoChan,
        )

        if export_sample:
            shortened_h5fname_sqz = (
                fname_split
                + filtered_fname_append
                + self.file_tag["SQZ"]
                + self.file_tag["ABBR_DS"]
            )
            array2use = hf_squeeze[:500, :, :]
            # array2use = array2use[:, :256, -256:]
            _ = self.write_to_file(
                array_to_write=array2use,
                filename=shortened_h5fname_sqz,
                chan_idx=chan_idx,
                dimension_labels=dimension_labels,
                element_size_um=element_size_um,
                date=False,
                return_fname=True,
                twoChan=twoChan,
            )
        hf_loaded = None
        gc.collect()
        return h5fname_sqz

    def concatH5s(self, H5s: list, fname_concat: str, hf_key: str = "imaging") -> str:
        """
        Concatenate HDF5 files and write the concatenated data to a new HDF5 file.

        Parameters:
            H5s (list): List of HDF5 files to concatenate.
            fname_concat (str): The name of the concatenated HDF5 file.
            hf_key (str, optional): The key to access the imaging data in the HDF5 files. Defaults to "imaging".

        Returns:
            str: The path to the concatenated HDF5 file.
        """
        datasets = []
        element_size_um = []
        for h5 in H5s:
            hf, hfname = self.read_file(h5, verbose=False)
            datasets.append(hf[hf_key][:])
            element_size_um.append(self._get_element_size(hf))
            hf.close()

        concatenatedH5 = np.concatenate(datasets, axis=0)

        fname_concat = self.write_to_file(
            array_to_write=concatenatedH5,
            filename=fname_concat,
            chan_idx=[1],
            element_size_um=element_size_um[0],
            dimension_labels=["t", "z", "y", "x", "c"],
            date=False,
            return_fname=True,
            twoChan=False,
        )

        return fname_concat

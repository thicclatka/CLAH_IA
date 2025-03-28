import os
import pandas as pd
import numpy as np
import zipfile
import argparse
import tifffile as tif
import isx

# import easygui
from scipy import sparse
from rich import print
from CLAH_ImageAnalysis import utils


class ISX_csv_converter:
    """Convert ISX CSV files containing CNMFe results to segDict format.

    This class processes CSV files and TIFF stacks exported from ISX containing CNMFe calcium imaging analysis results.
    It extracts temporal (C) and spatial (A) components and saves them in a standardized dictionary format (segDict).

    The converter:
    1. Finds folders containing ISX CSV files
    2. For each folder:
        - Extracts temporal components (activity traces)
        - Extracts spatial components (cell footprints)
        - Creates and saves a segDict with the components

    The segDict contains:
        - C: Filtered temporal components (accepted/undecided cells)
        - A: Filtered spatial components (accepted/undecided cells)
        - accepted_labels: Component acceptance status

    Args:
        file_path (str | None, optional): Path to folder containing session folders with ISX CNMFe results.
            If None, opens file dialog. Defaults to None.
    """

    def __init__(self, file_path: str | None = None) -> None:
        """Initialize the ISX CSV converter.

        Creates a converter object to process ISX CSV files containing CNMFe results.
        If no file path is provided, opens a GUI dialog to select the folder.

        Args:
            file_path (str | None, optional): Path to folder containing session folders with ISX CNMFe results.
                If None, opens file dialog. Defaults to None.

        Attributes:
            file_path (str): Path to folder containing session folders
            text_lib (dict): Dictionary of text constants from utils
            file_tag (dict): File naming patterns from text_lib
            cellmap_tiffs_fname (str): Name of folder containing cell map TIFFs
            accepted2use (list): List of cell acceptance statuses to include
            AvB_Check (bool): Whether to check for AvB freezing times
        """
        self.file_path = utils.check_folder_path(
            file_path,
            msg="Select folder containing session folders, where each session folder contains the CNMFe results files exported from ISX.",
        )

        self.file_path = file_path
        self.text_lib = utils.text_dict()
        self.file_tag = self.text_lib["file_tag"]
        self.cellmap_tiffs_fname = "cellmap_tiffs"
        self.accepted2use = ["accepted", "undecided"]

        self.AvB_Check = False

    def _find_CNMFE_file(self, folder: str = None, file_type: str = None) -> str:
        """Find the most recent CNMFE file in the specified folder.

        Args:
            folder (str, optional): Path to folder to search. If None, searches current directory.
            file_type (str, optional): Type of file to search for. Must be 'CSV' or 'ISXD'.

        Returns:
            str: Path to most recent matching CNMFE file.
        """
        if folder is not None:
            folder = os.path.abspath(folder)
            os.chdir(folder)

        if file_type not in ["CSV", "ISXD"]:
            raise ValueError(f"file_type must be 'CSV' or 'ISXD'. Got {file_type}.")

        if file_type == "ISXD":
            cnmf2use = self.file_tag["CNMFE2"]
        else:
            cnmf2use = self.file_tag["CNMFE"]
        file2use = self.file_tag[file_type]

        cellset_file = utils.findLatest([cnmf2use, file2use])

        if folder is not None:
            os.chdir("..")

        return cellset_file

    def _create_CTemp(self) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        """Create array of cell time course data from CSV.

        Imports CSV data and extracts:
        - Cell activity traces
        - Cell acceptance labels

        Returns:
            tuple containing:
                - CTemp (np.ndarray): Cell activity data (num_cells x num_timepoints)
                - accepted_labels (np.ndarray): Cell acceptance labels
            Or (None, None) if processing fails
        """
        utils.print_wFrame("Creating CTemp")

        try:
            if not self._find_CNMFE_file(file_type="ISXD"):
                utils.print_wFrame("Importing CSV as dataframe", frame_num=1)
                C_as_CSV = self._import_csvNremove_whitespace()

                utils.print_wFrame("Creating accepted_labels", frame_num=1)
                accepted_labels = C_as_CSV.iloc[0, 1:].astype("U").values

                utils.print_wFrame(
                    "Extracting time course data by cell from dataframe", frame_num=1
                )
                CTemp = np.transpose(C_as_CSV.iloc[1:, 1:].astype(float).values)

                CFrameTimes = np.array(C_as_CSV.iloc[1:, 0].astype(float).values)
            elif self._find_CNMFE_file(file_type="ISXD"):
                utils.print_wFrame("Importing CTemp from ISXD", frame_num=1)
                CTemp, _, accepted_labels, CFrameTimes = self._import_isx_cell_set()

            utils.print_wFrame(f"CTemp shape: {CTemp.shape}", frame_num=1)
            utils.print_wFrame(f"Cell number: {CTemp.shape[0]}", frame_num=2)
            utils.print_wFrame(f"Total time: {CTemp.shape[1]}", frame_num=2)
            utils.print_wFrame(
                f"accepted_labels shape: {accepted_labels.shape[0]}", frame_num=1
            )

            return CTemp, accepted_labels, CFrameTimes

        except Exception as e:
            print(f"Error creating CTemp: {e}")
            return None, None

    def _import_isx_cell_set(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Import ISX cell set file.

        Returns:
            tuple containing:
                - C_as_ISXD (np.ndarray): Cell activity data (num_cells x num_timepoints)
                - accepted_labels (np.ndarray): Cell acceptance labels
                - CFrameTimes (np.ndarray): Frame times (num_timepoints)
                - ASpat (np.ndarray): Spatial components (num_cells x num_pixels)
        """
        cellset = isx.CellSet.read(self._find_CNMFE_file(file_type="ISXD"))
        C, A, accepted_labels = [], [], []
        for i in range(cellset.num_cells):
            C.append(cellset.get_cell_trace_data(i))
            A.append(cellset.get_cell_image_data(i))
            accepted_labels.append(cellset.get_cell_status(i))
        C = np.array(C)
        A = np.array(A)
        accepted_labels = np.array(accepted_labels, dtype="U")

        secs = round(cellset.timing.period.to_msecs(), -1) / 1000
        CFrameTimes = np.arange(0, secs * C.shape[1], secs)
        return C, A, accepted_labels, CFrameTimes

    def _import_csvNremove_whitespace(self) -> pd.DataFrame:
        """Import CSV file and remove whitespace.

        Removes whitespace from:
        - Column names
        - Index values
        - String cell values

        Returns:
            pd.DataFrame: DataFrame with cleaned text content
        """
        csv2convert = pd.read_csv(
            self._find_CNMFE_file(file_type="CSV"), low_memory=False
        )

        # Remove whitespace from column names
        csv2convert.columns = csv2convert.columns.str.strip().str.replace(
            r"\s+", "", regex=True
        )

        # Remove whitespace from the index
        csv2convert.index = csv2convert.index.map(lambda x: "".join(str(x).split()))

        # Remove whitespace from all cell values
        csv2convert = csv2convert.applymap(
            lambda x: "".join(str(x).split()) if isinstance(x, str) else x
        )
        return csv2convert

    def _unzip_tiffStack(self) -> str:
        """Unzip TIFF stack files from ZIP archive.

        Extracts TIFF files and removes original ZIP.

        Returns:
            str: Path to extracted TIFF directory
        """
        zip_file = utils.findLatest(self.file_tag["ZIP"])

        output_dir = os.path.join(self.current_folder, self.cellmap_tiffs_fname)
        os.makedirs(output_dir, exist_ok=True)

        utils.print_wFrame(f"Unzipping {zip_file}", frame_num=2, end="", flush=True)

        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(output_dir)

        print("...done")

        utils.print_wFrame(f"Removing {zip_file}", frame_num=2)

        # Remove the original ZIP file
        os.remove(zip_file)

        return output_dir

    def _import_tiffStack_N_create_cellmap(self, folderWtiffs: str) -> np.ndarray:
        """Import TIFF files and create cellmap.

        Args:
            folderWtiffs (str): Folder containing TIFF files

        Returns:
            np.ndarray: Stacked cellmap array (num_cells x height x width)
        """
        tiff_files = os.listdir(folderWtiffs)
        tiff_files.sort()
        tiff_files = [file for file in tiff_files if "accepted-cells-map" not in file]

        cellmap = []
        for file in tiff_files:
            with tif.TiffFile(os.path.join(folderWtiffs, file)) as tif_file:
                cellmap.append(tif_file.asarray())

        cellmap = np.stack(cellmap)

        return cellmap

    def _create_ASpat(self) -> sparse.csr_matrix | None:
        """Create sparse matrix of spatial components from TIFFs.

        Process steps:
        1. Check if TIFFs need unzipping
        2. Import TIFFs to create 3D cellmap
        3. Convert to sparse matrix format

        Returns:
            sparse.csr_matrix | None: Spatial components matrix (n_pixels x n_cells) or None if failed
        """
        utils.print_wFrame("Creating ASpat")

        try:
            if not self._find_CNMFE_file(file_type="ISXD"):
                cellmap_tiffs_dir = os.path.join(
                    self.current_folder, self.cellmap_tiffs_fname
                )
                if not os.path.exists(cellmap_tiffs_dir):
                    utils.print_wFrame(
                        "No cellmap_tiffs directory found. Unzipping is required.",
                        frame_num=1,
                    )
                    tiffStackFolder = self._unzip_tiffStack()
                else:
                    utils.print_wFrame(
                        "cellmap_tiffs directory found. Unzipping is not required.",
                        frame_num=1,
                    )
                    tiffStackFolder = cellmap_tiffs_dir

                utils.print_wFrame(
                    "Importing tiff files & converting to sparse array", frame_num=1
                )
                cellmap = self._import_tiffStack_N_create_cellmap(tiffStackFolder)

            elif self._find_CNMFE_file(file_type="ISXD"):
                utils.print_wFrame("Importing A from ISXD", frame_num=1)
                _, cellmap, _, _ = self._import_isx_cell_set()

            # convert to sparse matrix
            ASpat = self._convert2sparse(cellmap)
            utils.print_wFrame(f"Sparse matrix dimensions: {ASpat.shape}", frame_num=1)

            return ASpat

        except Exception as e:
            print(f"Error creating ASpat: {e}")
            return None

    def _convert2sparse(self, cellmap: np.ndarray) -> sparse.csr_matrix:
        """Convert 3D cellmap to sparse matrix.

        Args:
            cellmap (np.ndarray): 3D array (n_cells x height x width)

        Returns:
            sparse.csr_matrix: Flattened sparse matrix (height*width x n_cells)
        """
        cell_num, x, y = cellmap.shape
        squeezed_cellmap = np.transpose(cellmap.reshape(cell_num, x * y))

        return sparse.csr_matrix(squeezed_cellmap)

    # def _create_eventRate(self, CTemp: np.ndarray, window_size: int = 5) -> np.ndarray:
    #     """Create event rate from temporal components.

    #     Args:
    #         CTemp (np.ndarray): Temporal components (n_cells x n_timepoints)
    #         window_size (int, optional): Window size for event rate calculation, in frames assuming 10Hz (100ms/Frame). Defaults to 5 (500 ms; 200ms before and after).

    #     Returns:
    #         np.ndarray: Event rate (n_cells x n_timepoints)
    #     """
    #     x = np.linspace(-(window_size - 1) / 2, (window_size - 1) / 2, window_size)
    #     sigma = window_size / (2 * np.sqrt(2 * np.log(2)))
    #     kernel = np.exp(-(x**2) / (2 * sigma**2))
    #     kernel = kernel / kernel.sum()  # normalize to sum to 1

    #     # Apply kernel using convolution
    #     eventRate = np.array([convolve(trace, kernel, mode="same") for trace in CTemp])

    #     return eventRate

    def _extract_CnA(
        self,
    ) -> tuple[np.ndarray, sparse.csr_matrix, np.ndarray, np.ndarray]:
        """Extract temporal and spatial components.

        Returns:
            tuple containing:
                - CTemp (np.ndarray): Temporal components (n_cells x n_timepoints)
                - ASpat (sparse.csr_matrix): Spatial components (n_pixels x n_cells)
                - accepted_labels (np.ndarray): Component acceptance status
                - CFrameTimes (np.ndarray): Frame times (n_timepoints)
        """
        CTemp, accepted_labels, CFrameTimes = self._create_CTemp()
        # eventRate = self._create_eventRate(CTemp=CTemp)
        ASpat = self._create_ASpat()

        return CTemp, ASpat, accepted_labels, CFrameTimes

    def _export_segDict(
        self,
        CTemp: np.ndarray | None,
        ASpat: sparse.csr_matrix | None,
        accepted_labels: np.ndarray | None,
        CFrameTimes: np.ndarray | None,
        CtxtByCFrameTimes: np.ndarray | None,
        FrzDict: dict | None,
        CtxtOrder: np.ndarray | None,
    ) -> None:
        """Create and export segmentation dictionary.

        Creates dictionary with filtered components (accepted/undecided cells only).
        Saves in HDF5 and pickle formats.

        Args:
            CTemp: Temporal components (n_cells x n_timepoints)
            ASpat: Spatial components (n_pixels x n_cells)
            accepted_labels: Component acceptance status
            CFrameTimes: Frame times (n_timepoints)
            CtxtByCFrameTimes: Context by frame times (n_timepoints)
            FrzDict: Freezing times dictionary
            CtxtOrder: Context order (n_contexts)

        Raises:
            ValueError: If any input is None
        """

        utils.print_wFrame("Creating segDict", end="", flush=True)

        if CTemp is None or ASpat is None and accepted_labels is None:
            raise ValueError(
                "CTemp, ASpat, or accepted_labels is None. No segDict created/exported."
            )

        accepted_idx = np.where(np.in1d(accepted_labels, self.accepted2use))[0]

        segDict = {
            "C": CTemp[accepted_idx, :],
            "C_all": CTemp,
            "A": ASpat[:, accepted_idx],
            "A_all": ASpat,
            "accepted_labels": accepted_labels,
            "CFrameTimes": CFrameTimes,
        }

        if self.AvB_Check:
            segDict["CtxtByCFrameTimes"] = CtxtByCFrameTimes
            segDict["FrzDict"] = FrzDict
            segDict["CtxtOrder"] = CtxtOrder

        print("...done")

        print("Exporting segDict:")
        utils.saveNloadUtils.savedict2file(
            dict_to_save=segDict,
            dict_name="segDict",
            filename=f"{os.path.basename(self.current_folder)}_segDict",
            filetype_to_save=[self.file_tag["PKL"], self.file_tag["H5"]],
        )

    def _find_folders2convert(self) -> list[str]:
        """Find folders containing ISX CSV files.

        Returns:
            list[str]: Absolute paths to folders with ISX CSV files
        """
        os.chdir(self.file_path)
        folder_contents = os.listdir(self.file_path)
        self.folders2convert = []
        for folder in folder_contents:
            if os.path.isdir(os.path.join(self.file_path, folder)):
                csv_check = bool(self._find_CNMFE_file(folder, "CSV"))
                isx_check = bool(self._find_CNMFE_file(folder, "ISXD"))
                if isx_check or csv_check:
                    self.folders2convert.append(os.path.abspath(folder))

        print(f"Found {len(self.folders2convert)} folders to convert")

        self.folders2convert.sort()
        for idx, folder in enumerate(self.folders2convert):
            utils.print_wFrame(
                f"{idx + 1:02d} - {folder}",
            )
        print()

    def _extractFreezingTimes_Context(self, CFrameTimes: np.ndarray) -> None:
        Frz_keys = ["A", "B"]
        Frz_DF_ALL = {}
        FrzStart = {}
        FrzStop = {}
        FrzDict = {
            "START": [],
            "STOP": [],
            "FRZ_BY_TIME": np.zeros_like(CFrameTimes),
        }

        timeDict = {key: None for key in Frz_keys}
        for key in Frz_keys:
            xlsx_file = utils.findLatest([key, self.file_tag["XLSX"]])
            df = pd.read_excel(xlsx_file, engine="openpyxl")
            time_idx = df[df.iloc[:, 0] == "Start time"].index[0]
            timeDict[key] = np.datetime64(pd.to_datetime(df.iloc[time_idx, 1]))

        # automate the order of the freezing keys
        # if A starts after B, then B is first
        if timeDict["A"] > timeDict["B"]:
            Frz_keys = ["B", "A"]

        CtxtOrder = np.array(Frz_keys, dtype="U1")

        CtxtByCFrameTimes = np.empty_like(CFrameTimes, dtype="U1")

        hp = len(CFrameTimes) // 2
        CtxtByCFrameTimes[0:hp] = Frz_keys[0]
        CtxtByCFrameTimes[hp:] = Frz_keys[1]

        for key in Frz_keys:
            BORIS_file = utils.findLatest(
                [self.file_tag["BORIS"], key.lower() + self.file_tag["CSV"]]
            )
            if BORIS_file:
                df = pd.read_csv(
                    BORIS_file,
                    skiprows=range(14),
                )
                df.columns = df.iloc[0]
                df = df.drop(df.index[0]).reset_index(drop=True)
                Frz_DF_ALL[key] = df
                CtxtAdj = 0 if key == Frz_keys[0] else CFrameTimes[hp]
                FrzStart[key] = (
                    df[df["Status"] == "START"]["Time"].astype(float).values + CtxtAdj
                )
                FrzStop[key] = (
                    df[df["Status"] == "STOP"]["Time"].astype(float).values + CtxtAdj
                )
            else:
                FrzStart[key] = None
                FrzStop[key] = None

        start2use = {}
        stop2use = {}
        for idx, key in enumerate(Frz_keys):
            if FrzStart[key] is not None:
                if key == Frz_keys[0]:
                    start2use[key] = FrzStart[key][FrzStart[key] < CFrameTimes[hp]]
                else:
                    start2use[key] = FrzStart[key][FrzStart[key] < CFrameTimes[-1]]
                stop2use[key] = FrzStop[key][: len(start2use[key])]
                if key == Frz_keys[1] and stop2use[key][-1] > CFrameTimes[-1]:
                    stop2use[key][-1] = CFrameTimes[-1]
                if key == Frz_keys[0] and stop2use[key][-1] > CFrameTimes[hp]:
                    stop2use[key][-1] = CFrameTimes[hp]
            else:
                start2use[key] = None
                stop2use[key] = None

        if start2use[Frz_keys[0]] is not None:
            FrzDict["START"] = np.concatenate(
                [start2use[Frz_keys[0]], start2use[Frz_keys[1]]]
            )
        else:
            FrzDict["START"] = None

        if stop2use[Frz_keys[0]] is not None:
            FrzDict["STOP"] = np.concatenate(
                [stop2use[Frz_keys[0]], stop2use[Frz_keys[1]]]
            )
        else:
            FrzDict["STOP"] = None

        if FrzDict["START"] is not None:
            for idx, (start, stop) in enumerate(zip(FrzDict["START"], FrzDict["STOP"])):
                start = np.floor(start * 10) / 10
                stop = np.ceil(stop * 10) / 10
                indices = np.where((CFrameTimes >= start) & (CFrameTimes <= stop))[0]
                FrzDict["FRZ_BY_TIME"][indices] = 1
        else:
            FrzDict["FRZ_BY_TIME"] = None

        return CtxtByCFrameTimes, FrzDict, CtxtOrder

    def _cycle_through_folders(self) -> None:
        """Process each folder sequentially.

        For each folder:
        1. Extract components
        2. Create and export segDict
        3. Handle any errors and continue

        Raises:
            ValueError: If component extraction fails (caught and logged)
        """

        for folder in self.folders2convert:
            try:
                self.current_folder = folder

                if "AvB" in self.current_folder:
                    self.AvB_Check = True

                os.chdir(self.current_folder)
                print(f"Processing {self.current_folder}")

                CTemp, ASpat, accepted_labels, CFrameTimes = self._extract_CnA()

                if self.AvB_Check:
                    CtxtByCFrameTimes, FrzDict, CtxtOrder = (
                        self._extractFreezingTimes_Context(CFrameTimes=CFrameTimes)
                    )
                else:
                    CtxtByCFrameTimes = None
                    FrzDict = None
                    CtxtOrder = None

                self._export_segDict(
                    CTemp=CTemp,
                    ASpat=ASpat,
                    accepted_labels=accepted_labels,
                    CFrameTimes=CFrameTimes,
                    CtxtByCFrameTimes=CtxtByCFrameTimes,
                    FrzDict=FrzDict,
                    CtxtOrder=CtxtOrder,
                )

                print(f"Finished processing: {self.current_folder}\n\n")

            except ValueError as e:
                print(f"Error processing {self.current_folder}: {e}")
                print("Moving on to next folder...\n\n")
                continue

    @property
    def run(self) -> None:
        """Run the full conversion process.

        1. Find folders to convert
        2. Process each folder:
            - Extract components
            - Create and save segDict
        """
        self._find_folders2convert()
        self._cycle_through_folders()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ISX CSV to segDict")
    parser.add_argument(
        "-p",
        "--path_to_isx",
        default=None,
        type=str,
        help="Path to folder containing session folders, where each session folder contains the CNMFe results files exported from ISX.",
    )
    args = parser.parse_args()

    ISX_csv_converter(args.path_to_isx).run

import os
import pandas as pd
import numpy as np
import zipfile
import tifffile as tif
from scipy import sparse
from rich import print
from CLAH_ImageAnalysis import utils


class ISX_csv_converter:
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.text_lib = utils.text_dict()
        self.file_tag = self.text_lib["file_tag"]
        self.cellmap_tiffs_fname = "cellmap_tiffs"
        self.accepted2use = ["accepted", "undecided"]

    def _find_csv(self, folder: str = None) -> str:
        if folder is not None:
            folder = os.path.abspath(folder)
            os.chdir(folder)

        csv_file = utils.findLatest(self.file_tag["CSV"])

        if folder is not None:
            os.chdir("..")

        return csv_file

    def _create_CTemp(self) -> tuple[np.ndarray, np.ndarray]:
        utils.print_wFrame("Creating CTemp")

        utils.print_wFrame("Importing CSV as dataframe", frame_num=1)
        C_as_CSV = self._import_csvNremove_whitespace()

        utils.print_wFrame("Creating accepted_labels", frame_num=1)
        accepted_labels = C_as_CSV.iloc[0, 1:].values

        utils.print_wFrame(
            "Extracting time course data by cell from dataframe", frame_num=1
        )
        CTemp = np.transpose(C_as_CSV.iloc[1:, 1:].astype(float).values)

        utils.print_wFrame(f"CTemp shape: {CTemp.shape}", frame_num=1)
        utils.print_wFrame(f"Cell number: {CTemp.shape[0]}", frame_num=2)
        utils.print_wFrame(f"Total time: {CTemp.shape[1]}", frame_num=2)
        utils.print_wFrame(
            f"accepted_labels shape: {accepted_labels.shape[0]}", frame_num=1
        )

        return CTemp, accepted_labels

    def _import_csvNremove_whitespace(self) -> pd.DataFrame:
        csv2convert = pd.read_csv(self._find_csv(), low_memory=False)

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
        zip_file = utils.findLatest(self.file_tag["ZIP"])

        output_dir = os.path.join(self.current_folder, self.cellmap_tiffs_fname)
        os.makedirs(output_dir, exist_ok=True)

        utils.print_wFrame(f"Unzipping {zip_file}", frame_num=2, end="", flush=True)

        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(output_dir)

        print("...done")

        utils.print_wFrame(f"Removing {zip_file}", frame_num=2)
        os.remove(zip_file)

        return output_dir

    def _import_tiffStack_N_create_cellmap(self, folderWtiffs: str) -> np.ndarray:
        tiff_files = os.listdir(folderWtiffs)
        tiff_files.sort()
        tiff_files = [file for file in tiff_files if "accepted-cells-map" not in file]

        cellmap = []
        for file in tiff_files:
            with tif.TiffFile(os.path.join(folderWtiffs, file)) as tif_file:
                cellmap.append(tif_file.asarray())

        cellmap = np.stack(cellmap)

        return cellmap

    def _create_ASpat(self) -> sparse.csr_matrix:
        utils.print_wFrame("Creating ASpat")

        cellmap_tiffs_dir = os.path.join(self.current_folder, self.cellmap_tiffs_fname)
        if not os.path.exists(cellmap_tiffs_dir):
            utils.print_wFrame(
                "No cellmap_tiffs directory found. Unzipping is required.", frame_num=1
            )
            tiffStackFolder = self._unzip_tiffStack()
        else:
            utils.print_wFrame(
                "cellmap_tiffs directory found. Unzipping is not required.", frame_num=1
            )
            tiffStackFolder = cellmap_tiffs_dir

        utils.print_wFrame(
            "Importing tiff files & converting to sparse array", frame_num=1
        )
        cellmap = self._import_tiffStack_N_create_cellmap(tiffStackFolder)

        # convert to sparse matrix
        ASpat = self._convert2sparse(cellmap)

        utils.print_wFrame(f"Sparse matrix dimensions: {ASpat.shape}", frame_num=1)

        return ASpat

    def _convert2sparse(self, cellmap: np.ndarray) -> sparse.csr_matrix:
        cell_num, x, y = cellmap.shape
        squeezed_cellmap = np.transpose(cellmap.reshape(cell_num, x * y))

        return sparse.csr_matrix(squeezed_cellmap)

    def _extract_CnA(self) -> tuple[np.ndarray, sparse.csr_matrix, np.ndarray]:
        CTemp, accepted_labels = self._create_CTemp()
        ASpat = self._create_ASpat()

        return CTemp, ASpat, accepted_labels

    def _export_segDict(
        self, CTemp: np.ndarray, ASpat: sparse.csr_matrix, accepted_labels: np.ndarray
    ) -> None:
        utils.print_wFrame("Creating segDict", end="", flush=True)

        accepted_idx = np.where(np.in1d(accepted_labels, self.accepted2use))[0]

        segDict = {
            "C": CTemp[accepted_idx, :],
            "C_raw": CTemp,
            "A": ASpat[:, accepted_idx],
            "A_raw": ASpat,
            "accepted_labels": accepted_labels,
        }
        print("...done")

        print("Exporting segDict:")
        utils.saveNloadUtils.savedict2file(
            dict_to_save=segDict,
            dict_name="segDict",
            filename=f"{os.path.basename(self.current_folder)}_segDict",
            filetype_to_save=[self.file_tag["H5"], self.file_tag["PKL"]],
        )

    def _find_folders2convert(self) -> list[str]:
        os.chdir(self.file_path)
        folder_contents = os.listdir(self.file_path)
        self.folders2convert = []
        for folder in folder_contents:
            if os.path.isdir(os.path.join(self.file_path, folder)):
                if self._find_csv(folder):
                    self.folders2convert.append(os.path.abspath(folder))

    def _cycle_through_folders(self) -> None:
        for folder in self.folders2convert:
            self.current_folder = folder
            os.chdir(self.current_folder)
            print(f"Processing {self.current_folder}")

            CTemp, ASpat, accepted_labels = self._extract_CnA()

            self._export_segDict(CTemp, ASpat, accepted_labels)

            print(f"Finished processing: {self.current_folder}\n\n")

    def run(self) -> None:
        self._find_folders2convert()
        self._cycle_through_folders()


if __name__ == "__main__":
    path_to_isx = "/mnt/DataDrive1/alex/Data_W_Test"
    converter = ISX_converter(path_to_isx)
    converter.run()

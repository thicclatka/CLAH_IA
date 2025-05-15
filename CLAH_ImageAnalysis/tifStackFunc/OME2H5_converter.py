import os
import pandas as pd
import numpy as np
import argparse
import tifffile as tif
import glob
import h5py


from rich import print
from CLAH_ImageAnalysis import utils


class OME2H5_converter:
    def __init__(self, file_path: str | None = None) -> None:
        self.file_path = utils.check_folder_path(
            file_path,
            msg="Select folder containing OME-TIFF files.",
        )

        self.file_path = file_path
        self.text_lib = utils.text_dict()
        self.file_tag = self.text_lib["file_tag"]

    def _find_folders2convert(self) -> list[str]:
        """Find folders containing ISX CSV files.

        Returns:
            list[str]: Absolute paths to folders with OME-TIFF files
        """
        os.chdir(self.file_path)
        folder_contents = os.listdir(self.file_path)
        self.folders2convert = []
        for folder in folder_contents:
            if os.path.isdir(os.path.join(self.file_path, folder)):
                ome_check = utils.findLatest(
                    [self.file_tag["OME"], self.file_tag["CYCLE"]]
                )
                if ome_check:
                    self.folders2convert.append(os.path.abspath(folder))

        print(f"Found {len(self.folders2convert)} folders to convert")

        self.folders2convert.sort()
        for idx, folder in enumerate(self.folders2convert):
            utils.print_wFrame(
                f"{idx + 1:02d} - {folder}",
            )
        print()

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
            print(f"Processing {folder}")
            try:
                os.chdir(folder)
                self.current_folder = folder
                self._convert_ome_to_h5()
                print(f"Finished processing: {folder}")

            except ValueError as e:
                print(f"Error processing {self.current_folder}: {e}")
                print("Moving on to next folder...\n\n")
                continue

    def _convert_ome_to_h5(self) -> None:
        """Convert OME-TIFF to H5.

        For each folder:
        1. Extract components
        2. Create and export segDict
        """
        ome_files = glob.glob(f"{self.file_tag['CYCLE']}*{self.file_tag['OME']}")
        ome_files.sort()
        utils.print_wFrame(f"Found {len(ome_files)} OME-TIFF files")
        for idx, ome_file in enumerate(ome_files):
            utils.print_wFrame(f"{idx + 1:02d} - {ome_file}", frame_num=1)

        initial_file = ome_files[0]
        h5fname = initial_file.replace(self.file_tag["OME"], self.file_tag["H5"])

        image_stack = []
        for ome_file in ome_files:
            image_stack.append(tif.imread(ome_file))

        image_stack = np.array(image_stack)
        image_stack = np.concatenate(image_stack, axis=0)

        with h5py.File(h5fname, "w") as f:
            f.create_dataset("imaging", data=image_stack)

        utils.print_wFrame(f"Created: {h5fname}")

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
    parser = argparse.ArgumentParser(description="Convert OME-TIFF to H5")
    parser.add_argument(
        "-p",
        "--path_to_ome",
        default=None,
        type=str,
        help="Path to folder containing OME-TIFF files.",
    )
    args = parser.parse_args()

    OME2H5_converter(args.path_to_isx).run

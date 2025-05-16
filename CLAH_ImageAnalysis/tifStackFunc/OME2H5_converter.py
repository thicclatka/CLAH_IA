import argparse
import glob
import os

import h5py
import numpy as np
import tifffile as tif
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
        print(f"Searching for OME-TIFF files in {self.file_path}")
        folder_contents = [
            os.path.join(self.file_path, f)
            for f in os.listdir(self.file_path)
            if os.path.isdir(os.path.join(self.file_path, f))
        ]
        self.folders2convert = []
        for folder in folder_contents:
            os.chdir(folder)
            ome_check = utils.findLatest([self.file_tag["CYCLE"], self.file_tag["IMG"]])
            tif_check = utils.findLatest(
                self.file_tag["IMG"], notInclude=[self.file_tag["OME"]]
            )
            if ome_check:
                self.folders2convert.append(folder)
            elif tif_check:
                self.folders2convert.append(folder)

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
                print(f"Finished processing: {folder}\n\n")

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
        h5_file = utils.findLatest([self.file_tag["CYCLE"], self.file_tag["H5"]])
        if h5_file:
            print(f"Found existing H5 file: {h5_file} \nSkipping conversion.")
            return

        ome_files = glob.glob(
            f"*{self.file_tag['CYCLE']}*{self.file_tag['OME']}{self.file_tag['IMG']}"
        )

        if len(ome_files) == 0:
            ome_files = glob.glob(f"*{self.file_tag['IMG']}")
        elif len(ome_files) > 1:
            ome_files.sort()

        utils.print_wFrame(f"Found {len(ome_files)} OME-TIFF files")
        for idx, ome_file in enumerate(ome_files):
            utils.print_wFrame(f"{idx + 1:02d} - {ome_file}", frame_num=1)

        utils.print_wFrame("Each stack size pre concatenation:")

        initial_file = ome_files[0]

        h5fname = initial_file.split(".")[0] + self.file_tag["H5"]
        if self.file_tag["CYCLE"] not in initial_file:
            h5fname = (
                initial_file.split(".")[0]
                + self.file_tag["CYCLE"]
                + self.file_tag["CODE"]
                + self.file_tag["H5"]
            )

        image_stack = []
        for idx, ome_file in enumerate(ome_files):
            curr_stack = tif.imread(ome_file)
            utils.print_wFrame(f"{idx + 1:02d} - {curr_stack.shape}", frame_num=1)
            image_stack.append(curr_stack)

        # image_stack is a list of (frames, x, y) arrays
        # Concatenate them along the first axis (frames)
        image_stack = np.concatenate(image_stack, axis=0)

        utils.print_wFrame(f"New Stack Size after concatenation: {image_stack.shape}")

        utils.print_wFrame("Exporting to H5")
        with h5py.File(h5fname, "w") as f:
            f.create_dataset("imaging", data=image_stack)

        utils.print_wFrame(f"Created: {h5fname}", frame_num=1)

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
        help="Path to folder containing session directories with OME-TIFF files.",
    )
    args = parser.parse_args()

    OME2H5_converter(args.path_to_ome).run

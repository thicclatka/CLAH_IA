import argparse
import os

# import shutil
from datetime import datetime

from rich import print

from CLAH_ImageAnalysis import utils


class ISX_folderRenamer:
    """ """

    def __init__(self, folder_path: str | None = None) -> None:
        self.main_parent_folder = utils.check_folder_path(
            folder_path,
            msg="Select directory of directories of session folders.",
        )
        self.text_lib = utils.text_dict()
        self.file_tag = self.text_lib["file_tag"]

        self.folder_paths = [
            os.path.join(self.main_parent_folder, folder)
            for folder in os.listdir(self.main_parent_folder)
        ]

    @staticmethod
    def _print_brk() -> None:
        utils.section_breaker("hash", mini=True)

    def _find_folders2rename(self) -> None:
        self.folders2rename = []
        for fpath in self.folder_paths:
            self.folders2rename.extend(
                utils.find_eligible_folders_w_ftag(self.file_tag["ISXD"], fpath)
            )
        self.folders2rename.sort()

    def _cycle_through_folders(self) -> None:
        self._print_brk()
        for folder in self.folders2rename:
            try:
                self.current_folder = folder
                self.currParent_current_folder = os.path.dirname(folder)

                os.chdir(self.currParent_current_folder)

                rename_check, proper_name_check = self._check_folder4previous_renaming()

                if not rename_check and proper_name_check:
                    print(f"Renaming {self.current_folder}")
                    new_folder_name = self.parsing_folder_name()
                    new_folder_name = os.path.join(
                        self.currParent_current_folder, new_folder_name
                    )
                    os.rename(self.current_folder, new_folder_name)
                    print(f"Finished renaming to: {new_folder_name}")
                elif rename_check and proper_name_check:
                    print(
                        f"Skipping {self.current_folder} as it has already been renamed to proper format"
                    )
                elif not rename_check and not proper_name_check:
                    print(
                        f"Skipping {self.current_folder} since it does not have enough elements to be renamed"
                    )
                self._print_brk()

            except ValueError as e:
                print(f"Error processing {self.current_folder}: {e}")
                print("Moving on to next folder...")
                self._print_brk()
                continue

    @staticmethod
    def _get_folder_strings(fpath: str) -> list[str]:
        return os.path.basename(fpath).split("_")

    def _check_folder4previous_renaming(self) -> tuple[bool, bool]:
        folder_strings = self._get_folder_strings(self.current_folder)
        if len(folder_strings) == 3:
            # Assumes that the 3 strings are the date, subject, and experiment strings
            return True, True
        elif len(folder_strings) < 3:
            return False, False
        elif len(folder_strings) > 3:
            return False, True

    def parsing_folder_name(self) -> None:
        def remove_string_from_folder_strings(
            folder_strings: list[str], string: str
        ) -> list[str]:
            return [f for f in folder_strings if f != string]

        def extract_date_string(folder_strings: list[str]) -> str:
            date_string = folder_strings[-1]
            new_date_string = str(
                datetime.strptime(date_string, "%Y%m%d-%H%M%S").strftime("%y%m%d")
            )
            folder_strings = remove_string_from_folder_strings(
                folder_strings, date_string
            )
            return folder_strings, new_date_string

        def extract_cohort_string(folder_strings: list[str]) -> str:
            cohort_string = folder_strings[0]
            folder_strings = remove_string_from_folder_strings(
                folder_strings, cohort_string
            )
            return folder_strings, cohort_string

        def extract_subjID(folder_strings: list[str]) -> str:
            if "Ctx" in folder_strings[-1]:
                subjID_string = folder_strings[-2]
            else:
                subjID_string = folder_strings[-1]
            folder_strings = remove_string_from_folder_strings(
                folder_strings, subjID_string
            )
            return folder_strings, subjID_string

        folder_strings = self._get_folder_strings(self.current_folder)
        folder_strings, date_string = extract_date_string(folder_strings)
        folder_strings, cohort_string = extract_cohort_string(folder_strings)
        folder_strings, subjID_string = extract_subjID(folder_strings)

        subject_string = f"{cohort_string}-{subjID_string}"

        exp_strings = "-".join(folder_strings)

        new_folder_name = f"{date_string}_{subject_string}_{exp_strings}"

        return new_folder_name

    @property
    def run(self) -> None:
        """Run the full conversion process.

        1. Find folders to convert
        2. Process each folder:
            - Extract components
            - Create and save segDict
        """
        self._find_folders2rename()
        self._cycle_through_folders()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Change ISX folder names so that they are in the correct format for CLAH_ImageAnalysis"
    )
    parser.add_argument(
        "-p",
        "--path_to_isx",
        default=None,
        type=str,
        help="Path to directory of directories of session folders.",
    )
    args = parser.parse_args()

    ISX_folderRenamer(args.path_to_isx).run

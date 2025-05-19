import argparse
import json
import os
import sys

import cv2
import easygui
import isx
import numpy as np
from tqdm import tqdm

from CLAH_ImageAnalysis.utils import (
    folder_tools,
    print_done_small_proc,
    print_header,
    print_wFrame,
    text_dict,
)


class MovieCropper:
    def __init__(self, file_path: str, running_main: bool = False):
        def get_file_path():
            while True:
                self.file_path = easygui.fileopenbox(
                    title="Select ISX file to crop",
                    filetypes=[f"*{self.isxd_tag}"],
                )
                if self.file_path is not None:
                    if self.file_path.endswith(self.isxd_tag):
                        break
                    else:
                        print_wFrame(
                            "File selected is not an ISXD file. Please try again.\n"
                        )
                else:
                    print_wFrame("No file selected. Please try again.\n")
                    print()

        print_header(
            "Movie Cropping for 1 Photon Imaging"
        ) if not running_main else None

        self.isxd_tag = text_dict()["file_tag"]["ISXD"]

        self.file_path = file_path
        if self.file_path is None:
            get_file_path()
        elif self.file_path is not None:
            if os.path.isdir(self.file_path):
                os.chdir(self.file_path)
                isxd_file = folder_tools.findLatest(self.isxd_tag)
                if not isxd_file:
                    print("No ISXD file found in directory. Please select file.")
                    get_file_path()
                else:
                    self.file_path = isxd_file

        self.crop_coords = []

        self.parentDir = os.path.dirname(self.file_path)
        self.fname = os.path.basename(self.file_path)
        # self.cropped_movie_name = self.fname.split(".")[0] + "_CROPPED" + self.isxd_tag

        # self.full_path_cropped_movie = os.path.join(
        #     self.parentDir, self.cropped_movie_name
        # )

        # if os.path.exists(self.full_path_cropped_movie):
        #     os.remove(self.full_path_cropped_movie)

        print(f"Running cropping utility on file: {self.fname}")

    @property
    def run(self):
        self.load_isx()
        self.run_cropping()
        # self.crop_movieNexport2ISXD()
        self.export_crop_coords()

    def load_isx(self):
        print("ISXD information:")
        self.movie = isx.Movie.read(self.file_path)
        self.total_frames = self.movie.timing.num_samples
        self.data_type = self.movie.data_type
        self.timing = self.movie.timing
        self.spacing = self.movie.spacing

        info_list = [
            f"File: {self.fname}",
            f"Total frames: {self.total_frames}",
            # f"Data type: {self.data_type}",
            # f"Timing: {self.timing}",
            # f"Spacing: {self.spacing}",
        ]

        for info in info_list:
            print_wFrame(info)
        print()

    def run_cropping(self):
        def crop_image(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.crop_coords = [(x, y)]  # Start point
                self.drawing = True
                self.display_img = self.img.copy()
                cv2.rectangle(
                    self.display_img, self.crop_coords[0], (x, y), (0, 0, 255), 2
                )
                cv2.imshow(curr_window_name, self.display_img)
            elif event == cv2.EVENT_MOUSEMOVE:
                if self.drawing:
                    # Make a fresh copy of the image
                    self.display_img = self.img.copy()
                    # Draw rectangle from start point to current position
                    cv2.rectangle(
                        self.display_img, self.crop_coords[0], (x, y), (0, 0, 255), 2
                    )
                    cv2.imshow(curr_window_name, self.display_img)

            elif event == cv2.EVENT_LBUTTONUP:
                self.drawing = False
                self.crop_coords.append((x, y))  # End point
                self.display_img = self.img.copy()
                cv2.rectangle(
                    self.display_img,
                    self.crop_coords[0],
                    self.crop_coords[1],
                    (0, 0, 255),
                    2,
                )
                cv2.imshow(curr_window_name, self.display_img)

        def window_name():
            return f"Image Cropper - Frame: {self.frame_idx:03d}"

        def create_cv2_window(window):
            cv2.namedWindow(window)
            cv2.setMouseCallback(window, crop_image)

        def reset_cv2_window(window):
            cv2.destroyAllWindows()
            create_cv2_window(window)

        def normalize_frame(frame):
            normalized_frame = cv2.normalize(
                frame, None, 0, np.iinfo(np.uint8).max, cv2.NORM_MINMAX
            ).astype(np.uint8)
            return cv2.cvtColor(normalized_frame, cv2.COLOR_GRAY2BGR)

        print("Cropping how to:")
        how_to_list = [
            "Press 'r' to reset the crop",
            "Press 'c' to confirm the crop",
            "Press 'n' to go to next frame",
            "Press 'p' to go to previous frame",
            "Press 'q' to quit without cropping",
        ]
        for how_to in how_to_list:
            print_wFrame(how_to)

        self.frame_idx = 0
        self.drawing = False

        first_frame = normalize_frame(self.movie.get_frame_data(self.frame_idx))
        self.img = first_frame.copy()
        self.display_img = self.img.copy()

        curr_window_name = window_name()
        create_cv2_window(curr_window_name)

        while True:
            cv2.imshow(curr_window_name, self.display_img)
            key = cv2.waitKey(1) & 0xFF

            # Press 'r' to reset the crop
            if key == ord("r"):
                self.frame_idx = 0
                self.display_img = first_frame.copy()
                self.crop_coords = []
                curr_window_name = window_name()
                reset_cv2_window(curr_window_name)

            # Press 'c' to confirm the crop and exit
            elif key == ord("c"):
                break

            # Press 'q' to quit without cropping
            elif key == ord("q"):
                self.crop_coords = []
                break

            elif key == ord("n"):
                if self.frame_idx < self.total_frames - 1:
                    self.frame_idx += 1
                    self.img = normalize_frame(
                        self.movie.get_frame_data(self.frame_idx)
                    )
                    self.display_img = self.img.copy()
                    curr_window_name = window_name()
                    reset_cv2_window(curr_window_name)

            elif key == ord("p"):
                if self.frame_idx > 0:
                    self.frame_idx -= 1
                    self.img = normalize_frame(
                        self.movie.get_frame_data(self.frame_idx)
                    )
                    self.display_img = self.img.copy()
                    curr_window_name = window_name()
                    reset_cv2_window(curr_window_name)

        cv2.destroyAllWindows()

        if self.crop_coords:
            print_wFrame(
                "Cropping complete. Exporting dimensions to JSON for motion correction to use."
            )
            # print_done_small_proc()
        else:
            print_wFrame("No crop selected. Exiting.")
            print()
            sys.exit()

    def export_crop_coords(self):
        json_fname = os.path.join(self.parentDir, text_dict()["file_tag"]["CROP_DIMS"])
        if self.crop_coords:
            with open(json_fname, "w") as f:
                json.dump(self.crop_coords, f)
            print_wFrame(f"Crop coordinates exported to: {json_fname}")
            print_done_small_proc()

    def crop_movieNexport2ISXD(self):
        if self.crop_coords:
            x1, y1 = self.crop_coords[0]
            x2, y2 = self.crop_coords[1]

            left = min(x1, x2)
            right = max(x1, x2)
            top = min(y1, y2)
            bottom = max(y1, y2)

            width = right - left
            height = bottom - top

            print(f"Original frame shape: {self.movie.spacing.num_pixels}")
            print(f"Crop dimensions: ({height}, {width})")

            cropped_movie = isx.Movie.write(
                self.full_path_cropped_movie,
                self.timing,
                isx.Spacing(
                    num_pixels=(height, width),
                ),
                self.data_type,
            )
            for frame_idx in tqdm(
                range(self.total_frames), desc="Cropping movie & exporting to isxd"
            ):
                frame = self.movie.get_frame_data(frame_idx)
                cropped_frame = frame[
                    top:bottom,
                    left:right,
                ]
                cropped_movie.set_frame_data(frame_idx, cropped_frame)

            cropped_movie.flush()

            print("Cropping & exporting complete:")
            print_wFrame(f"File saved as: {self.cropped_movie_name}")
            print_done_small_proc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-fp",
        "--file_path",
        default=None,
        type=str,
        help="Path to isxd file to crop. If directory is provided, will search for latest ISXD file in directory. If no file is found, will prompt user to select file.",
    )
    args = parser.parse_args()

    MovieCropper(args.file_path, running_main=True).run

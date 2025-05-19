import json
import os

import cv2
import isx
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_cropper import st_cropper

from CLAH_ImageAnalysis.utils import Streamlit_utils, findLatest


class MovieCropper:
    def __init__(self, file_path: str = None):
        self.isxd_tag = ".isxd"
        self.file_path = file_path
        self.crop_coords = []
        self.crop_dims_json = "crop_dims.json"

    def load_isx(self):
        self.movie = isx.Movie.read(self.file_path)
        self.total_frames = self.movie.timing.num_samples
        self.data_type = self.movie.data_type
        self.timing = self.movie.timing
        self.spacing = self.movie.spacing

    def normalize_frame(self, frame):
        normalized_frame = cv2.normalize(
            frame, None, 0, np.iinfo(np.uint8).max, cv2.NORM_MINMAX
        ).astype(np.uint8)
        return cv2.cvtColor(normalized_frame, cv2.COLOR_GRAY2BGR)

    def export_crop_coords(self):
        json_fname = os.path.join(os.path.dirname(self.file_path), self.crop_dims_json)
        if self.crop_coords:
            with open(json_fname, "w") as f:
                json.dump(self.crop_coords, f)
            st.success(f"Crop coordinates exported to: {json_fname}")

    @staticmethod
    def list_isxd_files(directory):
        """List all .isxd files in the given directory"""
        isxd_files = []
        try:
            for file in os.listdir(directory):
                if file.endswith(".isxd") and "CNMF" not in file:
                    isxd_files.append(os.path.join(directory, file))
        except Exception as e:
            st.error(f"Error accessing directory: {e}")
        return isxd_files


def init_session_state():
    # Initialize session state variables for coordinates
    if "start_point" not in st.session_state:
        st.session_state.start_point = None
    if "end_point" not in st.session_state:
        st.session_state.end_point = None
    if "drawing" not in st.session_state:
        st.session_state.drawing = False


def main():
    Streamlit_utils.setup_page_config(
        app_name="Movie Cropper for 1 Photon Imaging",
        app_abbrv="Cropper",
        init_session_state_func=init_session_state,
    )

    # Directory navigation
    if "current_dir" not in st.session_state:
        # st.session_state.current_dir = os.getcwd()
        st.session_state.current_dir = "/"

    # Show current directory
    st.text(f"Current directory: {st.session_state.current_dir}")

    # Directory navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬆️ Up one level"):
            st.session_state.current_dir = os.path.dirname(st.session_state.current_dir)
            # selected = setup_fb()
            st.rerun()

    with col2:
        new_dir = st.text_input("Enter directory path:", st.session_state.current_dir)
        if new_dir != st.session_state.current_dir and os.path.isdir(new_dir):
            st.session_state.current_dir = new_dir
            # selected = setup_fb()
            st.rerun()

    try:
        # List ISXD files
        isxd_files = MovieCropper.list_isxd_files(st.session_state.current_dir)
        if not isxd_files:
            selected_file = []
            st.warning(
                "No ISXD files found in current directory. Either go up one level or select a subdirectory."
            )
            current_entries = [
                d
                for d in os.listdir(st.session_state.current_dir)
                if not d.startswith(".")
            ]
            current_entries.sort()
            if st.session_state.current_dir == "/":
                current_entries = [d for d in current_entries if d in ["home", "mnt"]]
            st.write("Current entries:")
            for entry in current_entries:
                full_path = os.path.join(st.session_state.current_dir, entry)
                if os.path.isdir(full_path):
                    # Directory with folder icon
                    if st.button(
                        f"📁 {entry}", key=f"dir_{entry}", use_container_width=True
                    ):
                        st.session_state.current_dir = full_path
                        st.rerun()
                else:
                    # File with file icon
                    if entry.endswith(".isxd") and "CNMF" not in entry:
                        if st.button(
                            f"📄 {entry}", key=f"file_{entry}", use_container_width=True
                        ):
                            st.session_state.selected_file = full_path
        else:
            if len(isxd_files) == 1:
                st.write(f"Found ISXD file: {os.path.basename(isxd_files[0])}")
                selected_file = isxd_files[0]
            else:
                selected_file = st.selectbox(
                    "Found multiple ISXD files. Please select one:", isxd_files
                )
            st.write(
                "To select a different file or start over, click the '⬆️ Up one level' button."
            )

        if selected_file:
            #     selected_file = selected["path"]
            #     st.write(f"selected_file: {selected_file}")

            cropper = MovieCropper(selected_file)
            cropper.load_isx()

            # Frame selection slider
            frame_idx = st.slider("Select Frame", 0, cropper.total_frames - 1, 0)

            # Get and normalize frame
            frame = cropper.normalize_frame(cropper.movie.get_frame_data(frame_idx))

            # Convert to RGB for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert to PIL Image
            pil_image = Image.fromarray(frame_rgb)

            # Cropper options
            # st.sidebar.header("Cropper Options")
            # realtime_update = st.sidebar.checkbox("Update in Real Time", value=True)
            # box_color = st.sidebar.color_picker("Box Color", value="#00FF00")
            # aspect_choice = st.sidebar.radio(
            #     "Aspect Ratio", ["Free", "1:1", "16:9", "4:3", "2:3"]
            # )
            # aspect_dict = {
            #     "1:1": (1, 1),
            #     "16:9": (16, 9),
            #     "4:3": (4, 3),
            #     "2:3": (2, 3),
            #     "Free": None,
            # }
            # aspect_ratio = aspect_dict[aspect_choice]

            # Get cropped image
            cropped_img = st_cropper(
                pil_image,
                return_type="box",
                # realtime_update=realtime_update,
                # box_color=box_color,
                # aspect_ratio=aspect_ratio,
            )

            if cropped_img:
                # Extract coordinates from the dictionary
                left = int(cropped_img["left"])
                top = int(cropped_img["top"])
                width = int(cropped_img["width"])
                height = int(cropped_img["height"])

                # Calculate bottom-right coordinates
                x1, y1 = left, top
                x2, y2 = left + width, top + height

                # Display coordinates
                st.write(f"Crop Coordinates: ({x1}, {y1}) to ({x2}, {y2})")

                # Export button
                if st.button("Export Crop Coordinates"):
                    cropper.crop_coords = [(x1, y1), (x2, y2)]
                    cropper.export_crop_coords()

    except Exception as e:
        st.error(f"Error in st_canvas: {str(e)}")
        st.error(f"Error location: {e.__traceback__.tb_lineno}")
        raise e


if __name__ == "__main__":
    main()

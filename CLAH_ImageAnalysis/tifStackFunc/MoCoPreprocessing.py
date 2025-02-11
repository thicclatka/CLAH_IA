import isx
import os
import json
import numpy as np
from skimage import img_as_uint
from tqdm import tqdm
from PIL import Image
from CLAH_ImageAnalysis.core import BaseClass as BC
from CLAH_ImageAnalysis.tifStackFunc import MovieCropper

# from skimage import exposure
# import h5py


class MoCoPreprocessing(BC):
    def __init__(self) -> None:
        self.program_name = "ISXD"
        self.class_type = "utils"
        BC.__init__(self, program_name=self.program_name, mode=self.class_type)

    def read_file(self, file_path: str) -> None:
        """
        Reads an .isxd or .imu/.gpio file and returns the data.

        Parameters:
            file_path (str): The path to the file to be read.

        Returns:
            data (isx.Movie or isx.GpioSet): The data from the file.
        """
        if file_path.endswith(self.file_tag["ISXD"]):
            data = isx.Movie.read(file_path)
        elif file_path.endswith(self.file_tag["IMU"] or self.file_tag["GPIO"]):
            data = isx.GpioSet.read(file_path)
        elif file_path.endswith(self.file_tag["TIFF"]):
            data = self.read_image_TIFF(file_path)
        else:
            raise ValueError(f"File type not supported: {file_path}")

        return data

    def read_image_TIFF(self, file_path: str) -> np.ndarray:
        """
        Reads a TIFF file and returns the data as a numpy array.
        """
        with Image.open(file_path) as img:
            images = []
            for i in range(img.n_frames):
                img.seek(i)
                images.append(np.array(img))
        return np.array(images)

    def get_movie_data(self, file_path: str) -> tuple[int, tuple]:
        """
        Gets the movie data from an ISXD file.

        Parameters:
            file_path (str): The path to the file to be read.

        Returns:
            frames (int): The number of frames in the movie.
            dim (tuple): The dimensions of the movie.
        """

        if file_path.endswith(self.file_tag["TIFF"]):
            with Image.open(file_path) as img:
                frames = img.n_frames
                movie_dims = (frames, *img.size)
        else:
            movie = self.read_file(file_path)
            movie_dims = (movie.timing.num_samples, *movie.spacing.num_pixels)
            frames = movie_dims[0]

        return frames, movie_dims

    def _remove_minimum_value(self, frame: np.ndarray, global_max: float) -> np.ndarray:
        """
        Removes the minimum value from a frame to remove glow/vignette effect.

        Parameters:
            frame (np.ndarray): The frame to be processed.
            global_max (float): The global maximum value of the frame.

        Returns:
            np.ndarray: The processed frame in 16bit.
        """
        return img_as_uint((frame - frame.min()) / (global_max - frame.min()))

    def _apply_background_filter(
        self, frame: np.ndarray, gSig_filt: tuple
    ) -> np.ndarray:
        """
        Applies a high-pass filter to a frame.

        Parameters:
            frame (np.ndarray): The frame to be processed.
            gSig_filt (tuple): The size of the high-pass filter kernel.

        Returns:
            np.ndarray: The processed frame.
        """
        frame_filtered = self.utils.caiman_utils.apply_high_pass_filter_space(
            frame, gSig_filt=gSig_filt
        )
        frame_filtered = self.dep.normalization_utils.normalize_array(
            frame_filtered, dtype=np.uint16
        )
        return frame_filtered

    def _apply_bandpass_filter(
        self,
        frame: np.ndarray,
        low_cutoff_freq: float,
        high_cutoff_freq: float,
    ) -> np.ndarray:
        """
        Applies a bandpass filter to a frame.

        Parameters:
            frame (np.ndarray): The frame to be processed.
            low_cutoff_freq (float): The low cutoff frequency. Units are pixels^-1.
            high_cutoff_freq (float): The high cutoff frequency. Units are pixels^-1.

        Returns:
            np.ndarray: The processed frame.
        """
        frame_filtered = self.dep.filter_utils.apply_bandpass_filter(
            frame, low_cutoff_freq, high_cutoff_freq
        )
        frame_filtered = self.dep.normalization_utils.normalize_array(
            frame_filtered, dtype=np.uint16
        )
        return img_as_uint(frame_filtered)

    def _apply_median_blur_filter(
        self, frame: np.ndarray, window_size: int
    ) -> np.ndarray:
        """
        Applies a median blur filter to a frame.

        Parameters:
            frame (np.ndarray): The frame to be processed.
            window_size (int): The size of the median blur filter window.

        Returns:
            np.ndarray: The processed frame.
        """
        if frame.dtype != np.uint16:
            frame = img_as_uint(frame)
        return self.dep.filter_utils.apply_median_blur_filter(frame, window_size)

    def preprocessing_movie(
        self,
        movie_fp: str,
        output_fname: str,
        use_cropper: bool = False,
        temporal_downsample: int = 1,
        apply_bandpass: bool = True,
        apply_high_pass: bool = True,
        apply_CLAHE: bool = True,
        low_cutoff_freq: float = 0.005,
        high_cutoff_freq: float = 0.10,
        window_size: int = 3,
        gSig_filt: tuple = (2, 2),
        hp_threshold: float = 0.05,
        CLAHE_clip_limit: float = 2.0,
        # frame_threshold: int = 10000,
    ) -> np.ndarray:
        """
        Preprocesses a movie. Involves subtracting the minimum value to remove glow/vignette effect, applying a bandpass filter to isolate features of interest, applying a median blur filter to denoise, applying a high-pass filter to remove low-frequency signal, and applying CLAHE to enhance contrast.

        Parameters:
            movie_fp (str): The path to the movie file.
            output_fname (str): The name of the output file.
            use_cropper (bool): Whether to use the cropping utility. Defaults to False.
            temporal_downsample (int): The amount to downsample the movie by. Units are frames. Defaults to 4, i.e. only using every 4th frame.
            apply_bandpass (bool): Whether to apply a bandpass filter. Defaults to True.
            apply_high_pass (bool): Whether to apply a high-pass filter. Defaults to True.
            apply_CLAHE (bool): Whether to apply a CLAHE filter. Defaults to True.
            low_cutoff_freq (float): The low cutoff frequency for the bandpass filter. Units are pixels^-1.
            high_cutoff_freq (float): The high cutoff frequency for the bandpass filter. Units are pixels^-1.
            hp_threshold (float): The threshold for the global maximum to determine if a high-pass filter is needed.
            CLAHE_clip_limit (float): The clip limit for the CLAHE filter.

        Returns:
            filtered_arr (np.ndarray): The preprocessed movie.
        """

        # def determine_frame_threshold(frame_shape: tuple) -> bool:
        #     return frame_shape[0] > frame_threshold

        def determine_threshold(global_max: float) -> bool:
            return float(global_max / np.iinfo(np.uint16).max) < hp_threshold

        def determine_cropping() -> tuple[bool, list]:
            if os.path.exists(self.file_tag["CROP_DIMS"]):
                with open(self.file_tag["CROP_DIMS"], "r") as f:
                    crop_coords = json.load(f)
                return True, crop_coords
            else:
                return False, None

        if use_cropper:
            print("Cropping utility will be used...")
            MovieCropper(file_path=movie_fp).run

        apply_crop, crop_coords = determine_cropping()

        if apply_crop:
            left, top, right, bottom = (
                self.utils.image_utils.extract_LRTB_from_crop_coords(crop_coords)
            )

        bool_list = [
            ("BAND PASS FILTER", apply_bandpass),
            ("HIGH PASS FILTER", apply_high_pass),
            ("CLAHE", apply_CLAHE),
            ("CROPPING", apply_crop),
        ]

        movie = self.read_file(movie_fp)

        raw_movie_fname = movie_fp.split(".")[0] + self.file_tag["MP4"]

        if not os.path.exists(raw_movie_fname):
            self.print_wFrm(f"Exporting raw movie to {raw_movie_fname}")
            isx.export_movie_to_mp4(movie_fp, raw_movie_fname)
        else:
            self.print_wFrm(f"Raw movie already exists at {raw_movie_fname}")

        if movie_fp.endswith(self.file_tag["TIFF"]):
            total_frames = movie.shape[0]
            sample_frame = movie[0, :, :]
        else:
            total_frames = movie.timing.num_samples
            sample_frame = movie.get_frame_data(0)
        output_fname += "_PREPROCESSED" + self.file_tag["SQZ"] + self.file_tag["H5"]

        # resize to square, round pixels to nearest multiple of 50
        # also downsamples given the size of the image

        if apply_crop:
            sample_frame = sample_frame[top:bottom, left:right]

        final_output = self.utils.image_utils.resize_to_square(
            sample_frame, round_to=50
        )
        final_shape = final_output.shape

        frame_step = temporal_downsample
        frames2use = total_frames

        frame_range = range(0, total_frames, frame_step)

        filtered_arr = np.empty((frames2use, *final_shape), dtype=np.uint16)

        global_min = float("inf")
        global_max = float("-inf")
        # z_min = np.inf * np.ones_like(sample_frame)
        # z_max = np.zeros_like(sample_frame)

        pbar = tqdm(frame_range, desc="Finding global max & min")

        for frame_idx in pbar:
            if movie_fp.endswith(self.file_tag["TIFF"]):
                curr_frame = movie[frame_idx]
            else:
                curr_frame = movie.get_frame_data(frame_idx)

            global_min = min(global_min, curr_frame.min())
            global_max = max(global_max, curr_frame.max())
            # z_min = np.minimum(z_min, curr_frame)
            # z_max = np.maximum(z_max, curr_frame)
            pbar.set_postfix({"Max": global_max, "Min": global_min})

        print(
            f"Global max: {global_max} / {global_max / np.iinfo(np.uint16).max:.4f} of uint16 range"
        )

        print("Preprocessing steps:")
        for step in bool_list:
            self.print_wFrm(f"Applying {step[0]}: {step[1]}")

        for frame_idx in tqdm(frame_range, desc="Preprocessing movie"):
            if movie_fp.endswith(self.file_tag["TIFF"]):
                frame = movie[frame_idx]
            else:
                frame = movie.get_frame_data(frame_idx).astype(float)

            if apply_crop:
                frame = frame[top:bottom, left:right]

            # remove minimum value to remove glow/vignette effect
            # convert to 16bit
            frame2use = self._remove_minimum_value(frame, global_max)

            # use bandpass filter to isolate features of interest
            if apply_bandpass:
                frame2use = self._apply_bandpass_filter(
                    frame2use, low_cutoff_freq, high_cutoff_freq
                )

            # denoise frame with median blur filter
            frame2use = self._apply_median_blur_filter(frame2use, window_size)

            # use high-pass filter to remove low-frequency signal
            # use kernel size of 2
            # via caiman funcs
            if apply_high_pass:
                frame2use = self._apply_background_filter(frame2use, gSig_filt)

            # apply CLAHE to enhance contrast
            if apply_CLAHE:
                frame2use = self.dep.apply_CLAHE(frame2use, clip_limit=CLAHE_clip_limit)

            # normalize & keep as 16bit
            frame2use = (frame2use - frame2use.min()) / (
                frame2use.max() - frame2use.min()
            )
            frame2use = img_as_uint(frame2use)

            # resize to square, round pixels to nearest multiple of 50
            frame2use = self.utils.image_utils.resize_to_square(frame2use, round_to=50)

            filtered_arr[frame_idx] = frame2use

        # give artifact in the beginning of the array
        # make first 5 frames the same as the 5th frame
        filtered_arr[:5] = filtered_arr[5]

        return filtered_arr, apply_bandpass, apply_high_pass, apply_CLAHE, apply_crop

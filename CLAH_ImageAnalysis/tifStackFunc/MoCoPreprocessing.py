import isx
import os
import json
import numpy as np
from skimage import img_as_uint
from tqdm import tqdm
from multiprocessing import cpu_count
from multiprocessing import Pool
from functools import partial
from PIL import Image
from CLAH_ImageAnalysis.core import BaseClass as BC
from CLAH_ImageAnalysis.tifStackFunc import MovieCropper
from CLAH_ImageAnalysis.utils import print_wFrame
from CLAH_ImageAnalysis.utils import section_breaker
from CLAH_ImageAnalysis.dependencies import filter_utils
from CLAH_ImageAnalysis.dependencies import normalization_utils
from CLAH_ImageAnalysis.utils import caiman_utils
from CLAH_ImageAnalysis.utils import image_utils
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

    @staticmethod
    def _remove_minimum_value(frame: np.ndarray, global_min: float) -> np.ndarray:
        """
        Removes the global minimum value from a frame to remove glow/vignette effect.

        Parameters:
            frame (np.ndarray): The frame to be processed.
            global_min (float): The global minimum value of the frame.

        Returns:
            np.ndarray: The processed frame in 16bit.
        """
        frame_centered = frame - np.nanmean(frame)
        frame_positive = frame_centered - global_min
        return frame_positive

    @staticmethod
    def _apply_background_filter(frame: np.ndarray, gSig_filt: tuple) -> np.ndarray:
        """
        Applies a high-pass filter to a frame.

        Parameters:
            frame (np.ndarray): The frame to be processed.
            gSig_filt (tuple): The size of the high-pass filter kernel.

        Returns:
            np.ndarray: The processed frame.
        """
        frame_filtered = caiman_utils.apply_high_pass_filter_space(
            frame, gSig_filt=gSig_filt
        )
        frame_filtered = normalization_utils.normalize_array(
            frame_filtered, dtype=np.uint16
        )
        return frame_filtered

    @staticmethod
    def _apply_bandpass_filter(
        frame: np.ndarray,
        low_cutoff_freq: float | None = None,
        high_cutoff_freq: float | None = None,
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
        frame_filtered = filter_utils.apply_bandpass_filter(
            image=frame,
            low_cutoff_freq=low_cutoff_freq,
            high_cutoff_freq=high_cutoff_freq,
            use_spatial=True,
            normalize=False,
        )
        return frame_filtered

    @staticmethod
    def _apply_median_blur_filter(frame: np.ndarray, window_size: int) -> np.ndarray:
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
        return filter_utils.apply_median_blur_filter(frame, window_size)

    @staticmethod
    def _fix_defective_pixels(
        frame: np.ndarray, threshold_std: float = 3.0
    ) -> np.ndarray:
        """
        Fixes defective pixels in a frame.

        Parameters:
            frame (np.ndarray): The frame to be processed.
            threshold_std (float): Number of standard deviations to consider a pixel defective.
            window_size (int): Size of the window to check around each pixel.

        Returns:
            np.ndarray: The processed frame.
        """
        return filter_utils.detect_and_fix_defective_pixels(
            frame,
            threshold_std=threshold_std,
        )

    def preprocessing_movie(
        self,
        movie_fp: str,
        output_fname: str,
        use_cropper: bool = False,
        temporal_downsample: int = 1,
        apply_bandpass: bool = True,
        apply_high_pass: bool = False,
        apply_CLAHE: bool = True,
        fix_defective_pixels: bool = False,
        # low_cutoff_freq: float = 0.005,
        low_cutoff_freq: float | None = None,
        high_cutoff_freq: float = 0.2,
        window_size: int = 3,
        gSig_filt: tuple = (2, 2),
        CLAHE_clip_limit: float = 2.0,
        num_workers: int | None = None,
    ) -> np.ndarray:
        """
        Preprocesses a movie. Involves fixing defective pixels, subtracting the minimum value to remove glow/vignette effect,
        applying a bandpass filter to isolate features of interest, applying a median blur filter to denoise,
        applying a high-pass filter to remove low-frequency signal, and applying CLAHE to enhance contrast.

        Parameters:
            movie_fp (str): The path to the movie file.
            output_fname (str): The name of the output file.
            use_cropper (bool): Whether to use the cropping utility. Defaults to False.
            temporal_downsample (int): The amount to downsample the movie by. Units are frames. Defaults to 4.
            apply_bandpass (bool): Whether to apply a bandpass filter. Defaults to True.
            apply_high_pass (bool): Whether to apply a high-pass filter. Defaults to True.
            apply_CLAHE (bool): Whether to apply a CLAHE filter. Defaults to True.
            fix_defective_pixels (bool): Whether to fix defective pixels. Defaults to True.
            low_cutoff_freq (float): The low cutoff frequency for the bandpass filter. Units are pixels^-1.
            high_cutoff_freq (float): The high cutoff frequency for the bandpass filter. Units are pixels^-1.
            CLAHE_clip_limit (float): The clip limit for the CLAHE filter.
            num_workers (int): The number of workers to use for parallel processing. Defaults to None, which will use half of the available cores.
        Returns:
            filtered_arr (np.ndarray): The preprocessed movie.
        """

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
        else:
            top, bottom, left, right = None, None, None, None

        bool_list = [
            ("DEFECTIVE PIXEL FIX", fix_defective_pixels),
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

        wcrop_str = ""
        if apply_crop:
            sample_frame = sample_frame[top:bottom, left:right]
            wcrop_str = "; with cropping applied"

        final_output = self.utils.image_utils.resize_to_square(
            sample_frame, round_to=50
        )
        final_shape = final_output.shape

        frame_step = temporal_downsample
        frames2use = total_frames

        frame_range = range(0, total_frames, frame_step)

        print(
            f"Movie shape (before preprocessing{wcrop_str}): {(total_frames, *sample_frame.shape)}"
        )
        print(f"Movie shape (after preprocessing): {(total_frames, *final_shape)}")

        movie2use = []
        pbar = tqdm(frame_range, desc="Extracting movie for preprocessing")
        for frame_idx in pbar:
            if movie_fp.endswith(self.file_tag["TIFF"]):
                curr_frame = movie[frame_idx]
            else:
                curr_frame = movie.get_frame_data(frame_idx)
            movie2use.append(curr_frame)

        if num_workers is None:
            # use half of the available cores
            num_workers = int(max(1, cpu_count() // 2))
            print(f"Using {num_workers} cores for preprocessing")

        frames_per_worker = len(frame_range) // num_workers
        frames_per_worker2print = frames_per_worker
        batches2print = num_workers
        if len(frame_range) % num_workers != 0:
            frames_per_worker2print += 1
            batches2print += 1

        print(f"Splitting movie into {batches2print} batches:")
        self.print_wFrm(f"Each batch will have {frames_per_worker2print} frames")

        if len(frame_range) % num_workers != 0:
            self.print_wFrm(
                f"Except for last batch, which contains the remaining frames: {len(frame_range) % num_workers}"
            )
        # split movie2use into batches
        movies2useByBatch = [
            (movie2use[bidx : bidx + frames_per_worker], n_idx)
            for n_idx, bidx in enumerate(range(0, len(movie2use), frames_per_worker))
        ]
        self.print_done_small_proc()

        print("Preprocessing steps:")
        for step in bool_list:
            self.print_wFrm(f"Applying {step[0]}: {step[1]}")
        print()

        worker_params = {
            "apply_crop": apply_crop,
            "crop_coords": (top, bottom, left, right),
            "fix_defective_pixels": fix_defective_pixels,
            "apply_bandpass": apply_bandpass,
            "low_cutoff_freq": low_cutoff_freq,
            "high_cutoff_freq": high_cutoff_freq,
            "window_size": window_size,
            "apply_high_pass": apply_high_pass,
            "gSig_filt": gSig_filt,
            "apply_CLAHE": apply_CLAHE,
            "CLAHE_clip_limit": CLAHE_clip_limit,
        }

        with self.utils.ProcessStatusPrinter.output_btw_dots(
            pre_msg="See any warning messages/errors between dotted lines:",
            pre_msg_append=True,
            style="dotted",
            mini=True,
            done_msg=True,
        ):
            with Pool(num_workers) as pool:
                process_batch = partial(
                    self._preprocess_frame_batch, params=worker_params
                )
                results = list(
                    tqdm(
                        pool.imap(process_batch, movies2useByBatch),
                        total=len(movies2useByBatch),
                        desc="Processing frames in parallel",
                        position=0,
                    )
                )

        # combine results into a single array
        print("Combining results into a single array...")
        filtered_arr = np.empty((frames2use, *final_shape), dtype=np.uint16)
        for batch_idx, batch_frames in enumerate(results):
            start_idx = batch_idx * frames_per_worker
            end_idx = start_idx + len(batch_frames)
            filtered_arr[start_idx:end_idx] = batch_frames
        self.print_done_small_proc()

        # find global max & min
        global_min = float("inf")
        global_max = float("-inf")
        global_mean = float("inf")
        pbar = tqdm(filtered_arr, desc="Finding global max & min")
        for frame in pbar:
            frame_min = frame.min()
            frame_max = frame.max()
            frame_mean = frame.mean()
            global_min = min(global_min, frame_min)
            global_max = max(global_max, frame_max)
            global_mean = min(global_mean, frame_mean)
            pbar.set_postfix(
                {"Max": global_max, "Min": global_min, "Mean": global_mean}
            )

        print("Results:")
        self.print_wFrm(
            f"Global max: {global_max} / {global_max / np.iinfo(np.uint16).max:.4f} of uint16 range"
        )
        self.print_wFrm(
            f"Global min: {global_min} / {global_min / np.iinfo(np.uint16).max:.4f} of uint16 range"
        )

        # mean subtraction & global min subtraction
        pbar = tqdm(range(len(filtered_arr)), desc="Subtracting mean & global min")
        for i in pbar:
            # filtered_arr[i, :, :] = filtered_arr[i, :, :] - filtered_arr[i, :, :].mean()
            filtered_arr[i, :, :] = filtered_arr[i, :, :] - global_min
            max_val = filtered_arr[i, :, :].max()
            min_val = filtered_arr[i, :, :].min()
            mean_val = filtered_arr[i, :, :].mean()
            pbar.set_postfix({"Max": max_val, "Min": min_val, "Mean": mean_val})

        return (
            filtered_arr,
            apply_bandpass,
            apply_high_pass,
            apply_CLAHE,
            apply_crop,
        )

    @staticmethod
    def _preprocess_frame_batch(movieBatch_wID, params):
        """
        Preprocesses a batch of frames.
        """
        movieBatch = movieBatch_wID[0]
        batchID = movieBatch_wID[1]
        filtered_frames = []
        pbar = tqdm(
            movieBatch,
            desc=f"Processing batch {batchID:02d}",
            position=batchID + 1,
            leave=False,
            total=len(movieBatch),
            unit="frames",
        )
        for frame in pbar:
            frame2use = frame.copy()

            if params["apply_crop"]:
                top, bottom, left, right = params["crop_coords"]
                frame2use = frame2use[top:bottom, left:right]

            if params["fix_defective_pixels"]:
                frame2use = MoCoPreprocessing._fix_defective_pixels(frame2use)

            # apply bandpass filter (inscopix based)
            if params["apply_bandpass"]:
                frame2use = MoCoPreprocessing._apply_bandpass_filter(
                    frame2use, params["low_cutoff_freq"], params["high_cutoff_freq"]
                )

            # apply median blur filter
            frame2use = MoCoPreprocessing._apply_median_blur_filter(
                frame2use, params["window_size"]
            )

            # apply high-pass filter (Caiman based)
            if params["apply_high_pass"]:
                frame2use = MoCoPreprocessing._apply_background_filter(
                    frame2use, params["gSig_filt"]
                )

            # apply CLAHE (enhanced contrast)
            if params["apply_CLAHE"]:
                frame2use = normalization_utils.apply_CLAHE(
                    frame2use, params["CLAHE_clip_limit"]
                )

            # # normalize to uint16 range
            # frame2use = (frame2use - frame2use.min()) / (
            #     frame2use.max() - frame2use.min()
            # )
            # frame2use = img_as_uint(frame2use)

            # resize to square, round pixels to nearest multiple of 50
            frame2use = image_utils.resize_to_square(frame2use, round_to=50)

            filtered_frames.append(frame2use)

        return filtered_frames

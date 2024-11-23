import isx
import numpy as np
from skimage import img_as_uint
from tqdm import tqdm
from skimage import exposure
from PIL import Image
import h5py
from CLAH_ImageAnalysis.core import BaseClass as BC


class ISX_Utils(BC):
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

    def preprocessing_movie(
        self,
        movie_fp: str,
        output_fname: str,
        window_size: int = 3,
        gSig_filt: tuple = (2, 2),
        hp_threshold: float = 0.05,
        # frame_threshold: int = 10000,
    ) -> np.ndarray:
        """
        Preprocesses a movie. Involves subtracting the minimum value to remove glow/vignette effect,applying a median blur filter to denoise, and applying a high-pass filter to remove low-frequency signal.

        Parameters:
            file_path (str): The path to the file to be read.
            window_size (int): The size of the median blur filter window.
            gSig_filt (tuple): The size of the high-pass filter kernel.
            hp_threshold (float): The threshold for the global maximum to determine if a high-pass filter is needed.

        Returns:
            filtered_arr (np.ndarray): The preprocessed movie.
        """

        def determine_HP_threshold(global_max: float) -> bool:
            return float(global_max / np.iinfo(np.uint16).max) > hp_threshold

        # def determine_frame_threshold(frame_shape: tuple) -> bool:
        #     return frame_shape[0] > frame_threshold

        movie = self.read_file(movie_fp)

        if movie_fp.endswith(self.file_tag["TIFF"]):
            total_frames = movie.shape[0]
            sample_frame = movie[0, :, :]
        else:
            total_frames = movie.timing.num_samples
            sample_frame = movie.get_frame_data(0)
        output_fname += "_PREPROCESSED" + self.file_tag["SQZ"] + self.file_tag["H5"]

        # resize to square, round pixels to nearest multiple of 50
        # also downsamples given the size of the image
        final_output = self.utils.image_utils.resize_to_square(
            sample_frame, round_to=50
        )
        final_shape = final_output.shape

        frame_step = 1
        frames2use = total_frames
        # if total_frames > frame_threshold:
        #     frames2use = int(np.ceil(total_frames / 2))
        #     frame_step = 2
        #     print(f"Downsampling movie to {frames2use} frames")

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

        # determine if high-pass filter is needed
        # if global max is greater than 30% of the max value of the uint16 range, then apply high-pass filter
        # needs_high_pass = determine_HP_threshold(global_max)
        needs_high_pass = True

        print(
            f"Global max: {global_max} / {global_max / np.iinfo(np.uint16).max:.4f} of uint16 range"
        )
        self.print_wFrm(f"Applying high-pass filter: {needs_high_pass}")

        for frame_idx in tqdm(frame_range, desc="Preprocessing movie"):
            if movie_fp.endswith(self.file_tag["TIFF"]):
                frame = movie[frame_idx]
            else:
                frame = movie.get_frame_data(frame_idx).astype(float)

            frame2use = (frame - frame.min()) / (global_max - frame.min())
            frame2use = img_as_uint(frame2use)

            # denoise frame
            frame2use = self.dep.filter_utils.apply_median_blur_filter(
                array_stack=frame2use, window_size=window_size
            )

            if needs_high_pass:
                # use high-pass filter to remove low-frequency signal
                # use kernel size of 2
                # via caiman funcs
                frame2use = self.utils.caiman_utils.apply_high_pass_filter_space(
                    frame2use, gSig_filt=gSig_filt
                )

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

        return filtered_arr, needs_high_pass

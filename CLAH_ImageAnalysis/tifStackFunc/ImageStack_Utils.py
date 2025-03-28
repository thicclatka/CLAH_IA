import numpy as np
import tifffile as tiff
from tqdm import tqdm
import cv2
import multiprocessing
from CLAH_ImageAnalysis.core import BaseClass as BC


class ImageStack_Utils(BC):
    """
    A utility class for image stack processing and analysis.

    Parameters:
        basename (str): The base name for saving the processed images.
        GPU (str): The GPU type to use for processing. Defaults to "CUPY".
        n_proc4convparr (int): The number of processes to use for parallelized convolution. Defaults to 4.

    Attributes:
        basename (str): The base name for saving the processed images.
        n_proc4convparr (int): The number of processes to use for parallelized convolution.
        GPU (str): The GPU type to use for processing.
        trimmed_array (ndarray): The trimmed image stack.
        trimYX (ndarray): The coordinates of the trimmed region.
        tempfiltered_array (ndarray): The temporally filtered image stack.
        tempfilteredDS_array (ndarray): The downsampled temporally filtered image stack.
        norm_uint_tempfilteredDS_arr (ndarray): The normalized and corrected image stack.

    Methods:
        trim2pStack(array_to_trim, store=False): Trims the image stack to remove black borders.
        avg_CaCh_tifwriter(array_to_use, fname_save=[], Temp_Exp=False, Downsample=False): Computes the mean activity across the image stack and saves it as a TIFF file.
        caTempFilter(array_to_use, tau, store=False, parallelize=False): Applies a temporal filter to the calcium data using an exponential decay filter.
        downsampleStack(array_to_ds, DS_factor, store=False): Downsamples the image stack by a given factor.
        normalizeNcorrect_ImageStack(array_to_use, store=False): Normalizes and corrects the image stack.

    """

    def __init__(
        self,
        basename: list,
        onePhotonCheck: bool = False,
        # n_proc4convparr: int,
        # GPU=None,
    ) -> None:
        self.program_name = "ImageStack"
        self.class_type = "utils"
        BC.__init__(self, self.program_name, self.class_type)

        self.HAS_CUDA = self.dep.CUDA_utils.has_cuda()

        self.basename = basename
        self.GPUtxt = self.text_lib["GPU"]
        self.onePhotonCheck = onePhotonCheck
        # self.n_proc4convparr = n_proc4convparr

        # self.GPU = GPU if GPU is not None else self.GPUtxt["CUPY"]

    def trim2pStack(
        self, array_to_trim: np.ndarray, store: bool = False
    ) -> tuple | None:
        """
        Trims the given image stack based on the differences between consecutive frames.

        Parameters:
            array_to_trim (ndarray): The image stack to be trimmed.
            store (bool, optional): If True, the trimmed array and trim coordinates will be stored as attributes of the object.
                                    If False, the trimmed array and trim coordinates will be returned.
                                    Defaults to False.

        Returns:
            If store is False:
                tuple: A tuple containing the trimmed array and trim coordinates.
            If store is True:
                None

        Raises:
            None
        """

        def _check_isnotscalar(
            var: np.ndarray | list | int | float,
        ) -> np.ndarray | list | int | float:
            """
            Check if the input variable is not a scalar.
            """
            if isinstance(var, np.ndarray):
                return var.item() if var.size == 1 else var[0]
            elif not np.isscalar(var):
                return var[0]
            return var

        def _listNarray_check(
            var: np.ndarray | list,
        ) -> bool:
            """
            Check if the input variable is a non-empty numpy array or list.

            Parameters:
                var (np.ndarray | list): The input variable to check.

            Returns:
                bool: True if the input variable is a non-empty numpy array or list, False otherwise.
            """
            arrayCheck = isinstance(var, np.ndarray) and var.size > 0
            listCheck = isinstance(var, list) and len(var) > 0
            return arrayCheck or listCheck

        def _process_boundary(dim: np.ndarray, half_point: int) -> tuple[int, int]:
            """
            Process the boundary of the given dimension.

            Parameters:
                dim (np.ndarray): The dimension to process.
                half_point (int): The half point to use for processing.

            Returns:
                tuple[int, int]: A tuple containing the first and last indices of the boundary.
            """
            first = np.argwhere(dim[:half_point] == 0)
            last = np.argwhere(dim[half_point:] == 0)

            if _listNarray_check(first):
                first = first[-1]
            else:
                first = 0

            if _listNarray_check(last):
                last = last[0] + half_point - 2
            else:
                last = len(dim) - 1

            return first, last

        print("Trimming image stack")
        moco_arr = array_to_trim

        if len(moco_arr.shape) > 3:
            moco_arr = np.squeeze(moco_arr)

        [T, dy, dx] = moco_arr.shape

        minX = 0
        minY = 0

        maxX = dx
        maxY = dy

        for idx in tqdm(range(T - 1), desc="Trimming"):
            # moco_arr.shape = [frames, y, x];
            diff = np.abs(
                moco_arr[idx, :, :] - moco_arr[idx + 1, :, :]
            )  # diff.shape = [y, x]
            diff[np.isnan(diff)] = 0

            dimHorz = np.mean(diff, 0)  # for x of [y, x]
            dimVert = np.mean(diff, 1)  # for y of [y, x]

            horz_first, horz_last = _process_boundary(dimHorz, round(dx / 2))
            vert_first, vert_last = _process_boundary(dimVert, round(dy / 2))

            minX = max(minX, horz_first)
            maxX = min(maxX, horz_last)
            minY = max(minY, vert_first)
            maxY = min(maxY, vert_last)

        coords = [minX, maxX, minY, maxY]
        coords = [_check_isnotscalar(coord) for coord in coords]
        minX, maxX, minY, maxY = coords

        trimmed_array = moco_arr[:, minY:maxY, minX:maxX]
        trimYX = np.array([[minY, maxY], [minX, maxX]])
        self.print_done_small_proc(new_line=False)

        if store:
            self.trimmed_array = trimmed_array
            self.trimYX = trimYX
        else:
            return trimmed_array, trimYX

    def avg_CaCh_tifwriter(
        self,
        array_to_use,
        fname_save: str | list = [],
        Temp_Exp: bool = False,
        Downsample: bool = False,
        twoChanInt: int = [],
        bit_range: int = None,
    ):
        """
        Writes the average of the given image stack to a TIFF file.

        Parameters:
            array_to_use (ndarray): The image stack to use for averaging. It should have shape [frames, Y, X].
            Temp_Exp (bool, optional): Whether to adjust the filename to account for exponential decay filter. Defaults to False.
            Downsample (bool, optional): Whether to adjust the filename to account for downsampling. Defaults to False.
            twoChanInt (int, optional): The channel integer to use for the filename. Defaults to [].
            bit_range (int, optional): What bit range to convert the normalized array to. Defaults to 16.
        """

        def _append2fname_beforeExt(fname, append_str):
            return fname.split(".")[0] + append_str + self.file_tag["IMG"]

        if len(self.basename) > 1:
            self.rprint(
                "***NOTE: Given multiple sessions per day, saving each image into respective session & concat session folders"
            )

        bit_range = 16 if bit_range is None else bit_range

        for fname in self.basename:
            if len(self.basename) > 1:
                fname_save = []
                print_front = "Processing"
                if "Concat" in fname:
                    print_front += " (Concatenated):"
                else:
                    print_front += " (Session):"
                self.print_wFrm(f"{print_front} {fname}")
            if not fname_save:
                fname_save = fname

            fname_save = fname_save + "_eMC"
            # array_to_use [frames, Y, X]
            if Temp_Exp:
                self.print_wFrm(
                    "adjusting file name to account for exponential decay filter"
                )
                fname_save = (
                    fname_save
                    + self.file_tag["AVGCA"]
                    + self.file_tag["TEMPFILT"]
                    + self.file_tag["IMG"]
                )
            else:
                fname_save = fname_save + self.file_tag["AVGCA"] + self.file_tag["IMG"]

            if Downsample:
                self.print_wFrm("adjusting file name to account for downsampling")
                fname_save = _append2fname_beforeExt(
                    fname_save, self.file_tag["DOWNSAMPLE"]
                )

            if twoChanInt:
                fname_save = _append2fname_beforeExt(fname_save, f"_Ch{twoChanInt}")

            if bit_range == 8:
                fname_save = _append2fname_beforeExt(fname_save, self.file_tag["8BIT"])

            array_to_avg = array_to_use

            # image correcting
            # for 2-photon data to match previous Matlab output
            array_to_avg = self.image_corrector(array_to_avg)

            self.print_wFrm("Finding mean activity across image stack")
            avg = np.mean(array_to_avg, 0)

            # Normalize w/max & convert to 16bit
            norm_avg_uint = self.norm_uint_converter(avg, bit_range=bit_range)

            self.print_wFrm("Saving resulting values as tif", end="", flush=True)
            tiff.imwrite(fname_save, norm_avg_uint)
            self.print_done_small_proc(new_line=False)

            if bit_range == 8 and self.onePhotonCheck:
                self.save_colored_tiff(
                    norm_avg_uint, fname_save, clip_limit=4.0, tile_grid_size=(6, 6)
                )

    def norm_uint_converter(
        self, array_to_norm: np.ndarray, bit_range: int | None = None
    ) -> np.ndarray:
        """
        Normalize the input array by dividing it by its maximum value and convert it to either 8 or 16-bit range. By default, it converts to 16 bit.

        Parameters:
            array_to_norm (ndarray): The input array to be normalized.
            bit_range (int, optional): The bit range to convert the normalized array to. Defaults to 16.

        Returns:
            ndarray: The normalized array converted to a 8 16-bit range.
        """
        if bit_range is None:
            # default to 16-bit
            bit_range = 16

        self.print_wFrm("Normalizing with max value")
        norm_array = array_to_norm / np.max(array_to_norm)

        self.print_wFrm(f"Converting to {bit_range}-bit range")
        if bit_range == 8 or bit_range == 16:
            norm_array_img = self.dep.convert_bit_range(norm_array, bit_range)
        else:
            raise ValueError(f"Bit range {bit_range} not supported")

        return norm_array_img

    def image_corrector(self, image_stack_to_correct: np.ndarray) -> np.ndarray:
        """
        Corrects the image stack to match previous Matlab output.

        Parameters:
            image_stack_to_correct (ndarray): The image stack to be corrected.

        Returns:
            ndarray: The corrected image stack.
        """
        self.print_wFrm("Correcting image to match previous Matlab output")
        corrected_image_stack = image_stack_to_correct
        self.print_wFrm("Mirroring across Y axis", frame_num=1)
        corrected_image_stack = np.flip(corrected_image_stack, axis=2)
        self.print_wFrm("Rotating 90 degrees to left (counterclockwise)", frame_num=1)
        corrected_image_stack = np.rot90(corrected_image_stack, k=1, axes=(1, 2))

        return corrected_image_stack

    def caTempFilter(
        self,
        array_to_use: np.ndarray,
        tau: int = 3,
        store: bool = False,
        num_threads: int | None = None,
        temp_filter_start: int = 1,
        temp_filter_tail: int = 6,
    ):
        """
        Apply temporal filtering to calcium data using an exponential decay filter.

        Parameters:
            array_to_use (ndarray): The input array of calcium data.
            tau (float, optional): The time constant for the exponential decay filter. Defaults to 3.
            store (bool, optional): Whether to store the filtered array in the object. Defaults to False.
            num_threads (int, optional): The number of threads to use for parallelization. Defaults to None, which will use all available threads.
            temp_filter_start (int, optional): The starting frame for the exponential decay filter. Defaults to 1.
            temp_filter_tail (int, optional): The tail factor for the exponential decay filter. Defaults to 6.
        Returns:
            ndarray: The filtered array of calcium data.
        """
        # set up timekeeper
        TKEEPER = self.time_utils.TimeKeeper()
        print(
            f"Temporal filter of calcium data w/ exponential decay filter (tau = {tau})"
        )
        # Generate exponential decay filter
        yexp = self.dep.create_exponential_decay_filter(
            tau, start=temp_filter_start, tail_factor=temp_filter_tail
        )

        # Get shape of input array
        frames, dy, dx = array_to_use.shape

        # Reshape array to 2D (t, y*x)
        reshaped_array = array_to_use.reshape((frames, dy * dx))

        # setting threads
        # if None, will use all available threads
        num_threads = cv2.getNumberOfCPUs() if num_threads is None else num_threads

        cv2.setNumThreads(num_threads)

        self.print_wFrm(
            "Using {} of {} available threads".format(
                cv2.getNumThreads(), multiprocessing.cpu_count()
            )
        )

        convolved_data = self.dep.apply_filter2D(
            array2filter=reshaped_array,
            kernel=yexp[:, None],
            borderType=cv2.BORDER_CONSTANT,
        )

        array_filtered = convolved_data[:frames, :]

        array_to_return = array_filtered.reshape((frames, dy, dx))

        # Replace first 50 frames with 51st frame
        # control for any edge effects
        array_to_return[:50, :, :] = array_to_return[50, :, :]
        TKEEPER.setEndNprintDuration()
        if store:
            self.tempfiltered_array = array_to_return
        else:
            return array_to_return

    def downsampleStack(
        self, array_to_ds: np.ndarray, DS_factor: float = None, store: bool = False
    ) -> np.ndarray:
        """
        Downsamples the input array by a given factor.

        Parameters:
            array_to_ds (ndarray): The input array to be downsampled.
            DS_factor (float): The downsampling factor. Defaults to None, which will use the determine_DS_factor function.
            store (bool, optional): If True, the downsampled array will be stored in `self.tempfilteredDS_array`.
                                    Defaults to False.

        Returns:
            ndarray: The downsampled array.
        """

        frames = array_to_ds.shape[0]
        dims = list(array_to_ds.shape[1:])

        DS_factor = (
            self.image_utils.determine_DS_factor(dims)
            if DS_factor is None
            else DS_factor
        )

        if DS_factor > 1:
            print(f"Downsampling by a factor of {DS_factor}")

            # do 1/ds factor to get proper value
            DS_factor = 1 / DS_factor

            spatDs = tuple(int(x * DS_factor) for x in dims)

            array_post_ds = np.empty((frames, *spatDs))

            for frNum in tqdm(range(frames), desc="Downsampling"):
                array_post_ds[frNum, :, :] = self.image_utils.resize_to_specific_dims(
                    array_to_ds[frNum, :, :], spatDs
                )
        else:
            print(
                "Dimensions per frame are already small enough. Skipping downsampling..."
            )
            array_post_ds = array_to_ds

        if store:
            self.tempfilteredDS_array = array_post_ds
        else:
            return array_post_ds

    def min_zProj_removal(self, array_to_use: np.ndarray) -> np.ndarray:
        print("Finding min z-projection & subtracting from each frame")
        # z_min = np.zeros_like(array_to_use[0, :, :])

        # for frame_idx in tqdm(
        #     range(array_to_use.shape[0]), desc="Finding min z-projection"
        # ):
        #     z_min = np.minimum(z_min, array_to_use[frame_idx, :, :])

        z_min = np.min(array_to_use, axis=0)

        self.print_wFrm("Subtracting min z-projection from each frame")
        array_post_minZ = array_to_use - z_min

        self.print_done_small_proc()

        return array_post_minZ

    def normalizeNcorrect_ImageStack(
        self, array_to_use: np.ndarray, store: bool = False
    ) -> np.ndarray:
        """
        Normalize and correct an image stack.

        Parameters:
            array_to_use (ndarray): The input image stack to be normalized and corrected.
            store (bool, optional): Whether to store the normalized and corrected image stack. Defaults to False.

        Returns:
            ndarray: The normalized and corrected image stack, if `store` is False.
        """

        print("Normalizing image stack & converting to 16-bit range")
        norm_uint_tempfilteredDS_arr = self.norm_uint_converter(array_to_use)
        self.print_done_small_proc()

        # to match Matlab output
        print("Correcting image stack")
        norm_uint_tempfilteredDS_arr = self.image_corrector(
            norm_uint_tempfilteredDS_arr
        )
        self.print_done_small_proc()

        if store:
            self.norm_uint_tempfilteredDS_arr = norm_uint_tempfilteredDS_arr
        else:
            return norm_uint_tempfilteredDS_arr

    def apply_colormap2stack_export2avi(
        self,
        array_to_use: np.ndarray,
        fname_save: str,
        colormap: int = cv2.COLORMAP_PLASMA,
        fps: float = 20,
        clip_limit: float = 2.0,
        tile_grid_size: tuple = (8, 8),
    ) -> str:
        """
        Apply a colormap to an image stack and export it as an AVI video file.

        This method takes a 3D numpy array representing an image stack, applies a colormap
        to each frame, and saves the result as an AVI video file. The array is first
        normalized to 8-bit range (0-255) before applying the colormap.

        Parameters:
            array_to_use (np.ndarray): 3D array of shape (frames, height, width) containing
                the image stack to be processed.
            fname_save (str): Path where the output AVI file will be saved. If it doesn't
                end with .avi extension, it will be added automatically.
            colormap (int, optional): OpenCV colormap to apply to each frame. Defaults to
                cv2.COLORMAP_PLASMA.
            fps (float, optional): Frames per second for the output video. Defaults to 20.

        Returns:
            str: The full path to the saved AVI file.
        """
        AVI = self.file_tag["AVI"]

        # remove extension if present
        fname_save = fname_save.split(".")[0]

        if not fname_save.endswith(AVI):
            fname_save = fname_save + AVI

        array_8bit = self.dep.normalize_array(array_to_use, dtype=np.uint8)

        _, height, width = array_8bit.shape

        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(fname_save, fourcc, fps, (width, height))

        for frame in tqdm(array_8bit, desc="Creating colormapped AVI"):
            frame_adjusted = self.dep.apply_CLAHE(frame, clip_limit, tile_grid_size)
            frame_colored = self.dep.apply_colormap(frame_adjusted, colormap)
            out.write(frame_colored)

        self.print_wFrm("Exporting resulting AVI file")
        out.release()

        return fname_save

    def save_colored_tiff(
        self,
        array_to_save: np.ndarray,
        fname_save: str,
        colormap: int = cv2.COLORMAP_PLASMA,
        clip_limit: float = 4.0,
        tile_grid_size: tuple = (8, 8),
    ) -> str:
        """
        Save a colored TIFF file.
        """
        array_adjusted = self.dep.apply_CLAHE(array_to_save, clip_limit, tile_grid_size)
        colored_array = self.dep.apply_colormap(array_adjusted, colormap)

        fname_save = (
            fname_save.split(".")[0] + self.file_tag["CMAP"] + self.file_tag["IMG"]
        )

        cv2.imwrite(fname_save, colored_array)

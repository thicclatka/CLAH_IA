#! /usr/bin/env python
"""
This pipeline script, MoCo2segDict.py, is tailored for processing calcium imaging data.
It incorporates essential steps such as motion correction, trimming, filtering, and segmentation.
The script facilitates the handling of data from H5 files, applying advanced image processing
techniques to prepare the data for in-depth analysis.

Primary functionalities include:
-   Motion Correction: Stabilizes the imaging data by correcting movements across the sequence of frames,
    which is vital for the accuracy of further analyses.
-   Trimming and Filtering: Improves data quality by removing noise and artifacts and focusing on
    specific regions of interest.
-   Segmentation: Distinguishes individual neurons within the imaging data, enabling the extraction
    of neural activity signals from spatially distinct sources.

Customizable through command-line arguments, the script can accommodate various input requirements.
It relies heavily on the CaImAn package for key operations and efficiently handles large datasets
using parallel processing techniques.

Usage:
Execute the script from the command line with options for the data path, motion correction,
segmentation, and parallel convolution processing.

Parser arguments:
    --path, -p: Path to the data directory
    --sess2process, -s2p: List of sessions to process
    --motion_correct, -mc: Flag for motion correction (default: False)
    --segment, -sg: Flag for segmentation (default: False)
    --n_proc4MOCO, -n4mc: Number of processors for motion correction (default: 1)
    --n_proc4CNMF, -n4cnmf: Number of processors for CNMF (default: 1)
    --concatenate, -c: Flag for concatenating sessions (default: False)
    --overwrite, -o: Flag for overwriting existing files (default: False)
    --mc_iter, -mci: Number of iterations for motion correction (default: 1)
    --compute_metrics, -cm: Flag for computing metrics for motion correction (default: False)
    --use_cropper, -uc: Flag for using the cropping utility (default: False)

Example:
python MoCo2segDict.py --path /path/to/data --motion_correct yes --segment yes
"""

from CLAH_ImageAnalysis.core import run_CLAH_script
from CLAH_ImageAnalysis.core import extract_args_preParser
from CLAH_ImageAnalysis.tifStackFunc import M2SD_manager as M2SDM


class MoCo2segDict(M2SDM):
    def __init__(self, **kwargs) -> None:
        """
        Initializes an instance of the ClassName class.

        Parameters:
            path (list): A list of paths.
            sess2process (list): A list of sessions to process.
            n_proc4MOCO (int): Number of processors for motion correction. Default is 1.
            n_proc4CNMF (int): Number of processors for CNMF. Default is 1.
            motion_correct (bool): Flag indicating whether to perform motion correction. Default is True.
            segment (bool): Flag indicating whether to perform segmentation. Default is True.
            overwrite (bool): Flag indicating whether to overwrite existing files. Default is False.
            mc_iter (int): Number of iterations for motion correction. Default is 1.
            compute_metrics (bool): Flag indicating whether to compute metrics for motion correction. Default is False.
            use_cropper (bool): Flag indicating whether to use the cropping utility. Default is False.
            separate_channels (bool): Flag indicating whether to motion correct channels separately. Only applicable for 2photon data with 2 channels. Default is False.
        """
        self.program_name = "M2SD"
        path = kwargs.get("path", [])
        sess2process = kwargs.get("sess2process", [])
        n_proc4MOCO = kwargs.get("n_proc4MOCO", 26)
        n_proc4CNMF = kwargs.get("n_proc4CNMF", 1)
        concatCheck = kwargs.get("concatenate", False)
        motion_correct = kwargs.get("motion_correct", True)
        segment = kwargs.get("segment", True)
        overwrite = kwargs.get("overwrite", False)
        mc_iter = kwargs.get("mc_iter", 1)
        compute_metrics = kwargs.get("compute_metrics", False)
        use_cropper = kwargs.get("use_cropper", False)
        separate_channels = kwargs.get("separate_channels", False)
        export_postseg_residuals = kwargs.get("export_postseg_residuals", False)

        M2SDM.__init__(
            self,
            program_name=self.program_name,
            path=path,
            sess2process=sess2process,
            motion_correct=motion_correct,
            segment=segment,
            n_proc4MOCO=n_proc4MOCO,
            n_proc4CNMF=n_proc4CNMF,
            concatCheck=concatCheck,
            mc_iter=mc_iter,
            overwrite=overwrite,
            compute_metrics=compute_metrics,
            use_cropper=use_cropper,
            separate_channels=separate_channels,
            export_postseg_residuals=export_postseg_residuals,
        )

        self.step_headers = self.text_lib["steps"][self.program_name]["header"]

    def run_whole_proc(self) -> None:
        """
        Runs the whole processing procedure.

        This method overwrites the parent class method. It prints all the steps for the process and then calls the
        `sessBysess_processing` method to perform the processing steps in the specified order.
        """

        # print all steps for process
        self.print_header("Starting loop for steps 1-4")
        # Main for loop for steps 1-4
        self.sessBysess_processing(self.process_order)

    def process_order(self) -> None:
        """
        Executes the processing steps in the specified order.

        If motion correction is enabled, it performs motion correction using the M2SD_MOCO method,
        followed by post-motion correction processing using the M2SD_POST_MOCO_PROC method.
        If motion correction is disabled, it skips steps 2-4 and prints the corresponding header.

        If segmentation is enabled, it performs segmentation using the M2SD_SEGMENT method.
        If segmentation is disabled, it skips step 5 and prints the corresponding header.
        """

        if self.motion_correct:
            self.M2SD_MOCO()
            self.M2SD_POST_MOCO_PROC()
        else:
            # printing step 2-4 is being skipped
            self.print_header(self.step_headers["s2-3_skip"])
        if self.segment:
            self.M2SD_SEGMENT()
        else:
            # printing step 5 is being skipped
            self.print_header(self.step_headers["s4_skip"])

    def M2SD_MOCO(self) -> None:
        """
        Performs motion correction.

        This function loads the h5 file, performs motion correction,
        and saves the motion corrected image stack as a memmap file.
        """

        # Step 1 Loading h5
        self.print_header(self.step_headers["s1"])
        # H5 prep: loading & squeezing
        # if moco mmap is found, skips this step
        self.pre_moco_h5_tools()

        # Step 2 Motion Correction
        self.print_header(self.step_headers["s2"])

        # Motion Correction setup, correction, & saving memmap file
        # see tifstackfunc_const.py for the default Motion Correction values:
        # max_shifts, niter_rig, splits_rig, strides, overlaps, splits_els
        # upsample_factor_grid, max_deviation_rigid, shifts_opencv, nonneg_movie
        # starts cluster for motion correction if no mmap for moco is found
        # otherwise stores that mmap name to be loaded
        self.motion_correction()

        # self.view_movie(fname=self.moco.fname_tot_els)

    def M2SD_POST_MOCO_PROC(self) -> None:
        """
        Performs trimming and filtering after motion correction.

        This function loads the motion corrected image stack from the memmap file,
        performs trimming, filtering, and downsampling, and saves the processed
        image stack as a memmap file.
        """

        # Step 3 Trimming & filtering
        self.print_header(self.step_headers["s3"])

        # load from mmap the motion corrected array
        # reshapes loaded array from memmap into [frames,y,x]
        # also clear moco for memory
        with self.StatusPrinter.output_btw_dots(
            pre_msg="Reloading memory map of motion corrected image stack",
            post_msg="Memmap is loaded\n",
        ):
            self.load_mmap_ImageStack(store4Trim=True)

        # run Image Stack Utils to trim, tempfilter, downsample
        # tif is saved of average Ca activity after each step
        # before segmentation occurs, image stack is also normalized & corrected
        # parallelize is for GPU convolutions
        self.run_ISUtils4Segmentation()

        # save processed array as h5 to create filename for memory mapping
        # use basename (contained h5filename but without everything after _Cycle)
        # full name ending = _[DATE, if duplicate]_sqz_eMC_caChExpDS.h5
        # - array saved is the normalized & corrected tempfilteredDS_array
        self.write_procImageStack2H5()

        # saving memmap of post-processed image stack
        self.rprint("Saving memmap of post-processed image stack:")
        # memory map file in order 'C'
        self.save_memmap_postproc()

        # clear ISUtils for memory
        self.clear_ISUtils()
        self.print_done_small_proc()

    def M2SD_SEGMENT(self) -> None:
        """
        Performs segmentation using CNMF.

        This function loads the processed image stack from the memmap file,
        performs segmentation using CNMF, and saves the segmentation results
        as a segDict file.
        """

        # Step 4 segDict creation via CNMF
        self.print_header(self.step_headers["s4"])

        # restart cluster to clean up memory
        print("Starting cluster for segmentation")
        # use default n_process to use all available CPU Cores
        self.restart_cluster(n_processes=self.n_proc4CNMF)
        self.print_done_small_proc()

        print("Reloading memmap of processed image stack")
        self.utils.section_breaker("dotted", mini=True)
        # load mmap file of processed image stack
        # reshape into [frames, y, x]
        if self.motion_correct:
            mmaps2load = self.fname_mmap_postproc
        else:
            # w/no moco, need to find mmap file
            mmaps2load = self.find_mmap_fname()
            if not mmaps2load:
                # if no mmap is found assumes motion correction has not been done
                # need to stop cluster before running motion correction again
                self.stop_cluster()

                self.rprint("No post-processed memmap file found")
                self.rprint("Stopping cluster before running motion correction")
                self.rprint("Running Motion Correction & post-processing again...")
                self.utils.section_breaker("dotted", mini=True)
                print()
                self.M2SD_MOCO()
                self.M2SD_POST_MOCO_PROC()
                # load mmap file of processed image stack
                mmaps2load = self.fname_mmap_postproc

        self.load_mmap_ImageStack(mmap_fname_to_load=mmaps2load, store4CNMF=True)
        self.utils.section_breaker("dotted", mini=True)
        self.rprint("Memmap is loaded\n")

        # parameters for source extraction & deconvolution defined largely by
        # what is in CNMF_const file

        # Initialize CNMF class funcs
        # can also change these vars here:
        # - n_processes, k, merge_thresh, p, gnb, gSig
        # - frames, decay_time, min_SNR, r_values_min, use_cnn, thresh_cnn_min
        # - threshold & vmax for comp evaluation
        # - method_deconvolution, check_nan
        # otherwise uses default values from CNMF_enum
        # initialized class finds patches, evaluates components, plots components, & refines patches
        # saves results as segDict
        # - structure of segDict:
        #   - A_Spatial, C_Temporal, DF/F0, dx, dy, S_Deconvolution
        self.run_CNMF()

        # save Residuals as .mp4
        # concatenate movie of moco - reconstruced (w/wo background)
        # and movie of moco - reconstructed (w/background)
        # background is b dot f
        self.find_residual_postSeg()

        # STOP CLUSTER
        self.stop_cluster(final=True)


def main():
    import os
    import getpass
    from pathlib import Path
    from CLAH_ImageAnalysis.tifStackFunc import TSF_enum
    from sqljobscheduler import LockFileUtils

    def run_script(clear_terminal: bool = True):
        run_CLAH_script(
            MoCo2segDict,
            parser_enum=TSF_enum.Parser4M2SD,
            clear_terminal=clear_terminal,
        )

    help_flag = extract_args_preParser(
        parser_enum=TSF_enum.Parser4M2SD, flag2find="--help"
    ) or extract_args_preParser(parser_enum=TSF_enum.Parser4M2SD, flag2find="-h")

    if help_flag:
        run_script(clear_terminal=False)
        exit()

    sql_status = extract_args_preParser(
        parser_enum=TSF_enum.Parser4M2SD, flag2find="--from_sql"
    )

    if not sql_status:
        LockFileUtils.gpu_lock_check_timer(duration=60)

    if not LockFileUtils.check_gpu_lock_file():
        print("Creating GPU lock file for this run")
        LockFileUtils.create_gpu_lock_file(
            user=getpass.getuser(),
            script=Path(__file__).name,
            pid=int(os.getpid()),
            ctype="cli",
        )

    try:
        # run parser, create instance of class, and run the script
        run_script()

    finally:
        # remove GPU lock file
        LockFileUtils.remove_gpu_lock_file()


######################################################
#  run script if called from command line
######################################################
if __name__ == "__main__":
    main()

# from caiman import load as cm_load
import os
from caiman.motion_correction import MotionCorrect
from enum import Enum
# from caiman.motion_correction import high_pass_filter_space

# from caiman.motion_correction import motion_correct_oneP_nonrigid
from CLAH_ImageAnalysis import utils
from CLAH_ImageAnalysis.tifStackFunc import TSF_enum


def _brk():
    breaker = utils.text_dict()["breaker"]["hash"]
    print(breaker)


def NoRMCorre(
    h5filename: list[str],
    dview: object,
    max_shifts: int | None = None,
    niter_rig: int | None = None,
    splits_rig: int | None = None,
    strides: int | None = None,
    overlaps: int | None = None,
    splits_els: int | None = None,
    upsample_factor_grid: int | None = None,
    max_deviation_rigid: int | None = None,
    shifts_opencv: bool | None = None,
    nonneg_movie: bool | None = None,
    gSig_filt: int | None = None,
    pw_rigid: bool | None = None,
    use_cuda: bool | None = None,
    border_nan: bool | None = None,
    onePhotonCheck: bool = False,
    mc_iter: int = 1,
) -> list[MotionCorrect]:
    """
    Perform motion correction on a given movie file using NoRMCorre algorithm.

    Parameters:
        h5filename (str): Path to the movie file in HDF5 format.
        dview (object): A parallel computing view object.
        max_shifts (int, optional): Maximum allowed rigid shift. Defaults to MOCOpar["MAX_SHIFTS"].
        niter_rig (int, optional): Number of iterations for rigid motion correction. Defaults to MOCOpar["NITER_RIG"].
        splits_rig (int, optional): Number of splits for parallelization of rigid motion correction. Defaults to MOCOpar["SPLITS_RIG"].
        strides (int, optional): Start new path for pw-rigid motion correction every x pixels. Defaults to MOCOpar["STRIDES"].
        overlaps (int, optional): Overlap between paths for pw-rigid motion correction. Defaults to MOCOpar["OVERLAPS"].
        splits_els (int, optional): Number of splits for parallelization of elastic motion correction. Defaults to MOCOpar["SPLITS_ELS"].
        upsample_factor_grid (int, optional): Upsample factor to avoid smearing when merging patches. Defaults to MOCOpar["UPSAMPLE_FACTOR"].
        max_deviation_rigid (float, optional): Maximum deviation allowed for patch with respect to rigid shifts. Defaults to MOCOpar["MAX_DEV_RIG"].
        shifts_opencv (bool, optional): Use OpenCV for computing shifts. Defaults to MOCOpar["SHIFTS_OPENCV"].
        nonneg_movie (bool, optional): Ensure non-negative movie. Defaults to MOCOpar["NONNEG"].
        pw_rigid (bool, optional): Perform piece-wise rigid motion correction. Defaults to MOCOpar["PW_RIGID"].
        use_cuda (bool, optional): Use CUDA for motion correction. Defaults to MOCOpar["USE_CUDA"].
        gSig_filt (int, optional): Global standard deviation of the filters. Defaults to MOCOpar["GSIG_FILT"].
        mc_iter (int): Number of iterations for motion correction. Defaults to 1.

    Returns:
        mc (MotionCorrect): The motion-corrected movie object.
    """

    def _rename_mmap(old_map_fname: str, new_mmap_name: str) -> str:
        os.rename(old_map_fname, new_mmap_name)
        return new_mmap_name

    MOCOpar = utils.enum_utils.enum2dict(TSF_enum.MOCO_Params)
    MOCOpar4OnePhoton = utils.enum_utils.enum2dict(TSF_enum.MOCO_Params4OnePhoton)
    # MOCOpar4OnePhoton = utils.enum_utils.enum2dict(TSF_enum.MOCO_Params)

    MOCOpar2use = MOCOpar4OnePhoton if onePhotonCheck else MOCOpar
    # MOCOparEnum2use = (
    #     TSF_enum.MOCO_Params4OnePhoton if onePhotonCheck else TSF_enum.MOCO_Params
    # )
    # MOCOparEnum2use = TSF_enum.MOCO_Params

    # get the default values from the enum class
    niter_rig = int(MOCOpar2use["NITER_RIG"]) if niter_rig is None else niter_rig
    max_shifts = MOCOpar2use["MAX_SHIFTS"] if max_shifts is None else max_shifts
    splits_rig = MOCOpar2use["SPLITS_RIG"] if splits_rig is None else splits_rig
    strides = MOCOpar2use["STRIDES"] if strides is None else strides
    overlaps = MOCOpar2use["OVERLAPS"] if overlaps is None else overlaps
    splits_els = MOCOpar2use["SPLITS_ELS"] if splits_els is None else splits_els
    upsample_factor_grid = (
        MOCOpar2use["UPSAMPLE_FACTOR"]
        if upsample_factor_grid is None
        else upsample_factor_grid
    )
    max_deviation_rigid = (
        MOCOpar2use["MAX_DEV_RIG"]
        if max_deviation_rigid is None
        else max_deviation_rigid
    )
    shifts_opencv = (
        bool(MOCOpar2use["SHIFTS_OPENCV"]) if shifts_opencv is None else shifts_opencv
    )
    nonneg_movie = bool(MOCOpar2use["NONNEG"]) if nonneg_movie is None else nonneg_movie
    border_nan = MOCOpar2use["BORDER_NAN"] if border_nan is None else border_nan
    pw_rigid = bool(MOCOpar2use["PW_RIGID"]) if pw_rigid is None else pw_rigid
    use_cuda = bool(MOCOpar2use["USE_CUDA"]) if use_cuda is None else use_cuda

    if MOCOpar2use["GSIG_FILT"] is not None:
        gSig_filt = MOCOpar2use["GSIG_FILT"]

    # Setting up moco parameters than need to be arranged into tuples
    max_shifts = (max_shifts, max_shifts)  # maximum allow rigid shift
    # start new path for pw-rigid motion correction every x pxls
    strides = (strides, strides) if isinstance(strides, int) else strides
    # overlap between paths (size of path stirdes + overlaps)
    overlaps = (overlaps, overlaps) if isinstance(overlaps, int) else overlaps

    gSig_filt = (gSig_filt, gSig_filt) if isinstance(gSig_filt, int) else gSig_filt

    # utils.print_wFrame(f"Applying motion correction to: {h5filename}")
    # utils.print_wFrame(f"loading movie for motion correction")
    # min_mov = cm.load([h5filename], subindices=range(200)).min()

    # print what the parameters are and export to file
    parFile = "MotionCorr" + "_onePhoton" if onePhotonCheck else ""

    dictParams = {
        "MC_ITER": mc_iter,
        "NITER_RIG": niter_rig,
        "MAX_SHIFTS": max_shifts,
        "SPLITS_RIG": splits_rig,
        "STRIDES": strides,
        "OVERLAPS": overlaps,
        "SPLITS_ELS": splits_els,
        "UPSAMPLE_FACTOR": upsample_factor_grid,
        "MAX_DEV_RIG": max_deviation_rigid,
        "SHIFTS_OPENCV": shifts_opencv,
        "NONNEG": nonneg_movie,
        "BORDER_NAN": border_nan,
        "PW_RIGID": pw_rigid,
        "USE_CUDA": use_cuda,
        "GSIG_FILT": gSig_filt,
    }
    enumParams = Enum("MotionCorrParams", dictParams)

    # export the parameters to a file
    # also prints parameters to console for current motion correction run
    TSF_enum.export_settings2file(enumParams, parFile)

    # utils.print_wFrame(f"Motion correcting...(this should take some time)")

    # initiate template as None
    template = None
    # create list to hold outputs
    mc_output = []

    if len(h5filename) <= 2:
        for idx, h5_2moco in enumerate(h5filename):
            # utils.print_wFrame("loading movie for motion correction")
            # min_mov = cm_load([h5_2moco], subindices=range(200)).min()
            if idx == 1:
                utils.print_wFrame(
                    "Applying template from previous motion correction",
                    frame_num=1,
                )

            h2moco2print = utils.folder_tools.os_splitterNcheck(h5_2moco, "base")
            with utils.ProcessStatusPrinter.output_btw_dots(
                pre_msg=utils.create_multiline_string(
                    [
                        f"Motion correcting: {h2moco2print}",
                        f"Applying parameters for {'1-photon (Miniscope)' if onePhotonCheck else '2-photon'} data",
                        "See any warning messages/error bewteen dotted lines:",
                    ]
                ),
                post_msg=f"Motion correction complete for: {h2moco2print}",
                timekeep=True,
                timekeep_msg="Motion Correction",
                timekeep_seconds=False,
            ):
                # perform motion correction
                mc = MotionCorrect(
                    h5_2moco,
                    dview=dview,
                    max_shifts=max_shifts,
                    niter_rig=niter_rig,
                    splits_rig=splits_rig,
                    strides=strides,
                    overlaps=overlaps,
                    splits_els=splits_els,
                    upsample_factor_grid=upsample_factor_grid,
                    max_deviation_rigid=max_deviation_rigid,
                    shifts_opencv=shifts_opencv,
                    nonneg_movie=nonneg_movie,
                    border_nan=border_nan,
                    pw_rigid=pw_rigid,
                    use_cuda=use_cuda,
                    gSig_filt=gSig_filt,
                )

                mc.motion_correct(save_movie=True, template=template)

                # perform additional motion correction iterations
                # if mc_iter > 1
                if mc_iter > 1:
                    _brk()
                    print(
                        f"!!!Performing {mc_iter - 1} more iteration(s) of motion correction!!!"
                    )
                    print()
                    for it in range(mc_iter - 1):
                        mmap2use = mc.fname_tot_els[0]

                        print(f"Iteration {it + 1} of {mc_iter - 1}")
                        utils.print_wFrame(
                            f"Motion correcting: {os.path.basename(mmap2use)}"
                        )

                        # perform motion correction
                        mc = MotionCorrect(
                            mmap2use,
                            dview=dview,
                            max_shifts=max_shifts,
                            niter_rig=niter_rig,
                            splits_rig=splits_rig,
                            strides=strides,
                            overlaps=overlaps,
                            splits_els=splits_els,
                            upsample_factor_grid=upsample_factor_grid,
                            max_deviation_rigid=max_deviation_rigid,
                            shifts_opencv=shifts_opencv,
                            nonneg_movie=nonneg_movie,
                            border_nan=border_nan,
                            pw_rigid=pw_rigid,
                            use_cuda=use_cuda,
                            gSig_filt=gSig_filt,
                        )

                        mc.motion_correct(save_movie=True, template=template)

                        # remove old mmap
                        os.remove(mmap2use)

                        # rename new mmap
                        mc.fname_tot_els[0] = _rename_mmap(
                            mc.fname_tot_els[0], mmap2use
                        )
                        _brk()
                        print()

                # append the motion corrected movie to the output list
                mc_output.append(mc)

                if idx == 0 and not onePhotonCheck:
                    # set template for next motion correction
                    # use the last element of mc.templates_els
                    template = mc.total_template_els
    else:
        raise ValueError(
            "Session with more than 2 channels was found. Please provide  a session with at most 2 channels"
        )

    return mc_output

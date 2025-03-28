# from caiman import load as cm_load
import os
import shutil
import numpy as np
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
    params: dict | None = None,
    onePhotonCheck: bool = False,
    mc_iter: int = 1,
    compute_metrics: bool = False,
    separate_channels: bool = False,
) -> list[object]:
    """
    Perform motion correction on a given movie file using NoRMCorre algorithm.

    Parameters:
        h5filename (str): Path to the movie file in HDF5 format.
        dview (object): A parallel computing view object.
        params (dict, optional): Parameters for motion correction. Defaults to None. With the following keys:
        - max_shifts (int, optional): Maximum allowed rigid shift. Defaults to MOCOpar["MAX_SHIFTS"].
        - niter_rig (int, optional): Number of iterations for rigid motion correction. Defaults to MOCOpar["NITER_RIG"].
        - splits_rig (int, optional): Number of splits for parallelization of rigid motion correction. Defaults to MOCOpar["SPLITS_RIG"].
        - strides (int, optional): Start new path for pw-rigid motion correction every x pixels. Defaults to MOCOpar["STRIDES"].
        - overlaps (int, optional): Overlap between paths for pw-rigid motion correction. Defaults to MOCOpar["OVERLAPS"].
        - splits_els (int, optional): Number of splits for parallelization of elastic motion correction. Defaults to MOCOpar["SPLITS_ELS"].
        - upsample_factor_grid (int, optional): Upsample factor to avoid smearing when merging patches. Defaults to MOCOpar["UPSAMPLE_FACTOR"].
        - max_deviation_rigid (float, optional): Maximum deviation allowed for patch with respect to rigid shifts. Defaults to MOCOpar["MAX_DEV_RIG"].
        - shifts_opencv (bool, optional): Use OpenCV for computing shifts. Defaults to MOCOpar["SHIFTS_OPENCV"].
        - nonneg_movie (bool, optional): Ensure non-negative movie. Defaults to MOCOpar["NONNEG"].
        - pw_rigid (bool, optional): Perform piece-wise rigid motion correction. Defaults to MOCOpar["PW_RIGID"].
        - use_cuda (bool, optional): Use CUDA for motion correction. Defaults to MOCOpar["USE_CUDA"].
        - gSig_filt (int, optional): Global standard deviation of the filters. Defaults to MOCOpar["GSIG_FILT"].
        mc_iter (int): Number of iterations for motion correction. Defaults to 1.
        compute_metrics (bool): Calculate motion correction metrics. Defaults to False.
        separate_channels (bool): Whether to motion correct channels separately for motion correction. Defaults to False, in which case the motion correction performed on first channel is applied to the second channel as a template.

    Returns:
        mc (MotionCorrect): The motion-corrected movie object.
    """

    def _rename_mmap(mmap2rename: str, basename: str, iteration: int) -> str:
        """
        Rename the motion-corrected movie file.

        Parameters:
            mmap2rename (str): The original motion-corrected movie file name.
            basename (str): The base name of the motion-corrected movie file.
            iteration (int): The iteration number of the motion correction.

        Returns:
            str: The new motion-corrected movie file name.
        """
        base, params = basename.split("__")
        new_mmap_name = f"{base}_iter{iteration}__{params}"
        os.rename(mmap2rename, new_mmap_name)
        return new_mmap_name

    def _reduce_tuple_params(
        params: tuple, div_modifier: int = 2, add_modifier: int = 0
    ) -> tuple:
        """
        Reduce the parameters in a tuple by a given factor and add a given value.

        Parameters:
            params (tuple): The tuple of parameters to reduce.
            div_modifier (int): The factor to divide the parameters by. Defaults to 2.
            add_modifier (int): The value to add to the parameters. Defaults to 0.

        Returns:
            tuple: The reduced tuple of parameters.
        """
        return tuple([int((par // div_modifier) + add_modifier) for par in params])

    def _export_params2file(
        mc_iter: int,
        niter_rig: int,
        max_shifts: tuple,
        splits_rig: int,
        strides: tuple,
        overlaps: tuple,
        splits_els: int,
        upsample_factor_grid: int,
        max_deviation_rigid: int,
        shifts_opencv: bool,
        nonneg_movie: bool,
        border_nan: bool,
        pw_rigid: bool,
        use_cuda: bool,
        gSig_filt: tuple,
        enum_name: str,
        parFile: str,
    ) -> None:
        """
        Export the parameters to a file.

        Parameters:
            For mc_iter to gSig_filt, see docstring for NoRMCorre.
            enum_name (str): The name of the enum class to use for the parameters.
            parFile (str): The name of the file to save the parameters to.

        Returns:
            None
        """
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

        enumParams = Enum(enum_name, dictParams)
        TSF_enum.export_settings2file(enumParams, parFile)
        return

    if compute_metrics:
        MC_metric_fname = utils.text_dict()["Folders"]["MC_METRICS"]
        utils.folder_tools.create_folder(MC_metric_fname)

    MCPARS = utils.enum_utils.enum2dict(TSF_enum.MOCO_PARAMS)
    par_idx = 0 if not onePhotonCheck else 1
    MOCOpar2use = {k: v[par_idx] for k, v in MCPARS.items()}

    if params is None:
        params = {}

    niter_rig = params.get("NITER_RIG", None)
    max_shifts = params.get("MAX_SHIFTS", None)
    splits_rig = params.get("SPLITS_RIG", None)
    strides = params.get("STRIDES", None)
    overlaps = params.get("OVERLAPS", None)
    splits_els = params.get("SPLITS_ELS", None)
    upsample_factor_grid = params.get("UPSAMPLE_FACTOR", None)
    max_deviation_rigid = params.get("MAX_DEV_RIG", None)
    shifts_opencv = params.get("SHIFTS_OPENCV", None)
    nonneg_movie = params.get("NONNEG", None)
    border_nan = params.get("BORDER_NAN", None)
    pw_rigid = params.get("PW_RIGID", None)
    use_cuda = params.get("USE_CUDA", None)
    gSig_filt = params.get("GSIG_FILT", None)

    # get the default values from the enum class if not provided by user
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
    gSig_filt = MOCOpar2use["GSIG_FILT"] if gSig_filt is None else gSig_filt

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

    # export parameters to file
    _export_params2file(
        mc_iter=mc_iter,
        niter_rig=niter_rig,
        max_shifts=max_shifts,
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
        enum_name="MotionCorrParams",
        parFile="MotionCorr" + ("_onePhoton" if onePhotonCheck else ""),
    )

    # export the parameters to a file
    # also prints parameters to console for current motion correction run
    # utils.print_wFrame(f"Motion correcting...(this should take some time)")

    # initiate template as None
    template = None
    # create list to hold outputs
    mc_output = []

    # list to hold mmaps fnames for metrics
    mmaps4metrics = []

    if len(h5filename) <= 2:
        for idx, h5_2moco in enumerate(h5filename):
            if idx == 1 and not separate_channels:
                # set template for MC of next channel using template from previous channel
                utils.print_wFrame(
                    "Applying template from previous motion correction",
                    frame_num=0,
                )
                template = mc_output[0].total_template_els
            elif idx == 1 and separate_channels:
                # set template as None for MC of next channel
                # MC independent of previous channel
                utils.print_wFrame(
                    "Motion correction next channel independently of previous channel",
                    frame_num=0,
                )
                template = None

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
                # initiate motion correction object
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

                # perform motion correction
                mc.motion_correct(save_movie=True, template=template)
                if compute_metrics:
                    plot_shifts(mc, MC_metric_fname)
                    mmaps4metrics.append(mc.fname_tot_els[0])

                # perform additional motion correction iterations
                # if mc_iter > 1
                if mc_iter > 1:
                    _brk()
                    print(
                        f"!!!Performing {mc_iter - 1} more iteration(s) of motion correction!!!\n"
                    )

                    # keep track of original mmap name
                    # because caiman keeps appending __[PARAMS] to the end of even if it already exists
                    # so we need to keep track of the original name to avoid having multiple __[PARAMS]
                    mmap_orig = mc.fname_tot_els[0]

                    # list to hold mmaps fnames for metrics
                    # reset list if mc_iter > 1
                    # need to do this so that first mmap has correct name after renaming
                    mmaps4metrics = []

                    # if compute_metrics:
                    # mc_objs4metrics = []
                    # mc_objs4metrics.append(mc)

                    for it in range(mc_iter - 1):
                        mmap2use = _rename_mmap(mmap_orig, mmap_orig, it + 1)
                        mmaps4metrics.append(mmap2use)
                        print(f"Iteration {it + 1} of {mc_iter - 1}")
                        utils.print_wFrame(
                            f"Motion correcting: {os.path.basename(mmap2use)}"
                        )

                        # reduce max shifts by half
                        max_shifts = _reduce_tuple_params(max_shifts)
                        # reduce strides by 2/3
                        strides = _reduce_tuple_params(strides, div_modifier=1.5)
                        # reduce new strides by half and add 2
                        overlaps = _reduce_tuple_params(
                            strides, div_modifier=2, add_modifier=2
                        )
                        # reduct max deviation rig by 15%
                        max_deviation_rigid = int(max_deviation_rigid / 1.15)

                        print("Adjusting some parameters for motion correction:")
                        _export_params2file(
                            mc_iter=mc_iter,
                            niter_rig=niter_rig,
                            max_shifts=max_shifts,
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
                            enum_name="MotionCorrParams",
                            parFile=f"MotionCorr_iter{it + 2}"
                            + ("_onePhoton" if onePhotonCheck else ""),
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

                        # perform motion correction
                        mc.motion_correct(save_movie=True, template=template)

                        # plot shifts
                        if compute_metrics:
                            # mc_objs4metrics.append(mc)
                            plot_shifts(
                                mc_obj=mc,
                                metric_fname=MC_metric_fname,
                                iteration=it + 2,
                            )

                        # remove old mmap
                        # os.remove(mmap2use)

                        # rename new mmap to orgiginal name + iteration # which + 2 because loop starts at 0
                        mc.fname_tot_els[0] = _rename_mmap(
                            mc.fname_tot_els[0], mmap_orig, it + 2
                        )
                        mmaps4metrics.append(mc.fname_tot_els[0])

                        _brk()
                        print()

                # append the motion corrected movie to the output list
                mc2output = mc
                mc_output.append(mc2output)

                if compute_metrics:
                    print("Calculating metrics:")
                    import caiman as cm

                    bord_px_els = np.ceil(
                        np.maximum(
                            np.max(np.abs(mc.x_shifts_els)),
                            np.max(np.abs(mc.y_shifts_els)),
                        )
                    ).astype(int)

                    final_size = np.subtract(
                        mc.total_template_els.shape, 2 * bord_px_els
                    )

                    movie_orig = cm.load(h5_2moco)
                    movie_mc = [cm.load(mmap) for mmap in mmaps4metrics]

                    utils.print_wFrame("Plotting local correlations")
                    plot_local_correlations(
                        mov_obj_orig=movie_orig,
                        mov_obj_proc=movie_mc,
                        metric_fname=MC_metric_fname,
                    )

                    utils.print_wFrame("Running further quality assessments")
                    compute_metricNplot(
                        # mc_objs4metrics,
                        h5_2moco,
                        mmaps4metrics,
                        MC_metric_fname,
                        final_size,
                    )
                    utils.print_done_small_proc()
    else:
        raise ValueError(
            "Session with more than 2 channels was found. Please provide  a session with at most 2 channels"
        )

    return mc_output


def plot_shifts(mc_obj: object, metric_fname: str, iteration: int = 1) -> None:
    """
    Plot the x and y shifts from motion correction.

    Parameters:
        mc_obj (object): Motion correction object containing x_shifts_els and y_shifts_els attributes
        metric_fname (str): Path where the plot will be saved
        iteration (int, optional): Motion correction iteration number. Defaults to 1.

    Returns:
        None: Saves a plot of x and y shifts to the specified path
    """

    print("Plotting shifts...")
    fig_tools = utils.fig_tools

    fig_fname = os.path.basename(mc_obj.fname_tot_els[0].split("__")[0])
    if "iter" not in fig_fname:
        fig_fname += f"_iter{iteration}_shifts"
    else:
        fig_fname = fig_fname.split("_iter")[0] + f"_iter{iteration}_shifts"

    fig, ax = fig_tools.create_plt_subplots(figsize=(20, 10), nrows=1, ncols=2)
    ax = ax.flatten()

    ax[0].plot(mc_obj.x_shifts_els)
    ax[0].set_ylabel("x shifts (pixels)")
    ax[1].plot(mc_obj.y_shifts_els)
    ax[1].set_ylabel("y_shifts (pixels)")

    fig.supxlabel("Frames")

    fig_tools.save_figure(
        plt_figure=fig,
        fig_name=fig_fname,
        figure_save_path=metric_fname,
    )
    utils.print_done_small_proc()


def plot_local_correlations(
    mov_obj_orig: object,
    mov_obj_proc: list[object],
    metric_fname: str,
    cmap: str = "plasma",
) -> None:
    fig_tools = utils.fig_tools

    ncols = 1 + len(mov_obj_proc)

    fig, ax = fig_tools.create_plt_subplots(figsize=(20, 10), nrows=1, ncols=ncols)
    ax = ax.flatten()

    fig_tools.plot_imshow(
        fig=fig,
        axis=ax[0],
        data2plot=np.transpose(
            mov_obj_orig.local_correlations(eight_neighbours=True, swap_dim=False)
        ),
        title="Iteration 0",
        cmap=cmap,
    )

    for i, mov_obj in enumerate(mov_obj_proc):
        fig_tools.plot_imshow(
            fig=fig,
            axis=ax[i + 1],
            data2plot=np.transpose(
                mov_obj.local_correlations(eight_neighbours=True, swap_dim=False)
            ),
            title=f"Iteration {i + 1}",
            cmap=cmap,
        )

    fig_tools.save_figure(
        plt_figure=fig,
        fig_name="LocalCorrelations",
        figure_save_path=metric_fname,
    )


def plot_CorrOverFrames(correlations: list[object], metric_fname: str) -> None:
    fig_tools = utils.fig_tools

    fig, ax = fig_tools.create_plt_subplots(figsize=(20, 10), nrows=1, ncols=1)
    # ax = ax.flatten()

    for i, corr in enumerate(correlations):
        ax.plot(corr)
    ax.legend([f"Iteration {i}" for i in range(len(correlations))])
    ax.set_xlabel("Frames")
    ax.set_ylabel("Correlation (a.u.)")

    fig_tools.save_figure(
        plt_figure=fig,
        fig_name="CorrOverFrames",
        figure_save_path=metric_fname,
    )


def plot_ResOpticalFlow(mov_list: list[str], metric_fname: str) -> None:
    from caiman import load as cm_load

    npz2use = find_npz_files(metric_fname)
    npz2use.sort()

    npz2use = [os.path.join(os.getcwd(), metric_fname, npz) for npz in npz2use]

    fig, ax = utils.fig_tools.create_plt_subplots(
        figsize=(20, 20), nrows=len(mov_list), ncols=3
    )

    for idx, (mov, npz) in enumerate(zip(mov_list, npz2use)):
        with np.load(npz) as data:
            iter_str = f"Iteration {idx}"
            mean_img = np.transpose(np.mean(cm_load(mov), axis=0)[12:-12, 12:-12])
            lq, hq = np.percentile(mean_img, (0.5, 99.5))
            # plot mean image
            utils.fig_tools.plot_imshow(
                fig=fig,
                axis=ax[idx, 0],
                data2plot=mean_img,
                title="Mean",
                ylabel=iter_str,
                cmap="plasma",
                vmin=lq,
                vmax=hq,
            )
            # plot Corr image
            utils.fig_tools.plot_imshow(
                fig=fig,
                axis=ax[idx, 1],
                data2plot=np.transpose(data["img_corr"]),
                title="Correlation",
                cmap="plasma",
                vmin=0,
                vmax=0.35,
            )
            # plot mean optical flow
            flows = data["flows"]
            flows2plot = np.transpose(
                np.mean(
                    np.sqrt(flows[:, :, :, 0] ** 2 + flows[:, :, :, 1] ** 2), axis=0
                )
            )
            utils.fig_tools.plot_imshow(
                fig=fig,
                axis=ax[idx, 2],
                data2plot=flows2plot,
                title="Mean Optical Flow",
                cmap="plasma",
                vmin=0,
                vmax=0.3,
            )

    utils.fig_tools.save_figure(
        plt_figure=fig,
        fig_name="ResOpticalFlow",
        figure_save_path=metric_fname,
    )


def compute_metricNplot(
    # mc_objs: list[object],
    mov_orig_fname: str,
    mov_proc_fnames: list[str],
    metric_fname: str,
    final_size: tuple[int, int],
    swap_dim: bool = False,
    winsize: int = 100,
    resize_fact_flow: float = 0.2,
) -> None:
    def _format_npz_filename(npz: str) -> str:
        """Format NPZ filename to include iteration number."""
        if "iter" in npz:
            base_name = npz.split("__")[0]
            return f"{base_name}_metrics.npz"
        else:
            base_name = npz.split("_metrics")[0]
            return f"{base_name}_els_iter0_metrics.npz"

    from caiman.motion_correction import compute_metrics_motion_correction

    log_file = f"{metric_fname}/_mc_log_file.txt"
    os.makedirs(metric_fname, exist_ok=True)

    mov_list = [mov_orig_fname] + mov_proc_fnames

    with open(log_file, "a") as f:
        f.write("Quality Assessment:\n")
    utils.print_wFrame("Computing metrics for iteration 0 (original data)", frame_num=1)
    _, correlations_orig, _, _, crispness_orig = compute_metrics_motion_correction(
        mov_orig_fname,
        final_size[0],
        final_size[1],
        swap_dim,
        winsize=winsize,
        play_flow=False,
        resize_fact_flow=resize_fact_flow,
    )
    msg4log = f"|-- Iteration 0: crispness: {crispness_orig}\n"

    corr_list = []
    corr_list.append(correlations_orig)
    crispness_list = []
    for i, fname in enumerate(mov_proc_fnames):
        utils.print_wFrame(f"Computing metrics for iteration {i + 1}", frame_num=1)
        _, correlations_proc, _, _, crispness_proc = compute_metrics_motion_correction(
            fname,
            final_size[0],
            final_size[1],
            swap_dim,
            winsize=winsize,
            play_flow=False,
            resize_fact_flow=resize_fact_flow,
        )
        msg4log += f"|-- Iteration {i + 1}: crispness: {crispness_proc}\n"
        corr_list.append(correlations_proc)
        crispness_list.append(crispness_proc)

    with open(log_file, "a") as f:
        f.write(msg4log)

    # plot correlations
    plot_CorrOverFrames(corr_list, metric_fname)

    npz_output = find_npz_files(os.getcwd())

    # move npz files to metric_fname
    for npz in npz_output:
        npz_fname2use = _format_npz_filename(npz)
        os.rename(npz, npz_fname2use)
        shutil.move(npz_fname2use, os.path.join(metric_fname, npz_fname2use))

    plot_ResOpticalFlow(mov_list, metric_fname)

    # print("BEEP!")


def find_npz_files(dir2use: str) -> list[str]:
    return [
        f
        for f in os.listdir(dir2use)
        if f.endswith(utils.text_dict()["file_tag"]["NPZ"])
    ]

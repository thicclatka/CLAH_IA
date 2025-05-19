import numpy as np

from CLAH_ImageAnalysis.core import BaseClass as BC


class CRwROI_plots(BC):
    def __init__(
        self, ID: str, program_name: str, numSess: int, CRkey: dict, CRpar: dict
    ) -> None:
        self.class_type = "utils"
        BC.__init__(self, program_name=program_name, mode=self.class_type)
        # set ID
        self.ID = ID
        # RED, BLUE, GREEN
        self.RBG = [
            self.color_dict["red_base_rgb"],
            self.color_dict["blue_base_rgb"],
            self.color_dict["green_base_rgb"],
        ]

        self.colors = self.utils.color_dict_4cues()

        self.sim_rel_size = (20, 7)
        self.FOV_cluster_size = (20, 15)
        # self.cue_cell_color = self.fig_tools.hex_to_rgba(
        #     self.color_dict["red"], wAlpha=False
        # )
        # self.non_cue_cell_color = self.fig_tools.hex_to_rgba(
        #     self.color_dict["blue"], wAlpha=False
        # )

        self.numSess = numSess
        self.CRkey = CRkey
        self.CRpar = CRpar

        self.axis_fs = 18

    def _plot_footprints(
        self,
        footprints: list,
        title: str,
        alpha: float = 0.95,
        overlap_threshold: int = 1,
        overlap_enhance_factor: int = 3,
    ) -> None:
        """
        Plot the spatial footprints.

        Parameters:
            footprints (list): List of footprints to be plotted.
            title (str): Title of the plot.
            alpha (float, optional): Transparency of the footprints. Defaults to 0.95.
            overlap_threshold (int, optional): Threshold for overlapping footprints. Defaults to 1.
            overlap_enhance_factor (int, optional): Factor to enhance overlapping footprints. Defaults to 3.
        """
        # set up plot and figure titles
        plt_title, fig_title = self._pltNfig_title_config(title)
        main_title = f"{self.ID} \nSpatial Footprints: {plt_title}"
        save_title = f"spatial_footprints_{fig_title}"

        # set up fig & ax
        fig, ax = self.fig_tools.create_plt_subplots()
        ax.set_facecolor(self.color_dict["black"])

        combined_footprints = self.image_utils.create_combined_arr_wOverlap(
            array_list=footprints,
            overlap_threshold=overlap_threshold,
            overlap_enhance_factor=overlap_enhance_factor,
        )

        self.fig_tools.plot_imshow(
            fig,
            ax,
            combined_footprints,
            alpha=alpha,
            title=main_title,
            xticks=[],
            yticks=[],
        )

        # create legend
        legend_handles = []
        for i, _ in enumerate(footprints):
            color4legend = self.RBG[i % len(self.RBG)]
            legend_handles = self.fig_tools.create_legend_patch(
                legend2patch=legend_handles, facecolor=color4legend, label=f"S{i + 1}"
            )
        ax.legend(handles=legend_handles, loc="upper right", fontsize=10)
        self.fig_tools.save_figure(fig, save_title, self.CRkey["FIG_FOLDER"])

    def _plot_distSame_crwr(
        self,
        clusterer: object,
        kwargs_makeConjunctiveDistanceMatrix: dict,
    ) -> None:
        """
        Plot the distributions of same and different distances for the CRwROI analysis.

        Parameters:
            clusterer: The clusterer object used for analysis.
            kwargs_makeConjunctiveDistanceMatrix: Keyword arguments for the `make_conjunctive_distance_matrix` method.

        Returns:
            None
        """
        kwargs = kwargs_makeConjunctiveDistanceMatrix
        dConj, _, _, _, _, _ = clusterer.make_conjunctive_distance_matrix(
            s_sf=clusterer.s_sf,
            s_NN=clusterer.s_NN_z,
            s_SWT=clusterer.s_SWT_z,
            s_sesh=None,
            **kwargs,
        )
        dens_same_crop, dens_same, dens_diff, dens_all, edges, d_crossover = (
            clusterer._separate_diffSame_distributions(dConj)
        )
        if edges is None:
            print("No crossover found, so no plotting")
            return

        fig, axes = self.fig_tools.create_plt_subplots()
        axes.stairs(dens_same, edges, linewidth=5)
        axes.stairs(dens_same_crop, edges, linewidth=3)
        axes.stairs(dens_diff, edges)
        axes.stairs(dens_all, edges)
        axes.axvline(d_crossover, color="k", linestyle="--")
        axes.set_ylim([dens_same.max() * -0.5, dens_same.max() * 1.5])
        axes.set_title(self.CRkey["PAIRWISE"])
        axes.set_xlabel("distance or prob(different)")
        axes.set_ylabel("counts or density")
        axes.legend(
            [
                "same",
                "same (cropped)",
                "diff",
                "all",
                "diff - same",
                "all - diff",
                "(diff * same) * 1000",
                "crossover",
            ]
        )
        self.fig_tools.save_figure(
            fig,
            self.utils.convert_separate2conjoined_string(self.CRkey["PAIRWISE"]),
            self.CRkey["FIG_FOLDER"],
        )

    def _plot_sim_relationships_crwr(
        self,
        clusterer: object,
        max_samples: int,
        kwargs_scatter: dict,
        kwargs_makeConjunctiveDistanceMatrix: dict,
    ) -> None:
        """
        Plot the similarity relationships between different variables.

        Parameters:
            clusterer: The clusterer object used for similarity calculations.
            max_samples: The maximum number of samples to be plotted.
            kwargs_scatter: Additional keyword arguments for the scatter plot.
            kwargs_makeConjunctiveDistanceMatrix: Additional keyword arguments for the
                `make_conjunctive_distance_matrix` method of the clusterer object.

        Returns:
            None
        """
        dConj, _, sSF_data, sNN_data, sSWT_data, _ = (
            clusterer.make_conjunctive_distance_matrix(
                s_sf=clusterer.s_sf,
                s_NN=clusterer.s_NN_z,
                s_SWT=clusterer.s_SWT_z,
                s_sesh=None,
                **kwargs_makeConjunctiveDistanceMatrix,
            )
        )

        # subsampling similarities
        idx_rand = np.floor(
            np.random.rand(min(max_samples, len(dConj.data))) * len(dConj.data)
        ).astype(int)
        ssf_sub = sSF_data[idx_rand]
        snn_sub = sNN_data[idx_rand]
        sswt_sub = sSWT_data[idx_rand] if sSWT_data is not None else None
        dconj_sub = dConj.data[idx_rand]

        # set strings for x & y labels
        spfp_lbl = f"{self.CRkey['SIM']} {self.CRkey['SP_FP']}"
        nn_lbl = f"{self.CRkey['SIM']} {self.CRkey['NN']}"
        swt_lbl = f"{self.CRkey['SIM']} {self.CRkey['SWT']}"

        # plotting
        fig, axes = self.fig_tools.create_plt_subplots(
            ncols=3, figsize=self.sim_rel_size
        )
        fig.suptitle(self.CRkey["SIM_REL"])

        # Spatial Footprint vs Neural Network
        axes[0].scatter(ssf_sub, snn_sub, c=dconj_sub, **kwargs_scatter)
        axes[0].set_xlabel(spfp_lbl)
        axes[0].set_ylabel(nn_lbl)
        if sSWT_data is not None:
            # Scattering wavelet transform vs Spatial Footprint
            axes[1].scatter(ssf_sub, sswt_sub, c=dconj_sub, **kwargs_scatter)
            axes[1].set_xlabel(spfp_lbl)
            axes[1].set_ylabel(swt_lbl)
            # Scattering wavelet transform vs Neural Network
            axes[2].scatter(snn_sub, sswt_sub, c=dconj_sub, **kwargs_scatter)
            axes[2].set_xlabel(nn_lbl)
            axes[2].set_ylabel(swt_lbl)

        self.fig_tools.save_figure(
            fig,
            self.utils.convert_separate2conjoined_string(self.CRkey["SIM_REL"]),
            self.CRkey["FIG_FOLDER"],
        )

    def _plot_confidence_histograms(self, results: dict, bins: int = 50) -> None:
        """
        Plot histograms for cluster silhouette, cluster intra means, confidence, and sample silhouette.

        Parameters:
            results (dict): A dictionary containing the results of the analysis.
            bins (int, optional): The number of bins to use for the histograms. Defaults to 50.
        """
        confidence = (
            (np.array(results["quality_metrics"]["cluster_silhouette"]) + 1) / 2
        ) * np.array(results["quality_metrics"]["cluster_intra_means"])

        # plot histograms for cluster silhouette, cluster intra means, confidence, and sample silhouette
        fig, axes = self.fig_tools.create_plt_subplots(nrows=2, ncols=2)

        axes[0, 0].hist(results["quality_metrics"]["cluster_silhouette"], bins)
        axes[0, 0].set_xlabel("cluster silhouette")
        axes[0, 0].set_ylabel("cluster counts")

        axes[0, 1].hist(results["quality_metrics"]["cluster_intra_means"], bins)
        axes[0, 1].set_xlabel("cluster_intra_means")
        axes[0, 1].set_ylabel("cluster counts")

        axes[1, 0].hist(confidence, bins)
        axes[1, 0].set_xlabel("confidence")
        axes[1, 0].set_ylabel("cluster counts")

        axes[1, 1].hist(results["quality_metrics"]["sample_silhouette"], bins)
        axes[1, 1].set_xlabel("sample_silhouette score")
        axes[1, 1].set_ylabel("roi sample counts")

        self.fig_tools.save_figure(
            fig, "confidence_histograms", self.CRkey["FIG_FOLDER"]
        )

        # plot histo gor cluster counts by session

        _, counts = np.unique(results["clusters"]["labels"], return_counts=True)

        fig, axes = None, None
        fig, axes = self.fig_tools.create_plt_subplots()
        axes.hist(
            counts,
            results["ROIs"]["n_sessions"] * 2 + 1,
            range=(0, results["ROIs"]["n_sessions"] + 1),
        )
        axes.set_xlabel("n_sessions")
        axes.set_ylabel("cluster counts")
        self.fig_tools.save_figure(
            fig, "cluster_counts_by_session", self.CRkey["FIG_FOLDER"]
        )

    def _plot_FOV_clusters(
        self,
        subj_id: str,
        FOV_clusters: list,
        cluster_info: dict,
        isCell2plot: dict | None = None,
        PC_dict: dict | None = None,
        rectangular: bool = False,
        circular: bool = False,
        contour=False,
        preQC=False,
    ):
        """
        Plot FOV clusters.

        Parameters:
            subj_id (str): Subject ID.
            FOV_clusters (list): List of FOV clusters.
            cluster_info (dict): Cluster information.
            isCell2plot (dict | None): Dictionary containing boolean values indicating whether a cell is a specific cell type (which is defined by the keys) or not.
            PC_dict (dict | None): Dictionary containing cell type specific information necessary to plot the bounding boxes to highlight specific cell types.
            rectangular (bool, optional): Whether to plot rectangular bounding boxes. Defaults to False.
            circular (bool, optional): Whether to plot circular patches. Defaults to False.
            contour (bool, optional): Whether to plot contours. Defaults to False.

        Returns:
            None
        """
        if not preQC:
            QC2use = "POST_QC"
            clusterkey2use = self.CRkey["ACLUSTERS_QC"]
        else:
            QC2use = "PRE_QC"
            clusterkey2use = self.CRkey["ACLUSTERS"]

        # init total_sess var for legibility
        total_sess = self.numSess

        if isCell2plot is None:
            nrows2use = 1
            ctkeys2use = None
            ctkeysall = None
        else:
            nrows2use = 2
            ctkeys2use = [
                key
                for key in isCell2plot.keys()
                if key in ["CUE1", "CUE2", "BOTHCUES", "PLACE"]
            ]
            ctkeysall = ctkeys2use + ["NON"]

        if total_sess > 9 and isCell2plot is None:
            nrows2use = total_sess // 4
            ncols2use = (total_sess // nrows2use) + 1
        else:
            ncols2use = total_sess

        # plot the FOV clusters
        fig, axes = self.fig_tools.create_plt_subplots(
            ncols=ncols2use, nrows=nrows2use, figsize=self.FOV_cluster_size
        )

        cell_totals = []
        for sess in range(total_sess):
            total = cluster_info[clusterkey2use]
            total_dict = {}
            sum_of_specific_types = 0

            if ctkeys2use is not None:
                for cellType in ctkeys2use:
                    count = isCell2plot[cellType][QC2use][sess].sum()
                    total_dict[cellType] = count
                    sum_of_specific_types += count

            # Calculate 'NON' count
            non = total - sum_of_specific_types
            total_dict["NON"] = non
            total_dict["TOTAL"] = total

            cell_totals.append(total_dict)

        # set up titles for each subplot to be the session number
        titles = []
        for i, _ in enumerate(FOV_clusters):
            titles.append(f"Session {i + 1}")

        # set up suptitle
        suptitle = [
            f"{subj_id}",
            "Cells across Sessions",
            f"Total cells: {cluster_info[self.CRkey['UCELL']]}",
            f"Tracked Cells found: {cluster_info[self.CRkey['ACLUSTERS']]}",
        ]
        if not preQC:
            suptitle += [
                f"Clusters post thresholding: {cluster_info[self.CRkey['ACLUSTERS_QC']]}",
                f"Cluster Silhouette Threshold: {int(self.CRpar['CL_SILHOUETTE']):.1f}",
                f"Cluster Intra Means Threshold: {self.CRpar['CL_INTRA_MEANS']}",
            ]

        # create empty ticks for the x and y axes
        if ctkeys2use is not None:
            ax2use = axes[0, :total_sess]
        else:
            ax2use = axes[:total_sess]

        ticks = self.fig_tools.empty_tick_maker(total_sess)
        self.fig_tools.plot_imshow(
            fig,
            ax2use,
            FOV_clusters,
            suptitle=suptitle,
            title=titles,
            xticks=ticks,
            yticks=ticks,
        )

        if ctkeys2use is not None:
            if rectangular:
                for cidx, cellType in enumerate(ctkeys2use):
                    PC_bounds = PC_dict[cellType][self.CRkey["BOUND"]]
                    for idx, bounding_box in enumerate(PC_bounds):
                        self.fig_tools.plot_bounding_box(
                            axis=ax2use[idx],
                            bounding_box=bounding_box,
                            edgecolor=self.colors[cellType],
                        )
            elif circular:
                for cidx, cellType in enumerate(ctkeys2use):
                    PC_centroids = PC_dict[cellType][self.CRkey["CENT"]]
                    PC_radii = PC_dict[cellType][self.CRkey["MIN_D"]]
                    for idx, (centroids, radii) in enumerate(
                        zip(PC_centroids, PC_radii)
                    ):
                        for cent, rad in zip(centroids, radii):
                            self.fig_tools.plot_circle_patch(
                                ax2use[idx],
                                cent,
                                rad,
                                edgecolor=self.colors[cellType],
                            )
            elif contour:
                for cidx, cellType in enumerate(ctkeys2use):
                    PC_contours = PC_dict[cellType][self.CRkey["CONTOUR"]]
                    for idx, sess_ctrs in enumerate(PC_contours):
                        for cell_ctrs in sess_ctrs:
                            for ctrs in cell_ctrs:
                                self.fig_tools.plot_contour(
                                    ax2use[idx],
                                    ctrs,
                                    edgecolor=self.colors[cellType],
                                )

        if ctkeys2use is not None:
            for i, ctotal in enumerate(cell_totals):
                props = []
                labels = []
                labels4plot = []
                for ctkey in ctotal.keys():
                    if ctkey == "TOTAL":
                        continue
                    labels4plot.append(f"{ctkey}: {ctotal[ctkey]}")
                    labels.append(ctkey)
                    props.append(ctotal[ctkey] / ctotal["TOTAL"])
                colors4plot = [self.colors[ctkey] for ctkey in labels]

                # Create pie chart
                wedges, _, _ = axes[1, i].pie(
                    props,
                    labels=labels4plot,
                    colors=colors4plot,
                    autopct="%1.1f%%",
                    startangle=140,
                    wedgeprops={"alpha": 0.8},
                )
                axes[1, i].set_aspect("equal")

        # if total_sess < 4:
        #     axes[1, -1].legend(
        #         wedges,
        #         labels,
        #         title=f"Total: {cluster_info[clusterkey2use]}",
        #         loc="upper right",
        #         fontsize=self.axis_fs,
        #     )
        # axes[1, 0].set_ylabel("Proportion of Tracked Cells")

        save_title = "FOV_clusters" if not preQC else "FOV_clusters_preQC"
        self.fig_tools.save_figure(fig, save_title, self.CRkey["FIG_FOLDER"])

    def _pltNfig_title_config(self, title: str) -> tuple:
        """Generates the plot and figure titles for the given title.

        Parameters:
            title (str): The title to be used in the plot and figure titles.

        Returns:
            tuple: A tuple containing the plot title and figure title.
        """
        plt_title = self.CRkey["SHIFT_PLT"].format(title)
        fig_title = self.CRkey["SHIFT_FIG"].format(title.lower())
        return plt_title, fig_title

import numpy as np

from CLAH_ImageAnalysis.core import BaseClass as BC
from CLAH_ImageAnalysis.unitAnalysis import UA_enum


class CCF_Plotter(BC):
    """
    A class for plotting various types of data related to CCF (Cue Cell Finder Function) analysis.

    Attributes:
        refLapType (str): The reference lap type.
        lapTypeNameArr (list): A list of lap type names.
        fig_save_path (str): The path to save the generated figures.
        figsize (tuple): The size of the figure.
        colors (dict): A dictionary mapping cue types to colors.
        desired_order (list): A list specifying the desired order of lap types.

    Methods:
        __init__(self, refLapType, lapTypeNameArr): Initializes the CCF_plotter object.
        _setup_Fig_Axes_for_subplot(self): Sets up the figure and axes for subplots.
        plotUnitByTuning(self, posRatesRef, posRatesNonRef, IndToPlot, CC_Type, fig_name): Plots unit tuning data.
        _image_plot(self, arr_to_plot, xlabel, title): Plots an image.
        plot_MFR_SP(self, maxInd, CellCueInd, omitRatio, midFieldRateRef, midFieldRateOmit): Plots MFR (Mean Firing Rate) and SP (Spatial Profile) data.
        plot_spatial_profile(self, X, Y, title, xlabel, ylabel, shape, ax1, ax0_5): Plots a spatial profile.
        plot_cueTrigSig_OR_cueAmp(self, dict_to_plot, fname, ind, title_fs, stitle_fs, SEM, VIO): Plots cue-triggered signal or cue amplitude data.
        _plot_cueTrigSig_CTSuplot_eachCell(self): Plots cue-triggered signal or cue amplitude data for each cell.
        _plot_SEMcueTrigSig_OR_VioPlotcueAmp(self): Plots cue-triggered signal or cue amplitude data using SEM or Violin Plot.
    """

    def __init__(
        self, refLapType: str, lapTypeNameArr: list, forPres: bool = False
    ) -> None:
        """
        Initialize the CCF_Plotter object.

        Parameters:
            refLapType (str): The reference lap type.
            lapTypeNameArr (list): A list of lap type names.
            forPres (bool): Whether to export figures as svgs as well for presentation. Default is False.
        """
        self.program_name = "CCF"
        self.class_type = "utils"
        BC.__init__(self, self.program_name, mode=self.class_type)

        self.refLapType = refLapType
        self.lapTypeNameArr = lapTypeNameArr
        self.fig_save_path = self.text_lib["FIGSAVE"]["DEFAULT"]
        self.figsize = (15, 15)
        self.colors = self.utils.color_dict_4cueTrigSigplots()
        self.cueType_abbrev = self.text_lib["cueType_abbrev"]
        self.CCFPtxt = self.enum2dict(UA_enum.CCF_PLOT)
        self.desired_order = self.CCFPtxt["ORDER"]

        # font sizes
        self.axis_fs = 16
        self.title_fs = 18
        self.stitle_fs = 20

        self.Markers = {
            "PC": "+",
            "CC": "*",
            "MC": "x",
            "CUE1": "1",
            "CUE2": "2",
            "CUE": "C",
            "TONE": "T",
            "LED": "L",
        }

        self.cue_cell_color = self.fig_tools.hex_to_rgba(
            self.color_dict["red"], wAlpha=False
        )
        self.pc_color = self.fig_tools.hex_to_rgba(
            self.color_dict["blue"], wAlpha=False
        )

        self.forPres = forPres

        self.types2process = None

    def find_types2process_from_CCT(self) -> None:
        """
        Find the types to process from the Cue Cell Table.
        """

        cues2consider = [
            cue
            for cue in self.cue_types_set
            if "OMIT" not in cue and "SWITCH" not in cue
        ]
        if len(cues2consider) == 2:
            both_cues = set(self.CueCellTable["CUE1_IDX"]) & set(
                self.CueCellTable["CUE2_IDX"]
            )
            cue1 = list(set(self.CueCellTable["CUE1_IDX"]) - both_cues)
            cue2 = list(set(self.CueCellTable["CUE2_IDX"]) - both_cues)
            both_cues = list(both_cues)
            both_cues.sort()
            cue1.sort()
            cue2.sort()
            led, tone = None, None
        elif len(cues2consider) == 1:
            cue1 = list(self.CueCellTable["CUE1_IDX"])
            both_cues, cue2, led, tone = None, None, None, None
            cue1.sort()
        elif len(cues2consider) == 3:
            # TODO: implement for 3 cue version
            raise NotImplementedError("3 cue version not implemented yet")

        place = self.CueCellTable["PLACE_IDX"]
        type_order = ["CUE1", "CUE2", "BOTH", "TONE", "LED", "PLACE"]

        self.list_idc = []
        list_ind2check = [cue1, cue2, both_cues, led, tone, place]
        for list_ind in list_ind2check:
            if list_ind is not None:
                self.list_idc.append(list_ind)

        self.types2process = [
            cue
            for c_idx, cue in enumerate(type_order)
            if list_ind2check[c_idx] is not None
        ]

    def plot_CueCellTable(self, CueCellTable: dict) -> None:
        """
        Plot the Cue Cell Table.

        Parameters:
            CueCellTable (dict): The Cue Cell Table.
        """

        def autopct_format(pct: float) -> str:
            """
            Format the percentage for the pie chart.

            Parameters:
                pct (float): The percentage.

            Returns:
                str: The formatted percentage.
            """
            return f"{pct:.1f}%" if pct > 2 else ""

        self.CueCellTable = CueCellTable

        fig, ax = self.fig_tools.create_plt_subplots(figsize=self.figsize)

        # define cell categories & respective markers
        cellCategories = [
            ("CUE", "Cue Cells", self.Markers["CC"]),
            ("PLACE", "Place Cells", self.Markers["PC"]),
            ("START", "Start Cells", ""),
            ("NON", "NA Cells", ""),
        ]

        self.mossyCheck = False
        if "MOSSY" in CueCellTable.keys():
            self.mossyCheck = True
            cellCategories.append(("MOSSY", "Mossy Cells", self.Markers["MC"]))

        proportions = []
        labels = []

        for key, label, marker in cellCategories:
            prop = CueCellTable[f"{key}_prop"]
            proportions.append(prop)
            marker_text = f" ({marker})" if marker else ""
            labels.append(f"{label}{marker_text} - {prop * 100:.1f}%")

        wedges, _, autotexts = ax.pie(
            proportions,
            labels=None,
            autopct=autopct_format,
            startangle=90,
            textprops=dict(fontsize=self.axis_fs),
            pctdistance=0.90,
        )

        ax.legend(
            wedges,
            labels,
            title=f"Cell Types (Total ROIs: {CueCellTable['TOTAL']})",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1),
            fontsize=self.axis_fs,
            title_fontsize=self.title_fs,
        )

        ax.axis("equal")
        ax.set_title("Cell Type Proportions", fontsize=self.title_fs)

        self.fig_tools.save_figure(
            plt_figure=fig,
            fig_name="CueCellTable",
            figure_save_path=self.fig_save_path,
            forPres=self.forPres,
        )

    def _get_cell_str_with_marker(self, cell: str, latex: bool = False) -> str:
        """
        Get the cell string with appropriate markers, formatted for Matplotlib's mathtext.

        Parameters:
            cell (str): The cell string (e.g., "Cell_1").

        Returns:
            str: The formatted cell string (e.g., r"Cell 1$\mathregular{^{\\ast12}}$").
        """
        cell_num = int(cell.split("_")[-1])
        base_cell_str = cell.replace("_", " ")

        marker_str = ""

        is_cue_cell = cell_num in self.CueCellTable.get("CUE_IDX", [])
        is_place_cell = cell_num in self.CueCellTable.get("PLACE_IDX", [])
        is_mossy_cell = self.mossyCheck and cell_num in self.CueCellTable.get(
            "MOSSY_IDX", []
        )

        superscript_markers = []
        if is_cue_cell:  # Only add numeric/letter superscripts if it's a CUE cell
            if "CUE1" in self.CueCellTable and cell_num in self.CueCellTable.get(
                "CUE1_IDX", []
            ):
                superscript_markers.append("1")
            if "CUE2" in self.CueCellTable and cell_num in self.CueCellTable.get(
                "CUE2_IDX", []
            ):
                superscript_markers.append("2")
            if "TONE" in self.CueCellTable and cell_num in self.CueCellTable.get(
                "TONE_IDX", []
            ):
                superscript_markers.append("T")
            if "LED" in self.CueCellTable and cell_num in self.CueCellTable.get(
                "LED_IDX", []
            ):
                superscript_markers.append("L")

        # Construct the marker string using mathtext format
        if is_cue_cell and latex:
            superscript_content = "".join(superscript_markers)
            if superscript_content:
                marker_str = r"$\mathregular{^{\ast " + superscript_content + r"}}$"
            else:
                marker_str = r"$\mathregular{^{\ast}}$"
        elif is_cue_cell and not latex:
            marker_str = self.Markers["CC"]
        elif is_place_cell:
            marker_str = self.Markers["PC"]
        elif is_mossy_cell:
            marker_str = self.Markers["MC"]
        # else marker_str remains "" for START or NON

        # Return combined string
        return base_cell_str + marker_str

    def _setup_Fig_Axes_for_subplot(self) -> tuple[object, np.ndarray]:
        """
        Set up the figure and axes for subplots.

        Returns:
            fig (matplotlib.figure.Figure): The figure object.
            axes (numpy.ndarray): The array of axes objects.
        """
        fig, axes = self.fig_tools.create_plt_subplots(
            nrows=self.num_rows, ncols=self.num_cols, figsize=self.figsize
        )
        if self.num_rows == 1 and self.num_cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        return fig, axes

    def plotUnitByTuning(
        self,
        posRatesRef: np.ndarray,
        posRatesNonRef: dict,
        IndToPlot: np.ndarray,
        CC_Type: str,
        fig_name: str,
    ) -> None:
        """
        Plots the unit tuning for a given set of position rates.

        Parameters:
            posRatesRef (numpy.ndarray): The position rates for the reference lap type.
            posRatesNonRef (dict): A dictionary containing the position rates for non-reference lap types.
            IndToPlot (numpy.ndarray): The indices of the units to plot.
            CC_Type (str): The type of cue cells.
            fig_name (str): The name of the figure.
        """

        num_keys = len(posRatesNonRef)
        # add +2 to account for missing ref cue in posRatesNonRef
        self.numCols = int(np.ceil((num_keys + 2) / 2))
        if self.numCols == 1:
            self.numCols = 2

        # setting up array to plot, sorting inds
        arr_to_plot = posRatesRef[IndToPlot, :]
        self.maxVal = np.max(arr_to_plot)
        _, sortInd = self.dep.find_maxIndNsortmaxInd(arr_to_plot)
        fig, axes = self.fig_tools.create_plt_subplots(
            nrows=2, ncols=self.numCols, figsize=self.figsize, flatten=True
        )
        cbar_ref = None

        # plotting refLapType
        # also return cbar_ref for colorbar
        cbar_ref = self._plot_per_tuning(
            fig=fig,
            axis4implot=axes[0],
            axis4mean=axes[-1],
            arr_to_plot=arr_to_plot[sortInd, :],
            title=self.lapTypeNameArr[self.refLapType],
            return_cbar_ref=True,
        )

        # plotting remaining lapTypes
        for idx, nr_lT in enumerate(posRatesNonRef, start=1):
            if posRatesNonRef[nr_lT] is not None:
                if self.lapTypeNameArr[self.refLapType] == self.CCFPtxt["CUELEDTONE"]:
                    if nr_lT == "OMITALL":
                        title = nr_lT
                    else:
                        title = self.CCFPtxt[f"{nr_lT}_TTL"]
                elif nr_lT == "OMITCUE1" and "CUEwOPTO" in posRatesNonRef.keys():
                    title = "OPTO"
                else:
                    title = nr_lT

                # plot accordingly
                self._plot_per_tuning(
                    fig=fig,
                    axis4implot=axes[idx],
                    axis4mean=axes[-1],
                    arr_to_plot=posRatesNonRef[nr_lT][IndToPlot, :],
                    title=title,
                )

        # turn on legend and set title & xticks for mean plot
        axes[-1].set_title(self.CCFPtxt["MEAN_PR"])
        axes[-1].set_xticks(self.CCFPtxt["XTICKS"])
        axes[-1].legend()

        cbar_ax = fig.add_axes([0.05, 0.15, 0.01, 0.7])
        cbar = fig.colorbar(cbar_ref, cax=cbar_ax)
        cbar.ax.yaxis.set_ticks_position(self.CCFPtxt["CBAR_POS"])
        cbar.ax.yaxis.set_label_position(self.CCFPtxt["CBAR_POS"])

        fig.suptitle(f"{CC_Type} {self.CCFPtxt['CUECELLS']}")
        # plt.tight_layout()
        fig.subplots_adjust(top=0.91, left=0.1)
        self.fig_tools.save_figure(
            plt_figure=fig,
            fig_name=fig_name,
            tight_layout=False,
            figure_save_path=self.fig_save_path,
            forPres=self.forPres,
        )

    def _plot_per_tuning(
        self,
        fig: object,
        axis4implot: object,
        axis4mean: object,
        arr_to_plot: np.ndarray,
        title: str,
        return_cbar_ref: bool = False,
    ):
        im = self.fig_tools.plot_imshow(
            fig=fig,
            axis=axis4implot,
            data2plot=arr_to_plot,
            xlabel=self.CCFPtxt["XPOS"],
            xticks=self.CCFPtxt["XTICKS"],
            title=title,
            aspect="auto",
            cmap="jet",
            vmax=self.maxVal,
            return_im=True,
        )

        axis4mean.plot(np.mean(arr_to_plot, axis=0), label=title)

        if return_cbar_ref:
            return im

    def plot_MFR_SP(
        self,
        maxInd: np.ndarray,
        CellCueInd: np.ndarray,
        omitRatio: np.ndarray,
        midFieldRateRef: np.ndarray,
        midFieldRateOmit: np.ndarray,
    ) -> None:
        """
        Plots the spatial profile and histograms for cue/omit and cue vs. omit.

        Parameters:
            maxInd (np.ndarray): X values for the spatial profile of cue/omit.
            CellCueInd (np.ndarray): Y values for the spatial profile of cue/omit.
            omitRatio (np.ndarray): Values for the omitRatio histogram.
            midFieldRateRef (np.ndarray): X values for the spatial profile of cue vs. omit.
            midFieldRateOmit (np.ndarray): Y values for the spatial profile of cue vs. omit.

        Returns:
        None
        """
        fig, axes = self.fig_tools.create_plt_subplots(
            ncols=2, nrows=2, figsize=self.figsize, flatten=True
        )

        # Plot Spatial Profile
        self.plot_spatial_profile(
            ax=axes[0],
            X=maxInd[CellCueInd],
            Y=omitRatio,
            title="Spatial Profile of cue / omit",
            xlabel="pos",
            ylabel="cue/omit",
            ax1=True,
        )
        # Plot cue vs omit
        self.plot_spatial_profile(
            ax=axes[1],
            X=midFieldRateRef,
            Y=midFieldRateOmit,
            title="Cue(s) vs. Omit",
            xlabel="Cue",
            ylabel="Omit",
            ax0_5=True,
        )
        # Plot omitRatio histogram
        ax = axes[2]
        ax.hist(omitRatio, 20)
        ax.set_title("omitRatio Histogram")
        ax.set_xlabel("cue / omit")

        self.fig_tools.save_figure(
            plt_figure=fig,
            fig_name="Cue_Omit_SpatialProfile",
            figure_save_path=self.fig_save_path,
        )

    def plot_spatial_profile(
        self,
        ax: object,
        X: np.ndarray,
        Y: np.ndarray,
        title: str,
        xlabel: str,
        ylabel: str = "",
        shape: str = ".",
        ax1: bool = False,
        ax0_5: bool = False,
    ) -> None:
        """
        Plot the spatial profile.

        Parameters:
            ax (object): The axis object to plot on.
            X (np.ndarray): X-coordinates of the data points.
            Y (np.ndarray): Y-coordinates of the data points.
            title (str): Title of the plot.
            xlabel (str): Label for the x-axis.
            ylabel (str, optional): Label for the y-axis. Defaults to an empty string.
            shape (str, optional): Marker shape for the scatter plot. Defaults to ".".
            ax1 (bool, optional): Whether to add a horizontal line at y=2. Defaults to False.
            ax0_5 (bool, optional): Whether to add a diagonal line from (0,0) to (0.5,0.5). Defaults to False.
        """

        ax.scatter(X, Y, marker=shape)
        color = self.fig_tools.hex_to_rgba(self.color_dict["black"], wAlpha=False)
        if ax1:
            ax.axhline(y=2, color=color, linestyle=":")
        elif ax0_5:
            ax.plot([0, 0.5], [0, 0.5], color=color)
        ax.set_title(title, fontsize=self.axis_fs)
        ax.set_xlabel(xlabel, fontsize=self.axis_fs)
        ax.set_ylabel(ylabel, fontsize=self.axis_fs)

    def plot_meanOptoTrigSig(self, meanOTS: dict, ind: list = []) -> None:
        """
        Plot the mean opto-triggered signal.

        Parameters:
            meanOTS (dict): A dictionary containing the mean opto-triggered signal.
            ind (list, optional): A list of indices to select specific cells to plot. Defaults to an empty list.
        """

        self.ind = ind

        color_map = self.colors

        self.num_cols = 1
        self.num_rows = len(meanOTS.keys())

        fig, axes = self._setup_Fig_Axes_for_subplot()

        for i, (oType, mOTS_by_cueTypes) in enumerate(meanOTS.items()):
            for cueType, meanVal in mOTS_by_cueTypes.items():
                self.fig_tools.plot_SEM(
                    arr=meanVal,
                    color=color_map[cueType],
                    ax=axes[i],
                    x_ind=self.ind,
                    vline=True,
                    # baseline=self.baseline_slice,
                )
            cell_num = mOTS_by_cueTypes[next(iter(mOTS_by_cueTypes))].shape[-1]
            axes[i] = self._set_xticks(axes[i])
            self.fig_tools.add_text_box(
                ax=axes[i],
                text=r"$N_{{\mathrm{{cells}}}} = {}$".format(cell_num),
                xpos=0.05,
                va="top",
                ha="left",
                transform=axes[i].transAxes,
            )
            if i % self.num_cols == 0:
                axes[i].set_ylabel("Cue triggered average", fontsize=self.axis_fs)
            axes[i].set_title(oType, fontsize=self.title_fs)

        self._cT_cA_plotSaver(fig=fig, fname="Opto", extra_txt="MeanSEM_OptoGroups")

    def plot_cueTrigSig_OR_cueAmp(
        self,
        dict_to_plot: dict,
        fname: str,
        ind: list = [],
        SEM: bool = False,
        VIO: bool = False,
        OPTO: str | None = None,
        plot_by_cell_type: bool = False,
    ):
        """
        Plot cue-triggered signal or cue amplitude.

        Parameters:
            dict_to_plot (dict): A dictionary containing the data to plot. The keys are cell names and the values are dictionaries of cue types and their corresponding data.
            fname (str): The filename to save the plot.
            ind (list, optional): A list of indices to select specific cells to plot. Defaults to an empty list.
            SEM (bool, optional): If True, plot a single subplot figure with all cells using SEM. Defaults to False.
            VIO (bool, optional): If True, plot a single subplot figure with all cells using Violin Plot. Defaults to False.
            OPTO (str | None, optional): If provided, the plot will be saved with the specified opto type. Defaults to None.
            plot_by_cell_type (bool, optional): If True, plot the cue-triggered signal for each cell type. Defaults to False.
        """

        # store stuff into self
        self.dict_to_plot = dict_to_plot
        self.ind = ind
        self.fname = fname
        self.SEM = SEM
        self.VIO = VIO
        self.OPTO = OPTO

        # get slice for baseline
        self.baseline_slice = slice((np.abs(self.ind[0]) - 5), np.abs(self.ind[0]) + 1)

        # Get the number of cells
        num_cells = len(dict_to_plot.keys())

        # Calculate the number of rows and columns for the subplots
        self.num_rows = int(num_cells**0.5)
        self.num_cols = num_cells // self.num_rows
        if (
            num_cells % self.num_rows > 0
        ):  # If there are leftover cells, add an extra column
            self.num_cols += 1

        # Get all unique cue types
        self.cue_types_set = set()
        for cell, cueTypes in dict_to_plot.items():
            self.cue_types_set.update(cueTypes.keys())

        if SEM or VIO:  # Plot single subplot fig with all cells (SEM)
            self._plot_SEMcueTrigSig_OR_VioPlotcueAmp()
        else:  # Plot for each cue type
            self._plot_cueTrigSig_CTSuplot_eachCell()

        if plot_by_cell_type and SEM:
            # plot average CTS by significant cell type
            self._plot_cueTrigSig_byCellType()

        if plot_by_cell_type and VIO:
            # plot average cueAmp by significant cell type
            self._plot_cueAmp_byCellType()

    def _plot_cueTrigSig_CTSuplot_eachCell(self) -> None:
        """
        Plot cue-triggered signals for each cell in a subplot for each cue type.

        This method iterates over each cue type and creates a new figure for each cue type.
        For each cell, if the cell is associated with the current cue type, a line plot is
        created using the cell's cue-triggered signals. The x-axis values are generated based
        on the length of the cue-triggered signals. The method also sets the x-axis ticks and
        the title for each subplot. Unused subplots are removed before saving the plot.
        """

        for cueType in self.cue_types_set:
            self.print_wFrm(f"{cueType}", frame_num=1)
            # Create a new figure for each cue type
            fig, axes = self._setup_Fig_Axes_for_subplot()
            for i, (cell, cell_cueTypes) in enumerate(self.dict_to_plot.items()):
                if cueType in cell_cueTypes:
                    # If size is 1, plot a line plot
                    x_val = np.arange(
                        self.ind[0], self.ind[0] + len(cell_cueTypes[cueType])
                    )
                    axes[i].plot(x_val, cell_cueTypes[cueType], label=cueType)
                    axes[i] = self._set_xticks(axes[i])

                cell_str = self._get_cell_str_with_marker(cell, latex=True)
                axes[i].set_title(cell_str, fontsize=self.title_fs)

            # Remove unused subplots
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])
            self._cT_cA_plotSaver(fig=fig, fname=self.fname, suptitle=cueType)

    def _plot_SEMcueTrigSig_OR_VioPlotcueAmp(self) -> None:
        """
        Plot cue-triggered signals using SEM or Violin Plot.
        """

        # Create a mapping of cueTypes to colors
        color_map = self.colors
        fig, axes = self._setup_Fig_Axes_for_subplot()
        for i, (cell, cell_cueTypes) in enumerate(self.dict_to_plot.items()):
            for j, cueType in enumerate(cell_cueTypes):
                if self.SEM:
                    plot_data = cell_cueTypes[cueType]
                    self.fig_tools.plot_SEM(
                        arr=plot_data,
                        color=color_map[cueType],
                        ax=axes[i],
                        x_ind=self.ind,
                        vline=True,
                        # baseline=self.baseline_slice,
                    )
                    axes[i] = self._set_xticks(axes[i])
                    if i % self.num_cols == 0:
                        axes[i].set_ylabel(
                            "Cue triggered average", fontsize=self.axis_fs
                        )
                elif self.VIO:
                    # Sort cell_cueTypes by the order of keys in color_map
                    sorted_cell_cueTypes = {
                        c: cell_cueTypes[c]
                        for c in color_map.keys()
                        if c in cell_cueTypes
                    }
                    # plot accordingly
                    for k, sort_CT in enumerate(sorted_cell_cueTypes):
                        self.fig_tools.violin_plot(
                            ax=axes[i],
                            x=[k] * len(sorted_cell_cueTypes[sort_CT]),
                            y=sorted_cell_cueTypes[sort_CT],
                            color=color_map[sort_CT],
                        )
                    # create x tick labels for boxplot
                    abbrev_labels = [
                        self.cueType_abbrev[cueType]
                        for cueType in sorted_cell_cueTypes.keys()
                        if cueType in self.cueType_abbrev
                    ]
                    axes[i].set_xticks(range(0, len(abbrev_labels)))
                    axes[i].set_xticklabels(abbrev_labels, fontsize="small")
                    if i % self.num_cols == 0:
                        axes[i].set_ylabel("Post-cue Amp", fontsize=self.axis_fs)
            axes[i].set_title(
                self._get_cell_str_with_marker(cell, latex=True), fontsize=self.title_fs
            )
        # Remove unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        if self.SEM:
            suptitle = self.CCFPtxt["SEM"]
        elif self.VIO:
            suptitle = self.CCFPtxt["VIO"]

        if self.OPTO is not None:
            suptitle = f"{suptitle}_{self.OPTO}"
        self._cT_cA_plotSaver(fig=fig, fname=self.fname, extra_txt=suptitle)
        self._create_sep_legend()

    def _plot_cueTrigSig_byCellType(self) -> None:
        """
        Plot cue-triggered signals for each cell type.
        """
        color_map = self.colors

        if self.types2process is None:
            self.find_types2process_from_CCT()

        cts2plot = {}
        for cell_type, cell_idc in zip(self.types2process, self.list_idc):
            if cell_type not in cts2plot.keys():
                cts2plot[cell_type] = {}
            for cell in cell_idc:
                cell2use = f"Cell_{cell}"
                for cueType in self.dict_to_plot[cell2use].keys():
                    if cueType not in cts2plot[cell_type].keys():
                        cts2plot[cell_type][cueType] = []
                    cts2plot[cell_type][cueType].append(
                        np.nanmean(self.dict_to_plot[cell2use][cueType], axis=1)
                    )
        fig, axes = self.fig_tools.create_plt_subplots(
            nrows=len(self.types2process) // 2, ncols=2, flatten=True
        )

        for cell_idx, cell_type in enumerate(cts2plot.keys()):
            for cueType in cts2plot[cell_type].keys():
                data2plot = np.array(cts2plot[cell_type][cueType]).copy()
                for idx, cd in enumerate(data2plot):
                    if cd.max() > 100:
                        # set to nan if max is greater than 100
                        data2plot[idx] = np.full_like(cd, np.nan)
                # data2plot = data2plot / data2plot.max()
                n_cells = data2plot.shape[0]

                self.fig_tools.plot_SEM(
                    arr=data2plot.T,  # need to transpose to get average over cells per timepoint
                    color=color_map[cueType],
                    ax=axes[cell_idx],
                    x_ind=self.ind,
                    vline=True,
                )
                axes[cell_idx].set_title(
                    f"{cell_type} Cells (N = {n_cells})", fontsize=self.title_fs
                )
                axes[cell_idx] = self._set_xticks(axes[cell_idx])
                axes[cell_idx].set_ylabel(
                    "Cue triggered Response", fontsize=self.axis_fs
                )

        self._cT_cA_plotSaver(
            fig=fig,
            fname=self.fname,
            extra_txt=f"{self.CCFPtxt['SEM']}_byCellType",
        )

    def _plot_cueAmp_byCellType(self) -> None:
        """
        Plot cue amplitude for each cell type.
        """
        color_map = self.colors

        if self.types2process is None:
            self.find_types2process_from_CCT()

        ca2plot = {}
        for cell_type, cell_idc in zip(self.types2process, self.list_idc):
            if cell_type not in ca2plot.keys():
                ca2plot[cell_type] = {}
            for cell in cell_idc:
                cell2use = f"Cell_{cell}"
                for cueType in self.dict_to_plot[cell2use].keys():
                    if cueType not in ca2plot[cell_type].keys():
                        ca2plot[cell_type][cueType] = []
                    ca2plot[cell_type][cueType].append(
                        np.mean(self.dict_to_plot[cell2use][cueType])
                    )

        fig, axes = self.fig_tools.create_plt_subplots(
            nrows=len(self.types2process) // 2, ncols=2, flatten=True
        )

        abbrev_labels = [
            self.cueType_abbrev[cueType]
            for cueType in ca2plot[next(iter(ca2plot))].keys()
            if cueType in self.cueType_abbrev
        ]

        for cell_idx, cell_type in enumerate(ca2plot.keys()):
            for cT_idx, cueType in enumerate(ca2plot[cell_type].keys()):
                data2plot = np.array(ca2plot[cell_type][cueType]).copy()

                n_cells = data2plot.shape[0]
                self.fig_tools.violin_plot(
                    ax=axes[cell_idx],
                    x=[cT_idx] * len(data2plot),
                    y=data2plot,
                    color=color_map[cueType],
                )
                axes[cell_idx].set_title(
                    f"{cell_type} Cells (N = {n_cells})", fontsize=self.title_fs
                )
                axes[cell_idx].set_xticks(range(0, len(abbrev_labels)))
                axes[cell_idx].set_xticklabels(abbrev_labels, fontsize="small")
                axes[cell_idx].set_ylabel("Post-cue Amp", fontsize=self.axis_fs)

        self._cT_cA_plotSaver(
            fig=fig,
            fname=self.fname,
            extra_txt=f"{self.CCFPtxt['VIO']}_byCellType",
        )

    def _cT_cA_plotSaver(
        self, fig: object, fname: str, suptitle: list = [], extra_txt: list = []
    ) -> None:
        """
        Saves the plot as a figure.

        Parameters:
            fig (object): The figure object to save.
            fname (str): The filename of the figure.
            suptitle (list, optional): The list of strings for the super title. Defaults to [].
            extra_txt (list, optional): The list of strings for additional text. Defaults to [].

        Returns:
            None
        """
        if suptitle:
            fig.suptitle(suptitle, fontsize=self.stitle_fs)
            figname = f"_{fname}_{suptitle}"
        elif extra_txt:
            figname = f"_{fname}_{extra_txt}"
        else:
            figname = f"_{fname}"
        self.fig_tools.save_figure(
            plt_figure=fig,
            fig_name=figname,
            figure_save_path=f"{self.fig_save_path}/{fname}",
            forPres=self.forPres,
        )

    def _set_xticks(self, ax: object, num_ticks: int = 6) -> object:
        """
        Set the x-axis ticks for the given axis.

        Parameters:
            ax (object): The axis object to set the ticks for.
            num_ticks (int, optional): The number of ticks to be displayed on the x-axis. Default is 6.

        Returns:
            ax (object): The modified axis object.
        """
        ax.set_xticks(np.linspace(self.ind[0], self.ind[-1] - 1, num=num_ticks))
        return ax

    def _create_sep_legend(self) -> None:
        """
        Creates a separate legend for the plot.

        This method creates a legend for the plot based on the color mapping
        defined by the `colors` attribute. The legend elements are created
        based on the `cue_types_set` and `desired_order` attributes.
        """

        color_map = {
            key: self.colors[key]
            for key in (self.cue_types_set & set(self.desired_order))
            if key in self.colors
        }
        fig_legend = self.fig_tools.create_separate_legend(color_map=color_map)

        self._cT_cA_plotSaver(fig=fig_legend, fname=self.fname, extra_txt="_legend")

    def plot_sigASpat_overDSImage(
        self, sigASpat: dict, sigDict: dict, DS_image: str, sig_idx: int, dir_idx: int
    ) -> None:
        """
        Plot the spatial profiles of significant cues over a downsampled image.

        Parameters:
            sigASpat (dict): Dictionary containing the spatial profiles of significant cues.
            sigDict (dict): Dictionary containing the significant cues.
            DS_image (str): Path to the downsampled image.
            sig_idx (int): Index of the significant cue.
            dir_idx (int): Index of the direction cue.
        """

        self.fname = "ASpatial_Profile"
        color_map = self.colors
        image_file = self.findLatest(DS_image)
        for main_cue in sigASpat:
            img = self.image_utils.read_image(image_file)

            for comp_cue in sigASpat[main_cue]:
                self.print_wFrm(f"{main_cue} vs. {comp_cue}", frame_num=1)
                # init fig & axis for each comp_cue
                fig, ax = self.fig_tools.create_plt_subplots(figsize=self.figsize)
                ax.imshow(img, cmap="gray", aspect="equal")

                sigDictArr = np.array(sigDict[main_cue][comp_cue])
                SigCellIdxArr = np.where(sigDictArr[:, sig_idx] == 1)[0]
                DirIdxArr = sigDictArr[:, dir_idx]
                DirIdxArr = DirIdxArr[SigCellIdxArr]
                for spat_idx in range(sigASpat[main_cue][comp_cue].shape[-1]):
                    data = sigASpat[main_cue][comp_cue][:, spat_idx]
                    data_2d = data.reshape(img.shape)
                    if DirIdxArr[spat_idx] == 1:
                        colorPlot = color_map[main_cue]
                    elif DirIdxArr[spat_idx] == -1:
                        colorPlot = color_map[comp_cue]
                    # convert hex to RGB
                    RGB_color = self.fig_tools.hex_to_rgba(colorPlot, wAlpha=False)
                    # create colormap from RGB_color
                    cmap = self.fig_tools.make_segmented_colormap(
                        cmap_name="DSImage", hex_color=RGB_color, from_white=True
                    )
                    self.fig_tools.plot_imshow(
                        fig=fig,
                        axis=ax,
                        data2plot=data_2d.T,
                        cmap=cmap,
                        alpha=1,
                        vmax=data.max(),
                        title=f"{main_cue} vs. {comp_cue}",
                    )
                self.fig_tools.save_figure(
                    plt_figure=fig,
                    fig_name=f"ASP_{main_cue}_v_{comp_cue}",
                    figure_save_path=f"{self.fig_save_path}/{self.fname}",
                )
        self._create_sep_legend()

    def plot_OptoDiff_overDSImage(
        self, DS_image: str, OptoDiff: dict, ASpat: np.ndarray
    ) -> None:
        """
        Plot the Optogenetic Inhibition Effect over a downsampled image.

        Parameters:
            DS_image (str): Path to the downsampled image.
            OptoDiff (dict): Dictionary containing the Optogenetic Inhibition Effect data.
            ASpat (np.ndarray): Array containing the spatial profiles of significant cues.
        """

        image_file = self.findLatest(DS_image)

        img = self.image_utils.read_image(image_file)

        color_pos = self.colors["CUEwOPTO"]
        color_neg = self.colors["CUE1"]

        fig, ax = self.fig_tools.create_plt_subplots(figsize=self.figsize)
        ax.imshow(img, cmap="gray", aspect="equal")

        for cell, (mdiff, _) in OptoDiff.items():
            cell_num = int(cell.split("_")[-1])
            data = ASpat[:, cell_num]
            data_2d = data.reshape(img.shape)

            RGB_color = self.fig_tools.hex_to_rgba(
                color_pos if mdiff > 0 else color_neg, wAlpha=False
            )
            cmap = self.fig_tools.make_segmented_colormap(
                cmap_name="DSImage", hex_color=RGB_color, from_white=True
            )

            self.fig_tools.plot_imshow(
                fig=fig,
                axis=ax,
                data2plot=data_2d.T,
                cmap=cmap,
                alpha=1,
                vmax=data.max(),
            )

            # label cell_num
            self.fig_tools.label_cellNum_overDSImage(
                axis=ax, data=data_2d, cell_str=cell
            )

            # for cue cells, find contour & plot accordingly
            if (
                cell_num in self.CueCellTable["CUE_IDX"]
                or cell_num in self.CueCellTable["PC_IDX"]
            ):
                if cell_num in self.CueCellTable["PC_IDX"]:
                    contColor2use = self.pc_color
                elif cell_num in self.CueCellTable["CUE_IDX"]:
                    contColor2use = self.cue_cell_color

                contours = self.dep.geometric_tools.find_contours(
                    arr2use=data_2d.T, contour_level=0.36
                )
                for ctr in contours:
                    self.fig_tools.plot_contour(
                        axis=ax, contour=ctr, edgecolor=contColor2use
                    )

        ax.set_title("Optogenetic Inhibition Effect")
        self.fig_tools.save_figure(
            plt_figure=fig,
            fig_name="ASP_OptoDiff",
            figure_save_path=f"{self.fig_save_path}/Opto",
        )

    def plot_OptoDiff(
        self, OptoDiff: dict, diff_val_tuple: tuple, threshold: int = 10**2
    ) -> None:
        """
        Plot the Optogenetic Difference in Amplitude.

        Parameters:
            OptoDiff (dict): A dictionary containing the Optogenetic Inhibition Effect data.
            diff_val_tuple (tuple): A tuple containing the difference values for positive and negative effects.
            threshold (int, optional): The threshold for the difference values. Defaults to 10**2.
        """

        def _threshold_checker(val, val2check=None):
            # if val2check is none, set it to val
            # this is to see if val2check is different from val
            val2check = val if val2check is None else val2check
            return val if np.abs(val2check) < threshold else np.nan

        def _plot_mean_line(vals, color, linestyle="--"):
            mean = np.nanmean(vals[np.abs(vals) < threshold])
            axes.axhline(y=mean, color=color, linestyle=linestyle)

        n = None
        fig, axes = self.fig_tools.create_plt_subplots()
        color_pos = self.colors["CUEwOPTO"]
        color_neg = self.colors["CUE1"]
        cell_nums = []
        all_means = []

        # plot mean line for diff_val_tuple
        # [0] is for positive values, [1] is for negative values
        # !this is current threshold to determine which cells to focus on
        _plot_mean_line(np.array(diff_val_tuple[0]), color_pos)
        _plot_mean_line(np.array(diff_val_tuple[1]), color_neg)

        for cell, diff in OptoDiff.items():
            cell_num = int(cell.split("_")[-1])
            if cell_num in self.CueCellTable["PC_IDX"]:
                cell_num = f"{cell_num}{self.Markers['PC']}"
            elif cell_num in self.CueCellTable["CUE_IDX"]:
                cell_num = f"{cell_num}{self.Markers['CC']}"
            else:
                cell_num = f"{cell_num}"

            cell_nums.append(cell_num)
            mean = diff[0]
            mean_post_check = _threshold_checker(mean)
            std = diff[1]
            std_post_check = _threshold_checker(std, val2check=mean)
            color2use = color_pos if mean > 0 else color_neg

            # store means for opto labels later
            # all_means.append(mean)
            all_means.append(mean_post_check)

            self.fig_tools.bar_plot(
                ax=axes,
                X=cell_num,
                Y=mean_post_check,
                yerr=std_post_check,
                color=color2use,
            )
            if np.isnan(mean_post_check):
                axes.set_yscale("symlog", linthresh=10)
                if color2use == color_pos:
                    thresh2use = threshold
                else:
                    thresh2use = -threshold
                axes.plot(cell_num, thresh2use, marker="*", color=color2use)

        if len(cell_nums) > 30 and len(cell_nums) < 80:
            n = 5
        elif len(cell_nums) >= 80:
            n = 10
        if n is not None:
            sparse_labels = [
                cell_num
                if index % n == 0
                else (self.Markers["CC"] if self.Markers["CC"] in cell_num else "")
                or (self.Markers["PC"] if self.Markers["PC"] in cell_num else "")
                for index, cell_num in enumerate(cell_nums)
            ]
            axes.set_xticks(range(len(cell_nums)))
            axes.set_xticklabels(sparse_labels)

        # set title
        axes.set_title("Optogenetic Inhibition Effect")

        # set xlabel
        axes.set_xlabel("Cell", fontsize=self.axis_fs)

        # set ylabel
        axes.set_ylabel(
            "Mean Difference in Amplitude (Cue w/ Opto - Cue)",
            fontsize=self.axis_fs,
        )

        # Determine positions for the y-axis labels
        max_mean = max(all_means)
        min_mean = min(all_means)
        pos_label_y = max_mean + (
            0.1 * max_mean
        )  # 10% above the highest mean for visibility
        neg_label_y = min_mean + (
            0.1 * min_mean
        )  # 10% below the lowest mean for visibility

        textbox_tuple = (
            (pos_label_y, "Inhibition increases activity", color_pos),
            (neg_label_y, "Inhibition decreases activity", color_neg),
        )

        for y, text, color in textbox_tuple:
            self.fig_tools.add_text_box(
                ax=axes,
                text=text,
                xpos=0,
                ypos=y,
                color=color,
                ha="left",
                va="center",
                rotation="horizontal",
            )

        self.fig_tools.save_figure(
            plt_figure=fig,
            fig_name="OptoDiff",
            figure_save_path=f"{self.fig_save_path}/Opto",
        )

    def plot_Ca_Trace_wLaps(
        self,
        Ca_arr: np.ndarray,
        pos: np.ndarray,
        lapFrInds: list,
        pks: np.ndarray,
        cell_num: str,
    ) -> None:
        """
        Plot the calcium trace over laps.

        Parameters:
            Ca_arr (numpy.ndarray): The calcium array.
            pos (numpy.ndarray): The position array.
            lapFrInds (list): The indices of the laps.
            pks (numpy.ndarray): The peak indices.
            cell_num (str): The cell number.
        """
        fig = self.fig_tools.create_plotly()

        if pks.dtype == np.float64:
            pks = [int(pk) for pk in pks]

        # Normalize the position array
        # & have it match the max of the Ca_arr
        pos = (pos / pos.max()) * Ca_arr.max()
        self.fig_tools.add_plotly_trace(fig=fig, y=Ca_arr, mode="lines", name="Ca_arr")

        # add position to the plot
        self.fig_tools.add_plotly_trace(fig=fig, y=pos, mode="lines", name="pos")

        self.fig_tools.add_plotly_trace(
            fig=fig, x=pks, y=Ca_arr[pks], mode="markers", name="Peaks"
        )

        # add annotations for lap indices
        # makes each with a number
        for i, lap_index in enumerate(lapFrInds):
            fig.add_annotation(
                x=lap_index,
                y=Ca_arr.max(),
                text=str(i + 1),
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=-30,
            )

        # Update the layout
        fig.update_layout(
            title=f"Ca Trace Over Laps for Cell {cell_num}", xaxis_title="Time"
        )
        self.fig_tools.save_plotly(
            plotly_fig=fig,
            figure_save_path=f"{self.fig_save_path}/Traces",
            fig_name=f"Ca_Trace_Laps_Cell_{cell_num}",
        )

    def plot_cueVScueOpto(
        self,
        optoAmps: dict,
        OptoStd: dict,
        OptoIdx: dict,
        cues4opto: list,
        threshold: int = 10**2,
    ) -> None:
        """
        Plot the CUE vs CUEwOPTO amplitude.

        Parameters:
            optoAmps (dict): The opto amplitudes.
            OptoStd (dict): The opto standard deviations.
            OptoIdx (dict): The opto indices.
            cues4opto (list): The cues for opto.
            threshold (int, optional): The threshold for the difference values. Defaults to 10**2.
        """

        def filter_outliers(values):
            val_check = [val for val in values if val > threshold]
            if len(val_check) > 0:
                outliers_indices = [i for i, v in enumerate(values) if v in val_check]
                return outliers_indices
            else:
                return None

        fig, axes = self.fig_tools.create_plt_subplots(figsize=self.figsize)

        plus = "OPTOplus"
        minus = "OPTOminus"

        data = {
            key: {
                "cue_vals": optoAmps[key][cues4opto[0]],
                "cWo_vals": optoAmps[key]["CUEwOPTO"],
                "cue_std": OptoStd[key][cues4opto[0]],
                "cWo_std": OptoStd[key]["CUEwOPTO"],
                "idx": OptoIdx[key],
            }
            for key in [plus, minus]
        }

        max_vals = []
        outlier_counts = 0
        outlier_num = []

        for key in data.keys():
            color2use = self.color_dict["black"]

            cue_vals = np.array(data[key]["cue_vals"])
            cWo_vals = np.array(data[key]["cWo_vals"])

            cue_std = np.array(data[key]["cue_std"])
            cWo_std = np.array(data[key]["cWo_std"])

            optoIdx = data[key]["idx"]

            cellLabels = []
            for cell, _ in optoIdx:
                cell_str = self._get_cell_str_with_marker(cell)
                cellLabels.append(cell_str.split(" ")[-1])

            cue_outliers_indices = filter_outliers(cue_vals)
            cWo_outliers_indices = filter_outliers(cWo_vals)

            if cue_outliers_indices is not None and cWo_outliers_indices is not None:
                all_outliers_indices = list(
                    set(cue_outliers_indices).union(cWo_outliers_indices)
                )
                outlier_counts += len(all_outliers_indices)

                for outlier in all_outliers_indices:
                    outlier_num.append(cellLabels[outlier])

                # Remove outliers from both arrays
                filtered_cue_vals = np.delete(cue_vals, all_outliers_indices)
                filtered_cWo_vals = np.delete(cWo_vals, all_outliers_indices)
                filtered_cue_std = np.delete(cue_std, all_outliers_indices)
                filtered_cWo_std = np.delete(cWo_std, all_outliers_indices)

                filtered_cellLabels = np.delete(
                    np.array(cellLabels), all_outliers_indices
                )
            else:
                filtered_cue_vals = cue_vals
                filtered_cWo_vals = cWo_vals
                filtered_cue_std = cue_std
                filtered_cWo_std = cWo_std
                filtered_cellLabels = cellLabels

            axes.errorbar(
                filtered_cue_vals,
                filtered_cWo_vals,
                xerr=filtered_cue_std,
                yerr=filtered_cWo_std,
                fmt="o",
                color=color2use,
                ecolor=color2use,
                markersize=10,
            )
            for idx, label in enumerate(filtered_cellLabels):
                axes.annotate(
                    label,
                    (filtered_cue_vals[idx], filtered_cWo_vals[idx]),
                    textcoords="offset points",
                    xytext=(10, 8),
                    ha="center",
                    color=self.color_dict["red"],
                    fontsize=10,
                )

            axes.set_xlabel("CUE Only Max Amplitude", fontsize=self.axis_fs)
            axes.set_ylabel("CUEwOPTO Max Amplitude", fontsize=self.axis_fs)

            concatenated_vals = np.concatenate((filtered_cue_vals, filtered_cWo_vals))
            max_vals.append(np.nanmax(concatenated_vals))

        max_val = max(max_vals)
        axes.plot([0, max_val], [0, max_val], linestyle="--", color="gray")

        if outlier_counts > 0:
            outliers_text = f"Outliers (by cell num): {[num for num in outlier_num]}"

            axes.text(
                0.05,
                0.95,
                outliers_text,
                transform=axes.transAxes,
                fontsize=12,
                verticalalignment="top",
                bbox=dict(
                    boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"
                ),
            )

        # axes.legend()
        self.fig_tools.save_figure(
            plt_figure=fig,
            figure_save_path=f"{self.fig_save_path}/Opto",
            fig_name="cueVScueOpto",
            forPres=self.forPres,
        )

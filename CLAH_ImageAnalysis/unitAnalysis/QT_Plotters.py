import copy
import numpy as np
from sklearn.cluster import KMeans
from CLAH_ImageAnalysis.core import BaseClass as BC
from CLAH_ImageAnalysis.unitAnalysis import UA_enum


class QT_Plotters(BC):
    def __init__(
        self,
        sess_name: str,
        dpi: int = 300,
        Figure_Save_Path: str = "Figures",
        forPres: bool = False,
    ) -> None:
        """
        Initialize the QT_plotters class.

        Parameters:
            sess_name (str): The name of the session.
            dpi (int, optional): The DPI (dots per inch) for saving figures. Defaults to 300.
            Figure_Save_Path (str, optional): The path to save the figures. Defaults to "Figures".
            forPres (bool, optional): Whether to export svgs for presentation. Defaults to False.
        """

        self.program_name = "QT"
        self.class_type = "utils"
        BC.__init__(self, self.program_name, mode=self.class_type)

        self.CSSkey = self.enum2dict(UA_enum.CSS)
        self.sess_name = sess_name
        self.dpi = dpi
        self.Figure_Save_Path = Figure_Save_Path
        self.forPres = forPres

    def _find_pc_N_sortInd_N_postRates(self) -> list:
        """
        Find place cells, sort indices, and post rates.

        Returns:
            list: A list of post rates for each lap type.
        """
        maxInd = np.argmax(self.posRates_ref, axis=1)
        self.sortInd = np.argsort(maxInd)
        posRatesArr = [[] for _ in range(self.numLapTypes)]

        for lapType in range(self.numLapTypes):
            lapType_key = f"lapType{lapType + 1}"
            try:
                posRates = copy.deepcopy(
                    self.cueShiftStruc[self.CSSkey["PCLS"]][lapType_key][
                        self.CSSkey["POSRATE"]
                    ]
                )
                if self.usePC:
                    posRates = posRates[self.pc, :]
                posRatesArr[lapType] = posRates[self.sortInd, :]
            except Exception as e:
                print(f"Problem with lap type {lapType}: {e}")

        return posRatesArr

    def cueShiftTuning(
        self,
        cueShiftStruc: dict,
        refLapType: int,
        C_Temporal: np.ndarray,
        lapTypeNameArr: list,
        usePC: bool = True,
        figsize: tuple = (15, 10),
    ) -> None:
        """
        Plot cue shift tuning.

        Parameters:
            cueShiftStruc (dict): The cue shift structure.
            refLapType (int): The reference lap type.
            C_Temporal (ndarray): The temporal calcium data.
            lapTypeNameArr (list): The lap type name array.
            usePC (bool, optional): Whether to use place cells. Defaults to True.
            figsize (tuple, optional): The figure size. Defaults to (15, 10).
        """

        self.usePC = usePC
        self.refLapType = refLapType
        self.refLapType_key = f"lapType{self.refLapType + 1}"
        self.numLapTypes = len(cueShiftStruc[self.CSSkey["PCLS"]])
        # deepcopy CSS
        self.cueShiftStruc = copy.deepcopy(cueShiftStruc)
        # extract isPC from CSS
        self.isPC = self.cueShiftStruc[self.CSSkey["PCLS"]][self.refLapType_key][
            self.CSSkey["SHUFF"]
        ][self.CSSkey["ISPC"]]
        # select place cells for reference lap
        self.pc = np.where(self.isPC == 1)[0]
        # use place cell posRates from ref lap
        self.posRates_ref = self.cueShiftStruc[self.CSSkey["PCLS"]][
            self.refLapType_key
        ][self.CSSkey["POSRATE"]][self.pc, :]

        posRatesArr = self._find_pc_N_sortInd_N_postRates()

        nan_mask = np.array(lapTypeNameArr) == "nan"
        nan_sum = np.sum(nan_mask)
        if nan_sum > 0.5:
            legit_lt = self.numLapTypes - nan_sum
            # nan_idx = np.where(np.array(lapTypeNameArr) == "nan")[0]
        else:
            legit_lt = self.numLapTypes

        numCols = int(np.ceil((legit_lt + 1) / 2))
        if numCols == 1:
            numCols = 2
        lapTypeList = []
        lapTypeList_template = list(range(self.numLapTypes))
        # creates lapTypeList where refLapType is always first
        # -- note: for 2odor 1 location: refLap is cue1
        # omitboth or omitcue1 always 2nd
        lapTypeList = self.refLapType
        lapTypeList = [lapTypeList] + [self.numLapTypes - 1]
        lapTypeList = lapTypeList + [
            lt
            for lt in lapTypeList_template
            if lt != self.refLapType and lt != self.numLapTypes - 1
        ]

        # line up lapTypeName with lapTypeList
        lapTypeListName = [lapTypeNameArr[i] for i in lapTypeList]

        # Set up figure and GridSpec
        fig, axes = self.fig_tools.create_plt_subplots(
            nrows=2, ncols=numCols, figsize=figsize, flatten=True
        )
        xlabel = "Position"
        xticks = [0, 25, 50, 75, 100]
        cbar_ref = None

        # fig = plt.figure(figsize=figsize)
        # gs = gridspec.GridSpec(2, numCols)

        idx_plt = 0
        for idx, lt in enumerate(lapTypeList):
            if np.array(lapTypeListName)[idx] != "nan" and posRatesArr[lt].size > 0:
                ax = axes[idx_plt]
                im = self.fig_tools.plot_imshow(
                    fig=fig,
                    axis=ax,
                    data2plot=posRatesArr[lt],
                    xlabel=xlabel,
                    xticks=xticks,
                    title=f"{lapTypeListName[idx]}",
                    aspect="auto",
                    cmap="jet",
                    vmax=np.max(posRatesArr[self.refLapType]),
                    return_im=True,
                )
                cbar_ref = im
                idx_plt += 1
        cbar_ax = fig.add_axes([0.05, 0.15, 0.01, 0.7])
        cbar = fig.colorbar(cbar_ref, cax=cbar_ax)
        cbar.ax.yaxis.set_ticks_position("left")
        cbar.ax.yaxis.set_label_position("left")

        # Plot mean of each on the next subplot
        # ax = plt.subplot(gs[-1, -1])  # Plot on the last subplot
        ax = axes[-1]
        for i, j in enumerate(lapTypeList):
            if np.array(lapTypeListName)[i] != "nan":
                ax.plot(np.mean(posRatesArr[j], axis=0), label=f"{lapTypeListName[i]}")
        ax.legend()
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_title("Mean posRate")

        fig.suptitle(self.sess_name)
        # fig.tight_layout()
        # Remove unused subplots
        # for ax in fig.axes:
        #     if not ax.get_children():  # If the axes is empty
        #         fig.delaxes(ax)  # Remove the axes
        fig.subplots_adjust(top=0.91, left=0.1)
        self.fig_tools.save_figure(
            fig,
            f"{self.sess_name}_Plots_cueShiftTuning",
            self.Figure_Save_Path,
            self.dpi,
            tight_layout=False,
            forPres=self.forPres,
        )

        try:
            C = C_Temporal

            if C is not None:
                if usePC:
                    cpc = C[self.pc, :]
                else:
                    cpc = C

                # Normalize calcium data
                cpc2 = (cpc - np.min(cpc, axis=1)[:, None]) / (
                    np.max(cpc, axis=1) - np.min(cpc, axis=1)
                )[:, None]

                # Plot calcium data
                fig, axes = self.fig_tools.create_plt_subplots(figsize=figsize)
                self.fig_tools.plot_imshow(
                    fig=fig,
                    axis=axes,
                    data2plot=cpc2[
                        np.argsort(np.argmax(posRatesArr[refLapType], axis=1)), :
                    ],
                    aspect="auto",
                    cmap="jet",
                    suptitle="Calcium activity for place cells",
                )

                # Adjust spacing to make room for the suptitle
                fig.subplots_adjust(top=0.9)
                self.fig_tools.save_figure(
                    fig,
                    f"{self.sess_name}_CaActivity",
                    self.Figure_Save_Path,
                    self.dpi,
                )

        except Exception as e:
            print("Error loading or processing calcium imaging data:", e)

    def pks_amps_wavef(
        self,
        Ca_arr: np.ndarray,
        peaks: list,
        pkInds: list,
        amps: np.ndarray,
        cell_num: int,
    ) -> None:
        """
        Plot peak amplitudes and waveforms.

        Parameters:
            Ca_arr (ndarray): The calcium trace array.
            peaks (list): The list of peak indices.
            pkInds (list): The list of peak indices for clustering.
            amps (ndarray): The array of amplitudes.
        """

        try:
            kind = KMeans(n_clusters=2).fit_predict(amps.reshape(-1, 1))
        except Exception as e:
            print("Can't perform kmeans:", e)
            kind = None

        # Plotting
        fig, axes = self.fig_tools.create_plt_subplots(nrows=2, ncols=2, flatten=True)

        # Convert peaks to integers
        peaks = [int(i) for i in peaks]

        # Subplot 1
        ax = axes[0]
        t = range(len(Ca_arr))
        ax.plot(t, Ca_arr, label="Ca Trace")
        ax.scatter(
            [t[i] for i in peaks],
            [Ca_arr[i] for i in peaks],
            color="r",
            marker="*",
            label="Peaks",
        )
        if kind is not None:
            ax.scatter(
                [t[inds] for idx, inds in enumerate(pkInds) if kind[idx] == 0],
                [Ca_arr[inds] for idx, inds in enumerate(pkInds) if kind[idx] == 0],
                color="g",
                marker="x",
                label="Kind 0",
            )
            ax.scatter(
                [t[inds] for idx, inds in enumerate(pkInds) if kind[idx] == 1],
                [Ca_arr[inds] for idx, inds in enumerate(pkInds) if kind[idx] == 1],
                color="m",
                marker="x",
                label="Kind 1",
            )
        ax.legend()

        # Subplot 2
        ax = axes[1]
        for i in peaks:
            try:
                ax.plot(
                    Ca_arr[max(i - 100, 0) : min(i + 300, len(Ca_arr))]
                    - Ca_arr[max(i - 15, 0)]
                )
            except Exception as e:
                print(f"Error in plotting waveform for peak {i}: {e}")

        # Subplot 3
        ax = axes[2]
        dPks = np.diff(peaks)
        ax.plot((dPks / max(dPks)) * max(amps), label="1/rate")
        ax.plot(amps, label="Amplitudes")
        ax.legend()
        ax.set_xlabel("Spike #")

        # Subplot 4
        ax = axes[3]
        ax.scatter(np.ones(len(amps)), amps, marker="x")

        self.fig_tools.save_figure(
            plt_figure=fig,
            fig_name=f"pks_amps_wavef_{cell_num}",
            figure_save_path=f"{self.Figure_Save_Path}/peaks",
        )

    def plot_cellNum_overDSImage(
        self,
        A_Spat: np.ndarray,
        DS_image: str,
        fig_name: str = "_CellNum_DSImage",
    ) -> None:
        """
        Plot cell number over DS image.

        Parameters:
            A_Spat (ndarray): The spatial correlation coefficient array.
            DS_image (str): The path to the DS image.
            fig_name (str, optional): The name of the figure. Defaults to "_CellNum_DSImage".
        """

        file_check = self.utils.check_folder(
            f"{self.Figure_Save_Path}/{fig_name}{self.file_tag['PNG']}"
        )

        if not file_check:
            img = self.image_utils.read_image(self.findLatest(DS_image))

            fig, ax = self.fig_tools.create_plt_subplots()
            ax.imshow(img, cmap="grey", aspect="equal")

            cell_num = A_Spat.shape[-1]

            for cell in range(cell_num):
                data = A_Spat[:, cell]
                data_2d = data.reshape(img.shape)
                RGB_color = self.fig_tools.hex_to_rgba(
                    self.color_dict["green"], wAlpha=False
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

                self.fig_tools.label_cellNum_overDSImage(
                    axis=ax,
                    data=data_2d,
                    cell_str=f"Cell_{cell}",
                    color=self.color_dict["red"],
                    fontsize=10,
                )

            self.fig_tools.save_figure(
                plt_figure=fig,
                fig_name=fig_name,
                figure_save_path=self.Figure_Save_Path,
            )
        else:
            self.print_wFrm(f"{fig_name} already exists.")
            self.print_wFrm("Skipping")

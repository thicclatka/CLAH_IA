import os
import re
import warnings
import numpy as np
import seaborn as sns
from typing import Optional
from scipy.stats import sem
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.patches import Polygon
from matplotlib.font_manager import FontProperties
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.colors import Normalize
from typing import Any
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
# import plotly.tools as tls
# import plotly.io as pio

from CLAH_ImageAnalysis.utils import create_multiline_string
from CLAH_ImageAnalysis.utils import text_dict
from CLAH_ImageAnalysis.utils import color_dict
from CLAH_ImageAnalysis.utils import saveNloadUtils


if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
    warnings.filterwarnings("ignore", message="This plugin does not support raise()")
    warnings.filterwarnings(
        "ignore", message="This plugin does not support propagateSizeHints()"
    )

warnings.filterwarnings("ignore", category=FutureWarning)


def interactive_mode(plt_on: bool = False) -> None:
    """
    Set the interactive mode for matplotlib.
    """
    if plt_on:
        plt.ion()
    else:
        plt.ioff()


def check_color_format(color: str | tuple) -> str:
    """
    Check if a color is in hex or RGB format.

    Parameters:
        color (str | tuple): The color to check.

    Returns:
        str: The format of the color ("hex", "rgb", or "unknown").
    """
    assert color is not None, "Color cannot be None"
    assert isinstance(color, (str, tuple)), "Color must be string or tuple"

    if isinstance(color, tuple):
        assert len(color) in [3, 4], "RGB(A) tuple must have 3 or 4 values"
        assert all(isinstance(v, (int, float)) for v in color), (
            "RGB values must be numbers"
        )

    if isinstance(color, str) and re.match("^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{8})$", color):
        return "hex"
    elif (
        isinstance(color, tuple)
        and len(color) in [3, 4]
        and all(0 <= value <= 255 for value in color)
    ):
        return "rgb"
    else:
        return "unknown"


def determine_font2use(font_family: str) -> str:
    """
    Determine the font family to use. If the font family is not found, it will fall back to Arial. If Arial is not found, it will use the default sans-serif font.

    Parameters:
        font_family (str): The font family to use.

    Returns:
        str: The font family to use.
    """
    try:
        FontProperties(font_family)
        font2use = font_family
    except ValueError:
        # warnings.warn(f"Font {font_family} not found. Falling back to Arial.")
        try:
            FontProperties("Arial")
            font2use = "Arial"
        except ValueError:
            # warnings.warn("Arial not found. Using default sans-serif font.")
            font2use = "sans-serif"

    return font2use


def tighten_layoutWspecific_axes(
    coords2tighten: list,
) -> None:
    """
    Tighten the layout of the figure for specific axes.

    Parameters:
        plt_figure (matplotlib.figure.Figure): The figure to be tightened.
        coords2tighten (list): The coordinates of the axes to be tightened.
    """
    plt.tight_layout(rect=coords2tighten)


def save_figure(
    plt_figure,
    fig_name: str,
    figure_save_path: str | None = None,
    dpi: int = 300,
    tight_layout: bool = True,
    close_fig: bool = True,
    forPres: bool = False,
    NOPNG: bool = False,
    pValDict: dict = None,
    mean_semDict: dict = None,
    # font_family: str = "sans-serif",
    font_family: str = "Lato",
) -> None:
    """
    Save a matplotlib figure to a file.

    Parameters:
        plt_figure (matplotlib.figure.Figure): The figure to be saved.
        fig_name (str): The name of the figure file.
        figure_save_path (str): The directory path where the figure will be saved.
        dpi (int, optional): The resolution of the saved figure in dots per inch. Default is 300.
        tight_layout (bool, optional): Whether to apply tight layout to the figure before saving. Default is True.
        close_fig (bool, optional): Whether to close the figure after saving. Default is True.
        forPres (bool, optional): Whether to save the figure in presentation format. Default is False.
        NOPNG (bool, optional): If set to True, the figure will not be saved as a PNG file. Default is False.
        pValDict (dict, optional): The dictionary to be saved as a JSON file. Default is None.
        mean_semDict (dict, optional): The dictionary to be saved as a JSON file. Default is None.
        font_family (str, optional): The font family to be used for the figure. Default is "Lato".
    """

    def saveDictVals2json(
        dict_to_save: dict, subdir: str, fig_name: str, figure_save_path: str
    ) -> None:
        """
        Helper function to save a dictionary to a file.

        Parameters:
            dict_to_save (dict): The dictionary to be saved.
            subdir (str): The subdirectory where the file will be saved.
            fig_name (str): The base name of the JSON file.
            figure_save_path (str): The directory path where the figure will be saved.
        """

        json_name = fig_name.replace(PNG_end, "")
        saveNloadUtils.savedict2file(
            dict_to_save,
            f"{subdir}/{json_name}",
            filename=f"{figure_save_path}/{subdir}/{json_name}",
            filetype_to_save=JSON_end,
        )

    font2use = determine_font2use(font_family)
    for text_obj in plt_figure.findobj(match=lambda x: hasattr(x, "set_fontfamily")):
        text_obj.set_fontfamily(font2use)

    PNG_end = text_dict()["file_tag"]["PNG"]
    JSON_end = text_dict()["file_tag"]["JSON"]

    # checks to see if .png is at the end
    if not fig_name.endswith(PNG_end):
        fig_name += PNG_end
    # Ensure the directory exists
    if figure_save_path is not None and not os.path.exists(figure_save_path):
        os.makedirs(figure_save_path)

    # Construct the full file path
    if figure_save_path is not None:
        file_path = os.path.join(figure_save_path, fig_name)
    else:
        # if figure_save_path is None, save the figure in the current working directory
        file_path = fig_name

    if tight_layout:
        plt.tight_layout()

    # Save the figure
    if not NOPNG:
        plt_figure.savefig(file_path, dpi=dpi)

    if forPres:
        # plt.rcParams["ps.useafm"] = False
        # plt.rcParams["ps.fonttype"] = 42

        plt.rcParams["svg.fonttype"] = "none"

        # for axis in plt_figure.get_axes():
        #     axis.set_rasterized(False)

        # EPS_end = text_dict()["file_tag"]["EPS"]
        EPS_end = text_dict()["file_tag"]["SVG"]
        eps_name = fig_name.replace(PNG_end, EPS_end)
        eps_fpath = os.path.join(figure_save_path, "Presentation")

        if not os.path.exists(eps_fpath):
            os.makedirs(eps_fpath)

        eps_fpath = os.path.join(eps_fpath, eps_name)
        # plt_figure.savefig(eps_fpath, format="eps", dpi=dpi, bbox_inches="tight")
        plt_figure.savefig(eps_fpath, format="svg", dpi=dpi)

    # Close the figure
    if close_fig:
        plt.close("all")
    # print(f"Figure saved to {file_path}")

    # Save pValDict and mean_semDict to JSON files if they are provided
    if pValDict is not None:
        saveDictVals2json(pValDict, "pVals", fig_name, figure_save_path)
    if mean_semDict is not None:
        saveDictVals2json(mean_semDict, "mean_sem", fig_name, figure_save_path)

    return


def hex_to_rgba(hex_color: str, alpha: float = 1.0, wAlpha: bool = True) -> tuple:
    """
    Convert a hexadecimal color to an RGBA tuple.

    Parameters:
        hex_color (str): The hexadecimal color code.
            The color code should be a string representing a hexadecimal color value.
        alpha (float): The alpha value for the RGBA tuple (default is 1.0).
            The alpha value should be a float between 0.0 and 1.0, representing the transparency of the color.
        wAlpha (bool): Whether to include the alpha value in the RGBA tuple (default is True).
            If set to True, the returned RGBA tuple will include the alpha value.
            If set to False, the returned RGBA tuple will only include the RGB values.

    Returns:
        tuple: The RGBA tuple representing the converted color.
            The returned tuple will have four elements: (red, green, blue, alpha).
            The RGB values will be integers between 0 and 255, representing the color channels.
            The alpha value will be a float between 0.0 and 1.0, representing the transparency of the color.
    """
    if wAlpha:
        return mcolors.to_rgba(hex_color, alpha)
    else:
        return mcolors.to_rgb(hex_color)


def make_segmented_colormap(
    cmap_name: str, hex_color: str, from_white: bool = False
) -> mcolors.LinearSegmentedColormap:
    """
    Create a colormap that transitions from transparent to the given hex color.

    Parameters:
        hex_color (str): The hex color code to transition to.

    Returns:
        matplotlib.colors.LinearSegmentedColormap: The segmented colormap.
    """

    color_rgba = hex_to_rgba(hex_color)
    if from_white:
        color_tuple = (1, 1, 1, 0)
    else:
        color_tuple = (0, 0, 0, 0)
    return mcolors.LinearSegmentedColormap.from_list(
        cmap_name, [color_tuple, color_rgba, color_rgba]
    )


def create_cmap4categories(num_categories: int, cmap_name: str | None = None):
    """
    Create a colormap for a given number of categories.

    Parameters:
        num_categories (int): The number of categories.
        cmap_name (str, optional): The name of the colormap to use. Defaults to "tab10".

    Returns:
        cmap (matplotlib.colors.LinearSegmentedColormap): The created colormap.
    """
    if cmap_name is None:
        cmap_name = "tab10"

    if cmap_name not in plt.colormaps():
        raise ValueError(f"Colormap {cmap_name} not found.")

    cmap = plt.get_cmap(cmap_name, num_categories)
    return cmap


def make_segmented_colormap_wOverlap(
    colors: list, num_categories: int, base: int
) -> mcolors.LinearSegmentedColormap:
    """
    Create a segmented colormap with overlapping colors.

    Parameters:
        colors (list): A list of colors in either hex or rgb format.

    Returns:
        cmap (matplotlib.colors.LinearSegmentedColormap): The created segmented colormap.

    """
    # Calculate the maximum number of distinct combinations
    max_combinations = 2**num_categories - 1

    # Generate exponential steps using logspace
    if max_combinations > len(colors):
        raise ValueError(
            "Not enough colors provided for the number of session combinations."
        )

    colors4cmap = []
    for color in colors:
        if check_color_format(color) == "hex":
            colors4cmap.append(hex_to_rgba(color))
        elif check_color_format(color) == "rgb":
            colors4cmap.append(color)

    # # Normalize factor to cover the range from 0 to 1 in the colormap
    # color_steps = np.logspace(0, num_categories - 1, num=max_combinations, base=10)
    # color_steps /= color_steps.max()
    # color_steps = np.insert(color_steps, 0, 0)

    # # # Create the color list for the colormap
    # color_list = [(color_steps[i], color) for i, color in enumerate(colors4cmap)]

    # Create a LinearSegmentedColormap
    return mcolors.LinearSegmentedColormap.from_list("custom_gradient", colors4cmap)


def make_abrupt_colormap(hex_color: str) -> mcolors.ListedColormap:
    """
    Create a colormap that transitions abruptly to the given hex color.

    Parameters:
    hex_color (str): The hex color code to transition to.

    Returns:
    ListedColormap: A colormap object that transitions abruptly to the given hex color.
    """
    return mcolors.ListedColormap(["none", hex_color])


def create_legend_patch(
    legend2patch: list,
    facecolor: str,
    label: str,
    edgecolor: str = None,
    hatch: str = None,
    alpha: float = 1.0,
    marker: str | None = None,
) -> list:
    """
    Create a legend patch with the given face color and label.

    Parameters:
        legend2patch (list): The list of legend patches to append the new patch to.
        facecolor (str): The face color of the legend patch.
        label (str): The label of the legend patch.
        edgecolor (str): The edge color of the legend patch.
        hatch (str): The hatch pattern of the legend patch.
        marker (str): The marker of the legend patch.

    Returns:
        list: The updated list of legend patches.
    """

    facecolor = facecolor if isinstance(facecolor, tuple) else hex_to_rgba(facecolor)

    if marker is not None:
        facecolor = color_dict()["black"] if facecolor is None else facecolor
        edgecolor = color_dict()["black"] if edgecolor is None else edgecolor
        legend2patch.append(
            Line2D(
                [0],
                [0],
                color="none",
                marker=marker,
                markerfacecolor=facecolor,
                markeredgecolor=edgecolor if edgecolor else "none",
                markersize=10,
                label=label,
            ),
        )
    else:
        legend2patch.append(
            Patch(
                facecolor=facecolor,
                label=label,
                edgecolor=edgecolor,
                hatch=hatch,
                alpha=alpha,
            )
        )

    return legend2patch


def create_legend_patch_fLoop(
    facecolor: list,
    label: list,
    edgecolor: list = None,
    hatch: list = None,
    alpha: list | None = None,
    marker: list | None = None,
) -> list:
    """
    Create a legend patch for each facecolor, label, edgecolor, and hatch combination.

    Parameters:
        facecolor (list): The list of face colors.
        label (list): The list of labels.
        edgecolor (list): The list of edge colors.
        hatch (list): The list of hatch patterns.

    Returns:
        list: The list of legend patches.
    """
    if alpha is None:
        alpha = [1.0] * len(facecolor)

    if marker is None:
        marker = [None] * len(facecolor)

    legend2patch = []
    edgecolor = [None] * len(facecolor) if edgecolor is None else edgecolor
    hatch = [None] * len(facecolor) if hatch is None else hatch

    for fc, lb, ec, ht, al, mk in zip(
        facecolor, label, edgecolor, hatch, alpha, marker
    ):
        legend2patch = create_legend_patch(
            legend2patch=legend2patch,
            facecolor=fc,
            label=lb,
            edgecolor=ec,
            hatch=ht,
            alpha=al,
            marker=mk,
        )

    return legend2patch


def create_plt_subplots(
    nrows: int = 1,
    ncols: int = 1,
    figsize: tuple = (10, 10),
    flatten: bool = False,
    plt_on: bool = False,
) -> tuple:
    """
    Create a matplotlib figure and axes with the specified figsize.

    Parameters:
        nrows (int): The number of rows in the subplot grid.
        ncols (int): The number of columns in the subplot grid.
        figsize (tuple): The size of the figure in inches.
        flatten (bool): Whether to flatten the axes.
        plt_on (bool): Whether to turn on interactive plotting.

    Returns:
        fig (matplotlib.figure.Figure): The created figure.
        ax (matplotlib.axes.Axes): The created axes.
    """

    interactive_mode(plt_on)

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    if flatten and nrows * ncols > 1:
        ax = ax.flatten()
    return fig, ax


def create_plt_GridSpec(
    nrows: int = 1,
    ncols: int = 1,
    figsize: tuple = (10, 10),
    height_ratios: list = None,
    width_ratios: list = None,
    plt_on: bool = False,
) -> tuple:
    """
    Create a matplotlib figure and axes with the specified figsize.
    """

    interactive_mode(plt_on)

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(
        nrows, ncols, height_ratios=height_ratios, width_ratios=width_ratios
    )
    return fig, gs


def add_suplot_to_figViaGridSpec(
    fig: plt.Figure,
    gs: gridspec.GridSpec,
) -> None:
    """
    Add a suplot to a figure via GridSpec.

    Parameters:
        fig (plt.Figure): The figure to add the subplot to.
        gs (gridspec.GridSpec): The GridSpec to add the subplot to.

    Returns:
        axes (list): The list of axes. The size of the list is the same as the number of subplots. Organized by row-major order, so the first element is the first row and the first column, the second element is the first row and the second column, etc.
    """
    axes = []
    for i in range(gs.nrows):
        for j in range(gs.ncols):
            axes.append(fig.add_subplot(gs[i, j]))
    return axes


def plot_imshow(
    fig: plt.Figure,
    axis: plt.Axes,
    data2plot: np.ndarray,
    suptitle: str = "",
    title: str = None,
    cmap: str = None,
    norm: mcolors.Normalize | None = None,
    xlim: tuple = None,
    ylim: tuple = None,
    xlabel: str = None,
    ylabel: str = None,
    xticks: list = None,
    yticks: list = None,
    return_im: bool = False,
    aspect: str = None,
    interpolation: str = None,
    alpha: float = None,
    vmin: float | None = None,
    vmax: float | None = None,
    origin: str = None,
    extent: tuple = None,
    **kwargs,
):
    """
    Plot an image using imshow.

    Parameters:
        fig (Figure): The figure object to plot the image on.
        axis (Axes or list): The axis object(s) to plot the image on. If a list is provided, each image in `data2plot` will be plotted on the corresponding axis.
        data2plot (array-like or list): The image data to be plotted. If a list is provided, each image will be plotted on the corresponding axis.
        suptitle (str, optional): The title of the figure. Defaults to an empty string.
        cmap (str, optional): The colormap to be used for the image. Defaults to None.
        norm (mcolors.Normalize, optional): The normalization to be applied to the image. Defaults to None.
        xlim (tuple, optional): The x-axis limits of the plot. Defaults to None.
        ylim (tuple, optional): The y-axis limits of the plot. Defaults to None.
        xlabel (str, optional): The label for the x-axis. Defaults to None.
        ylabel (str, optional): The label for the y-axis. Defaults to None.
        aspect (str, optional): The aspect ratio of the image. Defaults to None.
        interpolation (str, optional): The interpolation method to be used. Defaults to None.
        alpha (float, optional): The transparency of the image. Defaults to None.
        vmin (float, optional): The minimum value of the colormap. Defaults to None.
        vmax (float, optional): The maximum value of the colormap. Defaults to None.
        origin (str, optional): The origin of the image. Defaults to None.
        extent (tuple, optional): The extent of the image. Defaults to None.
        **kwargs: Additional keyword arguments to be passed to the `imshow` function.

    Returns:
        None
    """
    if norm is None:
        if vmin is None:
            try:
                vmin = data2plot.min()
            except (AttributeError, TypeError):
                vmin = np.min(data2plot)
        if vmax is None:
            try:
                vmax = data2plot.max()
            except (AttributeError, TypeError):
                vmax = np.max(data2plot)
        norm = Normalize(vmin=vmin, vmax=vmax)

    if isinstance(axis, np.ndarray):
        from CLAH_ImageAnalysis.utils import iter_utils

        xlim, ylim, xlabel, ylabel, xticks, yticks, title = (
            iter_utils.prepare_None_iter(
                xlim, ylim, xlabel, ylabel, xticks, yticks, title
            )
        )
        axis = axis.flatten()
        for ax, img, x_lim, y_lim, x_label, y_label, x_ticks, y_ticks, title in zip(
            axis, data2plot, xlim, ylim, xlabel, ylabel, xticks, yticks, title
        ):
            ax.imshow(
                img,
                cmap=cmap,
                norm=norm,
                aspect=aspect,
                alpha=alpha,
                # vmin=vmin,
                # vmax=vmax,
                origin=origin,
                extent=extent,
                interpolation=interpolation,
                **kwargs,
            )
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(title)
            if x_ticks is not None:
                ax.set_xticks(x_ticks)
            if y_ticks is not None:
                ax.set_yticks(y_ticks)
        if isinstance(suptitle, list):
            suptitle = create_multiline_string(suptitle)

        fig.suptitle(suptitle)
    else:
        im = axis.imshow(
            data2plot,
            cmap=cmap,
            norm=norm,
            aspect=aspect,
            alpha=alpha,
            # vmin=vmin,
            # vmax=vmax,
            origin=origin,
            extent=extent,
            interpolation=interpolation,
            **kwargs,
        )
        axis.set_xlim(xlim)
        axis.set_ylim(ylim)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        if xticks is not None:
            axis.set_xticks(xticks)
        if yticks is not None:
            axis.set_yticks(yticks)
        if isinstance(suptitle, list):
            suptitle = create_multiline_string(suptitle)

        fig.suptitle(suptitle)

        if return_im:
            return im


def set_cbar_location(
    fig: plt.Figure,
    cbar_ref,
    axes_coord: tuple = (0.05, 0.15, 0.01, 0.7),
    position: str = "left",
) -> None:
    """
    Set the location of the colorbar.

    Parameters:
        fig (Figure): The figure object to add the colorbar to.
        cbar_ref: The reference object to the colorbar.
        axes_coord (tuple, optional): The coordinates of the colorbar axes. Defaults to (0.05, 0.15, 0.01, 0.7).
        position (str, optional): The position of the colorbar. Defaults to "left".
    """

    cbar_ax = fig.add_axes(axes_coord)
    cbar = fig.colorbar(cbar_ref, cax=cbar_ax)
    cbar.ax.yaxis.set_ticks_position(position)
    cbar.ax.yaxis.set_label_position(position)


def makeAxesLocatable(
    ax: plt.Axes,
) -> object:
    """
    Make an axes locatable object.
    """
    return make_axes_locatable(ax)


def append_axes2locator(
    locator: object,
    position: str,
    size: str,
    pad: float,
    sharex: plt.Axes | None = None,
    sharey: plt.Axes | None = None,
) -> None:
    locatorWaxes = locator.append_axes(
        position=position, size=size, pad=pad, sharex=sharex, sharey=sharey
    )
    return locatorWaxes


def plot_bounding_box(
    axis: plt.Axes,
    bounding_box: list,
    edgecolor: str,
    facecolor: str = "none",
    linewidth: int = 2,
) -> None:
    """
    Plot a bounding box on the given axis.

    Parameters:
        axis (Axes): The axis object to plot the bounding box on.
        bounding_box (list): The bounding box coordinates.
        edgecolor (str): The edge color of the bounding box.
        facecolor (str, optional): The face color of the bounding box. Defaults to "none".
        linewidth (int, optional): The line width of the bounding box. Defaults to 2.
    """

    for bbox in bounding_box:
        min_row, min_col, max_row, max_col = bbox
        width = max_col - min_col
        height = max_row - min_row
        rect = patches.Rectangle(
            (min_col, min_row),
            width,
            height,
            linewidth=linewidth,
            edgecolor=edgecolor,
            facecolor=facecolor,
        )
        axis.add_patch(rect)


def plot_circle_patch(
    axis: plt.Axes,
    centroid: tuple,
    radius: float,
    edgecolor: str,
    facecolor: str = "none",
    linewidth: int = 2,
) -> None:
    """
    Plot a circle patch on the given axis.

    Parameters:
        axis (Axes): The axis object to plot the circle patch on.
        centroid (tuple): The centroid coordinates of the circle.
        radius (float): The radius of the circle.
        edgecolor (str): The edge color of the circle.
        facecolor (str, optional): The face color of the circle. Defaults to "none".
        linewidth (int, optional): The line width of the circle. Defaults to 2.
    """

    circle = patches.Circle(
        (centroid[1], centroid[0]),
        radius,
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    axis.add_patch(circle)


def plot_contour(
    axis: plt.Axes,
    contour: np.ndarray,
    edgecolor: str,
    closed: bool = True,
    fill: bool = False,
    # facecolor: str = "none",
    linewidth: int = 2.5,
) -> None:
    """
    Plot a contour on the given axis.

    Parameters:
        axis (Axes): The axis object to plot the contour on.
        contour (np.ndarray): The contour coordinates.
        edgecolor (str): The edge color of the contour.
        closed (bool, optional): Whether the contour is closed. Defaults to True.
        fill (bool, optional): Whether to fill the contour. Defaults to False.
        linewidth (int, optional): The line width of the contour. Defaults to 2.5.
    """

    # tranpose contour to (x, y) format
    contour = contour[:, [1, 0]]

    polygon = Polygon(
        contour, closed=closed, edgecolor=edgecolor, fill=fill, linewidth=linewidth
    )
    axis.add_patch(polygon)


def show_plots() -> None:
    """
    Display all matplotlib figures.
    """
    plt.show()


def close_all_figs() -> None:
    """
    Close all matplotlib figures.
    """
    plt.close("all")


def empty_tick_maker(data2plot_len: int) -> list:
    """
    Create an empty tick list for the x and y axes.

    Parameters:
        data2plot_len (int): The length of the data to be plotted.

    Returns:
        list: A list of empty strings with the same length as the data to be plotted.
    """
    return [[] for _ in range(data2plot_len)]


def plot_SEM(
    arr: np.ndarray,
    color: str,
    ax: plt.Axes,
    x_ind: list,
    baseline: slice = None,
    vline: bool = False,
    linestyle: str = None,
    label=None,
    # threshold: float = None,
) -> None:
    """
    Plot the standard error of the mean (SEM) with shaded error bars.

    Parameters:
        arr (np.ndarray): Array of data points.
        color (str): Color of the plot.
        ax (plt.Axes): Axes object to plot on.
        x_ind (list): Start and end of the x-axis.
        baseline (slice, optional): Slice object defining the baseline frame range. Default is None.
        vline (bool, optional): Whether to add a vertical line at y=0. Default is False.
        linestyle (str, optional): The line style of the plot. Default is None.
        label (str, optional): The label for the plot. Default is None.
    """
    x_ax = np.linspace(x_ind[0], x_ind[-1], arr.shape[0])
    try:
        if arr.ndim == 2:
            sem_val = sem(arr, axis=1, nan_policy="omit")
            mean_arr = np.nanmean(arr, axis=1)
        elif arr.ndim == 3:
            sem_val = sem(arr, axis=(1, 2), nan_policy="omit")
            mean_arr = np.nanmean(arr, axis=(1, 2))
        else:
            raise ValueError("Array must be 2D or 3D.")
    except ValueError as e:
        print("Error while getting Mean & SEM: ", e)

    mean_arr = (
        mean_arr - np.nanmean(mean_arr[baseline]) if baseline is not None else mean_arr
    )

    ax.fill_between(
        x_ax, mean_arr - sem_val, mean_arr + sem_val, color=color, alpha=0.2
    )
    ax.plot(
        x_ax,
        mean_arr,
        color=color,
        linestyle=linestyle if not None else "-",
        label=label if not None else None,
    )

    if vline:
        # Add a horizontal line at y=0
        ax.axvline(0, color="black", linewidth=0.25)
    return


def shaded_error_bar(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    err: np.ndarray,
    line_props: dict = None,
    patch_color: str = "blue",
    edge_color: str = "black",
    face_alpha: float = 0.5,
) -> None:
    """
    Plot a shaded error bar.

    Parameters:
        x (np.ndarray): x-coordinates of the data points.
        y (np.ndarray): y-coordinates of the data points.
        err (np.ndarray): Error values for the data points.
        line_props (dict, optional): Properties for the line plot. Default is None.
        patch_color (str, optional): Color of the shaded patch. Default is 'blue'.
        edge_color (str, optional): Color of the edges around the patch. Default is 'black'.
        face_alpha (float, optional): Alpha value for the shaded patch. Default is 0.5.
    """

    # Create a patch
    yP = np.concatenate((y - err, y[::-1] + err[::-1]))
    xP = np.concatenate((x, x[::-1]))
    plt.fill(xP, yP, color=patch_color, edgecolor="none", alpha=face_alpha)

    # Make pretty edges around the patch
    ax.plot(x, y - err, "-", color=edge_color)
    ax.plot(x, y + err, "-", color=edge_color)

    # Now replace the line
    if line_props is None:
        line_props = {}
    ax.plot(x, y, **line_props)


def plot_confusion_matrix(
    ax: plt.Axes,
    cm: np.ndarray,
    group_labels: list,
    annot: bool = True,
    cmap: str = "Blues",
    format: str = "d",
    cbar: bool = False,
) -> None:
    """
    Plot a confusion matrix.

    Parameters:
        ax (plt.Axes): The axis object to plot the confusion matrix on.
        cm (np.ndarray): The confusion matrix.
        group_labels (list): The labels for the groups.
        annot (bool, optional): Whether to annotate the confusion matrix. Default is True.
        cmap (str, optional): The colormap to use. Default is "Blues".
        format (str, optional): The format of the annotations. Default is "d".
        cbar (bool, optional): Whether to add a colorbar. Default is False.
    """

    sns.heatmap(cm, annot=annot, cmap=cmap, fmt=format, ax=ax, cbar=cbar)
    # ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.xaxis.set_ticklabels(group_labels)
    ax.yaxis.set_ticklabels(group_labels)


def violin_plot(
    ax: plt.Axes = None,
    **kwargs,
) -> None:
    """
    Create a violin plot.

    Parameters:
        ax (plt.Axes, optional): The axes on which to draw the violin plot. If not provided, a new figure and axes will be created.
        **kwargs: Additional keyword arguments to pass to the violinplot function.
                    See https://seaborn.pydata.org/generated/seaborn.violinplot.html for more details.
    """
    sns.violinplot(ax=ax, **kwargs)


def create_separate_legend(color_map: dict) -> plt.Figure:
    """
    Create a separate legend figure.

    Parameters:
        color_map (dict): A dictionary mapping legend labels to colors.

    Returns:
        fig_legend (matplotlib.figure.Figure): The separate legend figure.
    """

    fig_legend = plt.figure(figsize=(3, 3))
    legend_elements = [
        Line2D([0], [0], color=color, lw=2, label=ct) for ct, color in color_map.items()
    ]
    fig_legend.legend(handles=legend_elements, loc="center")
    return fig_legend


def bar_plot(
    ax: plt.Axes,
    X: np.ndarray,
    Y: np.ndarray,
    color: str = None,
    edgecolor: str = None,
    title: str = None,
    ylim: tuple = None,
    linestyle: str = None,
    label: str = None,
    yerr: np.ndarray = None,
    hatch: str = None,
    width: float = 0.5,
    alpha: float = 1.0,
) -> None:
    """
    Create a bar plot.

    Parameters:
        color (str, optional): The color of the bars. Default is None.
        edgecolor (str, optional): The edge color of the bars. Default is None.
        title (str, optional): The title of the plot. Default is None.
        ylim (tuple, optional): The y-axis limits. Default is None.
        linestyle (str, optional): The line style of the bars. Default is None.
        label (str, optional): The label for the bars. Default is None.
        yerr (array-like, optional): The y-error bars. Default is None.
        hatch (str, optional): The hatch pattern for the bars. Default is None.
        width (float, optional): The width of the bars. Default is 0.5.
    """

    ax.bar(
        X,
        Y,
        color=color,
        edgecolor=edgecolor,
        linestyle=linestyle,
        label=label,
        yerr=yerr,
        width=width,
        alpha=alpha,
        hatch=hatch,
    )
    ax.set_title(title) if title is not None else None
    ax.set_ylim(ylim) if ylim is not None else None


def line_plot(
    ax: plt.Axes,
    X: np.ndarray,
    Y: np.ndarray,
    color: str = None,
    label: str = None,
    linestyle: str = None,
    linewidth: float = 2.5,
) -> None:
    ax.plot(X, Y, color=color, label=label, linestyle=linestyle, linewidth=linewidth)


def scatter_plot(
    ax: plt.Axes,
    X: np.ndarray | float | int,
    Y: np.ndarray,
    color: str = None,
    edgecolor: str = None,
    title: str = None,
    ylim: tuple = None,
    marker: str = "o",
    label: str = None,
    alpha: float = 1.0,
    s: float = 50,
    jitter: float = 0.03,
) -> None:
    """
    Create a scatter plot.

    Parameters:
        ax (plt.Axes): The axes to plot on
        X (np.ndarray): X coordinates
        Y (np.ndarray): Y coordinates
        color (str, optional): Color of the markers
        edgecolor (str, optional): Edge color of the markers
        title (str, optional): Plot title
        ylim (tuple, optional): Y-axis limits
        marker (str, optional): Marker style
        label (str, optional): Label for legend
        alpha (float, optional): Transparency of markers
        s (float, optional): Size of markers
    """
    if isinstance(X, float) or isinstance(X, int):
        X = np.repeat(X, Y.shape[0])

    if jitter is not None:
        X = X + np.random.normal(0, jitter, size=X.shape)

    ax.scatter(
        X,
        Y,
        color=color,
        edgecolor=edgecolor,
        marker=marker,
        label=label,
        alpha=alpha,
        s=s,
    )
    if title is not None:
        ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(ylim)


def create_index4grouped_barplot(
    n_bars: int,
    index: np.ndarray,
    width: float,
    measureLoop_idx: int,
    num_groups: int,
    gLoop_idx: int,
    offset_scaling: int = 2,
) -> list:
    """
    Create an index array for a grouped bar plot.

    Parameters:
        n_bars (int): The number of bars.
        index (np.ndarray): The index array.
        width (float): The width of the bars.
        measureLoop_idx (int): The measure loop index.
        num_groups (int): The number of groups.
        gLoop_idx (int): The group loop index.
        offset_scaling (int, optional): The offset scaling factor. Default is 2.

    Returns:
        index4plot (list): The index list for the grouped bar plot.
    """

    offset = (width * n_bars) / (offset_scaling * num_groups)
    index4plot = (
        index
        - offset
        + (width * measureLoop_idx * num_groups)
        + (gLoop_idx - (num_groups - 1) / 2) * width
    )

    if num_groups == 1:
        index4plot = [idx + (width / 2) for idx in index4plot]

    return index4plot


def add_text_box(
    ax: plt.Axes,
    text: str,
    xpos: float = 0.5,
    ypos: float = 0.95,
    transform: Any = None,
    color: str = None,
    ha: str = None,
    va: str = None,
    rotation: float = None,
    fontsize: int = 14,
    bbox: dict | None = None,
) -> None:
    """
    Add a text box to the plot.

    Parameters:
        ax (plt.Axes): The axis object to add the text box to.
        text (str): The text to add to the plot.
        xpos (float, optional): The x-position of the text box. Default is 0.5.
        ypos (float, optional): The y-position of the text box. Default is 0.95.
        transform (plt.transform, optional): The transform to use for the text box. Default is None.
        color (str, optional): The color of the text. Default is None.
        ha (str, optional): The horizontal alignment of the text. Default is None.
        va (str, optional): The vertical alignment of the text. Default is None.
        rotation (float, optional): The rotation of the text. Default is None.
        fontsize (int, optional): The font size of the text. Default is 14.
        bbox (dict, optional): The bounding box properties. Default is None.
    """

    kwargs = {
        "x": xpos,
        "y": ypos,
        "s": text,
        "fontsize": fontsize,
    }

    if bbox is None:
        bbox = dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white")
    if transform is None:
        transform = ax.transAxes

    # Optional arguments
    if transform is not None:
        kwargs["transform"] = transform
    if color is not None:
        kwargs["color"] = color
    if rotation is not None:
        kwargs["rotation"] = rotation
    if bbox is not None:
        kwargs["bbox"] = bbox
    if ha is not None:
        kwargs["ha"] = ha
    if va is not None:
        kwargs["va"] = va

    ax.text(**kwargs)


def save_plotly(
    plotly_fig: go.Figure,
    fig_name: str,
    figure_save_path: str = None,
    auto_open: bool = False,
) -> None:
    """
    Save a Plotly figure as an HTML file.

    Parameters:
        plotly_fig (plotly.graph_objects.Figure): The Plotly figure to save.
        figure_save_path (str): The path to save the figure to.
        fig_name (str): The name of the figure.
        auto_open (bool, optional): Whether to open the figure after saving. Default is False.
    """

    if figure_save_path is not None:
        if not os.path.exists(figure_save_path):
            os.makedirs(figure_save_path)

    if not fig_name.endswith(".html"):
        fig_name += ".html"

    # Construct the full file path
    if figure_save_path is None:
        file_path = fig_name
    else:
        file_path = os.path.join(figure_save_path, fig_name)

    # Save the Plotly figure as an HTML file
    plotly_fig.write_html(file_path, auto_open=auto_open)


def create_plotly() -> go.Figure:
    """
    Create a Plotly figure.

    Returns:
        fig (plotly.graph_objects.Figure): The Plotly figure.
    """
    return go.Figure()


def create_plotly_subplots(
    rows: int,
    cols: int,
    shared_xaxes: bool = False,
    shared_yaxes: bool = False,
    vertical_spacing: float = 0.02,
) -> go.Figure:
    """
    Create a Plotly figure with subplots.

    Parameters:
        rows (int): Number of rows in the subplot grid.
        cols (int): Number of columns in the subplot grid.
        shared_xaxes (bool, optional): Whether to share x-axes between subplots. Default is False.
        shared_yaxes (bool, optional): Whether to share y-axes between subplots. Default is False.
        vertical_spacing (float, optional): Vertical spacing between subplots. Default is 0.02.

    Returns:
        plotly.graph_objects.Figure: A Plotly figure with the specified subplot layout.
    """

    return make_subplots(
        rows=rows,
        cols=cols,
        shared_xaxes=shared_xaxes,
        shared_yaxes=shared_yaxes,
        vertical_spacing=vertical_spacing,
    )


def add_plotly_trace(
    fig: go.Figure,
    x: np.ndarray = None,
    y: np.ndarray = None,
    mode: str = None,
    name: str = None,
    row: int = None,
    col: int = None,
    **kwargs,
) -> None:
    """
    Add a trace to a Plotly figure.

    Parameters:
        fig (plotly.graph_objects.Figure): The Plotly figure to add the trace to.
        x (np.ndarray, optional): The x-coordinates of the trace. Default is None.
        y (np.ndarray, optional): The y-coordinates of the trace. Default is None.
        mode (str, optional): The mode of the trace. Default is None.
        name (str, optional): The name of the trace. Default is None.
        row (int, optional): Row number for subplot. Default is None.
        col (int, optional): Column number for subplot. Default is None.
        **kwargs: Additional keyword arguments for the trace.
                    See https://plotly.com/python-api-reference/generated/plotly.graph_objects.Scatter.html for more details.
    """
    fig.add_trace(
        go.Scatter(x=x, y=y, mode=mode, name=name, **kwargs),
        row=row,
        col=col,
    )


def label_cellNum_overDSImage(
    axis: plt.Axes,
    data: np.ndarray,
    cell_str: str,
    color: str = "white",
    fontsize: int = 9,
) -> None:
    """
    Label a cell in a heatmap.

    Parameters:
        axis (plt.Axes): The axis object to add the text box to.
        data (np.ndarray): The data array.
        cell_str (str): The cell string.
        color (str, optional): The color of the text. Default is "white".
        fontsize (int, optional): The font size of the text. Default is 9.
    """

    # get coords for cell to label each cell
    max_coords = np.unravel_index(np.argmax(data, axis=None), data.shape)
    cell_num_int = int(cell_str.split("_")[-1])
    axis.text(
        max_coords[0],
        max_coords[1],
        str(cell_num_int),
        color=color,
        fontsize=fontsize,
        ha="center",
        va="center",
    )


def create_legend_wNO_duplicates(axis: plt.Axes) -> None:
    """
    Create a legend with no duplicates.

    Parameters:
        axis (plt.Axes): The axis object to create the legend on.
    """

    handles, labels = axis.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    axis.legend(unique_labels.values(), unique_labels.keys())


def create_sig_2samp_annotate(
    ax: plt.Axes,
    arr0: np.ndarray,
    arr1: np.ndarray,
    coords: tuple,
    twoSamp: bool = True,
    paired: bool = False,
    parametric: bool = True,
    xytext: tuple = (0, 0),
    textcoords: str = "offset points",
    ha: str = "center",
    fontsize: int = 14,
    fontweight: str = "bold",
    color: str = "black",
    return_Pval: bool = False,
) -> Optional[float]:
    """

    Create a string annotation for significance testing between two samples and add it to a plot.

    Parameters:
        ax (plt.Axes): The matplotlib Axes object to annotate.
        arr0 (np.ndarray): The first array of data points.
        arr2 (np.ndarray): The second array of data points.
        coords (tuple): The (x, y) coordinates to place the annotation.
        parametric (bool, optional): If True, use a parametric test (t-test). If False, use a non-parametric test (Mann-Whitney U test). Default is True.
        xytext (tuple, optional): The offset (in points) from the coords to place the text. Default is (0, 0).
        textcoords (str, optional): The coordinate system that xytext is given in. Default is "offset points".
        ha (str, optional): The horizontal alignment of the text. Default is "center".
        fontsize (int, optional): The font size of the annotation text. Default is 14.
        fontweight (str, optional): The font weight of the annotation text. Default is "bold".
        color (str, optional): The color of the annotation text. Default is "black".
        return_Pval (bool, optional): If True, return the p-value. Default is False.
    """
    if paired:
        twoSamp = False

    if parametric:
        if twoSamp:
            from scipy.stats import ttest_ind

            _, pVal = ttest_ind(arr0, arr1, nan_policy="omit")
        elif paired:
            from scipy.stats import ttest_rel

            _, pVal = ttest_rel(arr0, arr1, nan_policy="omit")

    else:
        from scipy.stats import mannwhitneyu

        _, pVal = mannwhitneyu(arr0, arr1, nan_policy="omit")

    if coords[-1] is None:
        # coords is usually (x, y)
        # if y is None, then find the average of the two arrays and add half of standard deviation for marker's y position
        meanArr0 = np.nanmean(arr0)
        meanArr1 = np.nanmean(arr1)
        avgMean = np.mean([meanArr0, meanArr1])

        std_arr0 = np.nanstd(arr0)
        std_arr1 = np.nanstd(arr1)
        avgStd = np.mean([std_arr0, std_arr1])
        coords2use = (coords[0], avgMean + (avgStd / 2))
    else:
        coords2use = coords

    if pVal < 0.05:
        # mkr_sig = f"*\np={pVal:.3f}"
        mkr_sig = "*"
        if pVal < 0.01:
            mkr_sig = "**"
            if pVal < 0.001:
                mkr_sig = "***"
        ax.annotate(
            mkr_sig,
            coords2use,
            textcoords=textcoords,
            xytext=xytext,
            ha=ha,
            fontsize=fontsize,
            fontweight=fontweight,
            color=color,
        )
    if return_Pval:
        return pVal


def delete_axes(fig: plt.Figure) -> None:
    """
    Delete axes from a figure.

    Parameters:
        fig (plt.Figure): The figure to delete the axes from.
    """

    for ax in fig.get_axes():
        if not ax.get_children():
            ax.remove()


def format_axes4UMAP(
    ax: plt.Axes, fontsize: int = 14, fontweight: str = "bold"
) -> None:
    """
    Format axis labels and remove spines for UMAP plots.

    Parameters:
        ax (plt.Axes): The axis to format.
        fontsize (int, optional): The font size of the axis labels. Default is 14.
        fontweight (str, optional): The font weight of the axis labels. Default is "bold".
    """

    # name labels
    ax.set_xlabel(
        "UMAP 1",
        fontsize=fontsize,
        fontweight=fontweight,
    )
    ax.set_ylabel(
        "UMAP 2",
        fontsize=fontsize,
        fontweight=fontweight,
    )

    # remove spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Remove tick labels (numbers) and tick marks from axes
    ax.tick_params(axis="x", labelbottom=False, bottom=False)
    ax.tick_params(axis="y", labelleft=False, left=False)

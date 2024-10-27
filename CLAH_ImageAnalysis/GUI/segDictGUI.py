import tkinter as tk
from textwrap import dedent

from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from typing import Any, Optional

from CLAH_ImageAnalysis.GUI import BaseGUI
from CLAH_ImageAnalysis.unitAnalysis import UA_enum
from CLAH_ImageAnalysis.utils import color_dict
from CLAH_ImageAnalysis.utils import enum2dict
from CLAH_ImageAnalysis.utils import findLatest
from CLAH_ImageAnalysis.utils import image_utils
from CLAH_ImageAnalysis.utils import load_file
from CLAH_ImageAnalysis.utils import text_dict

text_lib = text_dict()
file_tag = text_lib["file_tag"]
dict_name = text_lib["dict_name"]
FR, EFR = text_lib["frames"]["FR"], text_lib["frames"]["EFR"]
GUICLASS = text_lib["GUICLASSUTILS"]
GUI_ELE = text_lib["GUI_ELEMENTS"]
color_dict = color_dict()
cSS_str = enum2dict(UA_enum.CSS)
done_gui = text_lib["completion"]["GUI"]


DS_image = (
    file_tag["AVGCA"] + file_tag["TEMPFILT"] + file_tag["DOWNSAMPLE"] + file_tag["IMG"]
)


class segDictGUI(BaseGUI):
    """
    A class representing the graphical user interface for segDict.

    Attributes:
        segDict (dict): A dictionary containing segmentation data.
        cueShiftTuning (dict): A dictionary containing cue shift tuning data.
        pksDict (dict): A dictionary containing peak data.
        fileID (str): The ID of the loaded file.
        ax_list (list): A list of axes objects for the plots.
        Frame_List (list): A list of frame objects for resizing.
        canvas_list (list): A list of canvas objects for the plots.
        figsize_ASP (tuple): The figure size for the ASP plot.
        figsize_CTP (tuple): The figure size for the CTP plot.
        figsize_CTP_trc (tuple): The figure size for the CTP trace plot.
        CTP_pad (int): The padding for the CTP plot.
        CTP_VAR (tk.StringVar): The variable for the CTP entry.
        ASP_VAR (tk.StringVar): The variable for the ASP entry.
        entryspan_ASP (int): The number of columns used for the ASP entry.
        figspan_ASP (int): The number of columns used for the ASP plot.
        figspan_CTP (int): The number of columns used for the CTP plot.
        figspan_ACTP (int): The number of columns used for the CTP trace plot.
    """

    def __init__(self, enable_console: bool = False) -> None:
        print(GUI_ELE["GUI_TITLE_START"].format(GUI_ELE["SD_TITLE"]))

        # init BaseGUI with 1400x1400+100+100 geometry
        super().__init__(
            enable_console=enable_console,
            tot_column_used=15,
            x=1400,
            y=1400,
            x_offset=100,
            y_offset=100,
        )

        # set GUI title
        self.set_GUI_title(GUI_ELE["SD_TITLE"])

        # init global vars for specific GUI
        print(f"{FR}{GUICLASS['BU_GLOBALVARS']}")
        self.init_global_vars_SD()

        # init widgets
        print(f"{FR}{GUICLASS['BU_INITWIDGETS']}")
        self.init_widgets()

        # display menubar
        print(f"{FR}{GUICLASS['BU_MENUBAR']}")
        self.display_menu_bar()

        # TODO fix scale
        # TODO plot ASPAT image with overlay & CTP plot w/pks

        # Enable window resizing
        self.root.bind("<Configure>", self.on_resize())
        # run GUI
        self.run_GUI()

    ######################################################
    #  GUI setup funcs
    ######################################################

    def init_global_vars_SD(self) -> None:
        """
        Initializes global variables for the SD class.

        This method initializes the following global variables:
        - segDict: A dictionary for segmentation.
        - cueShiftTuning: A dictionary for cue shift tuning.
        - pksDict: A dictionary for peaks.
        - fileID: The file ID.
        - ax_list: A list of axes.
        - Frame_List: A list of frames.
        - canvas_list: A list of canvases.
        - figsize_ASP: The figure size for ASP.
        - figsize_CTP: The figure size for CTP.
        - figsize_CTP_trc: The figure size for CTP trace.
        - CTP_pad: The padding for CTP.
        - CTP_VAR: A string variable for CTP.
        - ASP_VAR: A string variable for ASP.
        - entryspan_ASP: The span for ASP entry.
        - figspan_ASP: The span for ASP figure.
        - figspan_CTP: The span for CTP figure.
        - figspan_ACTP: The span for ACTP figure.
        """
        self.segDict = {}
        self.cueShiftTuning = {}
        self.pksDict = {}
        self.fileID = None
        self.ax_list = []
        self.Frame_List = []
        self.canvas_list = []
        self.figsize_ASP = (3, 3)
        self.figsize_CTP = (7, 2)
        self.figsize_CTP_trc = (2, 2)
        self.CTP_pad = 100
        self.CTP_VAR = tk.StringVar()
        self.ASP_VAR = tk.StringVar()
        self.CTP_VAR.set("0")
        self.ASP_VAR.set("0")

        self.entryspan_ASP = int(self.tot_column_used // (4))
        self.figspan_ASP = int(self.tot_column_used // (4 / 3))
        self.figspan_CTP = int(self.tot_column_used // (3 / 2))
        self.figspan_ACTP = int(self.tot_column_used // (3))

    def init_widgets(self) -> None:
        """
        Initializes the widgets for the GUI.

        This method creates and configures the figure frames, entry frames,
        plots, buttons, and menu bar. It also calls the `on_resize` method
        to handle resizing of the GUI.

        Parameters:
            None

        Returns:
            None
        """
        self._create_fig_frames()
        self._create_entry_frames()
        self._create_plts()
        self._create_buttons()
        self._fill_in_menu_bar()
        # self._format_post_load_entries()
        self.on_resize()

    def _create_buttons(self) -> None:
        """
        Creates buttons for the GUI and sets the reset command.

        Parameters:
            None

        Returns:
            None
        """
        self.create_reset_button(row=4, reset_command=self.reset_gui)

    def _fill_in_menu_bar(self) -> None:
        """
        Fills out the menu bar with File, Edit, and View menus.

        - File menu: Contains options for loading a file, saving a figure, and quitting the application.
        - Edit menu: Contains an option for resetting the GUI.
        - View menu: Contains an option for toggling the console.

        Returns:
            None
        """
        # Filling out menu bar
        # Add File menu
        file_commands = {
            GUI_ELE["LOAD"]: self.load_file4GUI,
            GUI_ELE["SAV_FIG"]: [],
            "separator": None,
            "Quit": self.quit_app,
        }
        self.MenuBarUtils.create_menu(GUI_ELE["MMENU"], file_commands)
        self.MenuBarUtils.update_menu_item_state(
            GUI_ELE["MMENU"], GUI_ELE["SAV_FIG"], tk.DISABLED
        )

        # Add Edit menu
        edit_commands = {GUI_ELE["RESET"]: self.reset_gui}
        self.MenuBarUtils.create_menu(GUI_ELE["EDIT"], edit_commands)

        if self.enable_console:
            # Add View menu
            view_commands = {
                GUI_ELE["CONSOLE"]: self.toggle_console,
            }
            self.MenuBarUtils.create_menu(GUI_ELE["VIEW"], view_commands)

    def _create_fig_frames(self) -> None:
        """
        Creates frames for figures.

        This method creates frames for the ASP image, CTP plot, and CTP smaller plot (peak trace plot).
        These frames are used to display the figures in the GUI.

        Returns:
            None
        """
        # Create frames for figs
        # Frame for ASP image
        figframe_ASpat_config = self.config_maker(
            row=1, column=0, columnspan=self.figspan_ASP, sticky="nsew"
        )
        self.WidgetUtils.create_tk_widget(
            widget_type="FRAME", name="ASPAT_FIG", config=figframe_ASpat_config
        )

        # frame for CTP plot
        figframe_CTP_config = self.config_maker(
            row=2, column=0, columnspan=self.figspan_CTP, sticky="nsew"
        )
        self.WidgetUtils.create_tk_widget(
            widget_type="FRAME",
            name="CTP_FIG",
            config=figframe_CTP_config,
        )

        # frame for CTP smaller plot (peak trace plot)
        figframe_CTP_peaktrace_config = self.config_maker(
            row=2, column=self.figspan_CTP, columnspan=self.figspan_ACTP, sticky="nsew"
        )
        self.WidgetUtils.create_tk_widget(
            widget_type="FRAME",
            name="CTP_FIG_TRACE",
            config=figframe_CTP_peaktrace_config,
        )

    def _create_entry_frames(self) -> None:
        """
        Creates entry frames for ASP and CTP in the GUI.

        This method creates two entry frames: one for ASP (ASpat) and one for CTP.
        It configures the frames and stores them in the respective variables for later use.
        The frames are also stored in a list for easy access during resizing.

        Returns:
            None
        """
        # frame entry for ASP entry
        entryframe_ASpat_config = self.config_maker(
            row=1, column=self.figspan_ASP, rowspan=1, sticky="nsew"
        )
        self.WidgetUtils.create_tk_widget(
            widget_type="FRAME", name="ASPAT_ENTRY", config=entryframe_ASpat_config
        )
        # frame for CTP entry
        entryframe_CTP_config = self.config_maker(
            row=3, column=0, columnspan=self.tot_column_used, sticky="nsew"
        )
        self.WidgetUtils.create_tk_widget(
            widget_type="FRAME", name="CTP_ENTRY", config=entryframe_CTP_config
        )

        # store into dicts for later use
        self.ASP_FIG_FRAME = self.WidgetUtils.frames["ASPAT_FIG"]
        self.CTP_FIG_FRAME = self.WidgetUtils.frames["CTP_FIG"]
        self.CTP_FIG_TRACE_FRAME = self.WidgetUtils.frames["CTP_FIG_TRACE"]
        self.ASP_ENTRY = self.WidgetUtils.frames["ASPAT_ENTRY"]
        self.CTP_ENTRY = self.WidgetUtils.frames["CTP_ENTRY"]
        # store frames into list for easy access later for resizing
        self.Frame_List = [
            self.ASP_FIG_FRAME,
            self.CTP_FIG_FRAME,
            self.CTP_FIG_TRACE_FRAME,
            self.ASP_ENTRY,
            self.CTP_ENTRY,
        ]

    def _create_plts(self) -> None:
        """
        Create plots and canvases for the GUI.

        This method creates three plots and their corresponding canvases:
        - A Spatial plot (ASP)
        - C Temporal plot (CTP)
        - C Temporal pk trace (CTP_trc)

        The plots are created using the `create_canvas_plt` method and are stored in instance variables.
        The canvases are also stored in instance variables for easy access later.

        Additionally, a toolbar is created for the C Temporal plot (CTP) using the `NavigationToolbar2Tk` class.

        Parameters:
            None

        Returns:
            None
        """
        # create plts/canvas
        # A Spatial plot
        self.fig_ASP, self.ax_ASP, self.canvas_ASP = self.create_canvas_plt(
            master=self.WidgetUtils.frames["ASPAT_FIG"],
            figsize=self.figsize_ASP,
            row=0,
            column=0,
            columnspan=self.figspan_ASP,
        )
        self.ax_ASP.set_facecolor(color_dict["black"])

        # C Temporal plot
        self.fig_CTP, self.ax_CTP, self.canvas_CTP = self.create_canvas_plt(
            master=self.WidgetUtils.frames["CTP_FIG"],
            figsize=self.figsize_CTP,
            row=1,
            column=0,
            columnspan=self.figspan_CTP,
        )

        # C Temporal pk trace
        self.fig_CTP_trc, self.ax_CTP_trc, self.canvas_CTP_trc = self.create_canvas_plt(
            master=self.WidgetUtils.frames["CTP_FIG_TRACE"],
            figsize=self.figsize_CTP_trc,
            row=0,
            column=0,
            columnspan=self.figspan_ACTP,
        )

        # create toolbar for CTP
        toolbar_config = self.config_maker(
            row=0, column=0, columnspan=self.figspan_CTP, sticky="nsew"
        )
        self.WidgetUtils.create_tk_widget(
            widget_type="FRAME",
            name="CTP_TOOLBAR",
            config=toolbar_config,
            diff_parent=self.CTP_FIG_FRAME,
        )
        self.toolbar_frame = self.WidgetUtils.frames["CTP_TOOLBAR"]
        self.toolbar_CTP = NavigationToolbar2Tk(self.canvas_CTP, self.toolbar_frame)
        self.toolbar_CTP.update()

        # store into lists for easy access later
        self.ax_list = [self.ax_ASP, self.ax_CTP, self.ax_CTP_trc]
        self.canvas_list = [self.canvas_ASP, self.canvas_CTP, self.canvas_CTP_trc]

        # draw canvas
        for can in self.canvas_list:
            can.draw()

    def on_resize(self) -> None:
        """
        This method is called when the window is resized.
        It updates the root, resizes the root and frames, and redraws the canvases.
        """
        self.root.update()
        # resize root
        self.config_rowsNcols2resize(columns=True, rows=True)
        # resize frames
        for frame in self.Frame_List:
            self.config_rowsNcols2resize(columns=True, rows=True, parent=frame)
        self.canvas_CTP.draw()
        self.canvas_ASP.draw()
        self.canvas_CTP_trc.draw()
        self.root.update()

    ######################################################
    #  Load file funcs
    ######################################################

    def load_file4GUI(self) -> None:
        """
        Loads a file for the GUI.

        If a file is already loaded, it resets the GUI.
        It then performs various update operations and prints messages.
        """
        if self.fileID:
            self.reset_gui()
        self._CSS_load_utils()
        self._update_load_label_post_load()
        self._update_ASP_wDSimage_at_Start()
        self._format_post_load_entries()
        self.print_post_loadORreset_msg()
        self.update_ASpat_image_wDS_Entry()
        self.update_CTP_plot(reset=True)

    def _CSS_load_utils(self) -> None:
        """
        Load utilities for CSS (Cue Shift Structure).

        This method loads the necessary utilities for CSS, including the pkl file name,
        file ID, date, session, cueShiftStruc, segDict, and pksDict. It also removes the
        cueShiftStruc after extracting the required information.

        Returns:
            None
        """
        (
            self.pkl_fname,
            self.fileID,
            self.date,
            self.session,
        ) = self.get_pkl_fnameNfileID(
            dict_name["CSS"], file_tag["PKL"], date_first=True
        )
        print("cueShiftStruc selected...", end="", flush=True)
        self.cueShiftStruc = load_file(self.pkl_fname, file_tag["PKL"])
        print("extracting segDict...", end="", flush=True)
        self.segDict = load_file(
            self.cueShiftStruc["segDict"].replace(file_tag["MAT"], file_tag["PKL"]),
            file_tag["PKL"],
        )
        print("filling in pksDict...", end="", flush=True)
        self.pksDict = self.cueShiftStruc[cSS_str["PKS"]]
        # remove cueShiftStruc
        del self.cueShiftStruc

    def _update_load_label_post_load(self) -> None:
        """
        Updates the load label widget after loading a file.

        This method generates a configuration for the load label widget
        based on the current file ID and updates the widget with the new
        configuration.

        Parameters:
            self: The instance of the class.

        Returns:
            None
        """
        load_label_wfileID_config = self.config_maker(
            text=GUI_ELE["LOAD_ID"].format(self.fileID)
        )
        self.WidgetUtils.tk_update(
            name=GUICLASS["BU_LOAD"], config=load_label_wfileID_config
        )

    def _format_post_load_entries(self) -> None:
        """
        Formats and sets up the post-load entries in the GUI.

        This method creates and configures various widgets such as labels, entry fields, and scales
        for the post-load entries in the GUI. It sets up the available range label, ASPAT entry,
        CTP scale, and CTP entry. It also performs necessary bindings and validations for the widgets.

        Parameters:
            None

        Returns:
            None
        """
        max_val = self.segDict["A_Spatial"].shape[1]
        range_label_config = self.config_maker(
            text=f" Available Range: 0 - {max_val} ",
            row=0,
            column=0,
            sticky="nsew",
            in_=self.ASP_ENTRY,
        )
        self.WidgetUtils.create_tk_widget(
            widget_type="LABEL",
            name=GUICLASS["BU_RANGE"],
            config=range_label_config,
            diff_parent=self.WidgetUtils.frames["ASPAT_ENTRY"],
        )

        # set up ASPAT entry
        entry_ASPAT_config = self.config_maker(
            textvariable=self.ASP_VAR,
            row=1,
            column=0,
            sticky="nsew",
            in_=self.ASP_ENTRY,
            bindings={"<Return>": self.update_ASpat_image_wDS_Entry(reset=True)},
            justify="right",
        )
        self.WidgetUtils.create_tk_widget(
            widget_type="ENTRY",
            name="ENTRY_ASPAT",
            config=entry_ASPAT_config,
            diff_parent=self.WidgetUtils.frames["ASPAT_ENTRY"],
        )
        self.config_rowsNcols2resize(columns=True, parent=self.ASP_ENTRY)

        # set up CTP scale
        self.CTP_ENTRY_columnspan = self.CTP_ENTRY.grid_info()["columnspan"]
        scale_CTP_config = self.config_maker(
            from_=0,
            to=max_val,
            orient="horizontal",
            variable=self.CTP_VAR,
            command=self.on_slider_move,
            row=0,
            column=0,
            columnspan=1,
            padx=self.CTP_pad,
            sticky="nsew",
            in_=self.CTP_ENTRY,
        )
        self.WidgetUtils.create_tk_widget(
            widget_type="SCALE",
            name="CTP_SCALE",
            config=scale_CTP_config,
            diff_parent=self.CTP_ENTRY,
        )
        self.SCALE = self.WidgetUtils.scales["CTP_SCALE"]

        # set up CTP entry
        self.CTP_VAR.trace("w", self.on_var_change)
        vcmd = (self.root.register(lambda P: self.validate_entry(P, max_val)), "%P")
        entry_CTP_config = self.config_maker(
            textvariable=self.CTP_VAR,
            validate="key",
            validatecommand=vcmd,
            row=1,
            column=0,
            columnspan=1,
            padx=self.CTP_pad,
            sticky="nsew",
            in_=self.CTP_ENTRY,
            justify="center",
        )
        self.WidgetUtils.create_tk_widget(
            widget_type="ENTRY",
            name="ENTRY_CTP",
            config=entry_CTP_config,
            diff_parent=self.CTP_ENTRY,
        )
        self.config_rowsNcols2resize(columns=True, parent=self.CTP_ENTRY)

    ######################################################
    # Reset GUI
    ######################################################

    def reset_gui(self) -> None:
        """
        Resets the GUI by performing the following actions:
        1. Initializes global variables.
        2. Clears all axes in the ax_list.
        3. Sets the facecolor of ax_ASP to black.
        4. Redraws all canvases in the canvas_list.
        5. Destroys specific entries in the Frame_List.
        6. Rebuilds entry frames.
        7. Prints a message after resetting the GUI.
        """
        print("Resetting GUI...", end="", flush=True)
        self.init_global_vars_SD()

        for axes in self.ax_list:
            axes.clear()

        self.ax_ASP.set_facecolor(color_dict["black"])

        for can in self.canvas_list:
            can.draw()

        for entry in self.Frame_List:
            if entry in [self.ASP_ENTRY, self.CTP_ENTRY] and entry is not None:
                entry.destroy()
        # after destroying must rebuild entry frames
        self._create_entry_frames()

        self.print_post_loadORreset_msg()

    ######################################################
    #  Plotting funcs
    ######################################################

    def _update_ASP_wDSimage_at_Start(self) -> None:
        """
        Updates the ASP (A Spatial Projection) with the latest DS (Downsampled) image.

        This method clears the ASP plot, reads the latest DS image file, and displays it on the ASP plot.

        Parameters:
            None

        Returns:
            None
        """
        self.ax_ASP.clear()
        image_file = findLatest(DS_image)
        img = image_utils.read_image(image_file)
        self.ax_ASP.set_title(
            GUI_ELE["PLT_TITLE_CSS"].format(self.date, self.session), fontsize=7
        )
        self.ax_ASP.imshow(img, cmap="gray", aspect="equal")
        self.canvas_ASP.draw()

    def update_ASpat_image_wDS_Entry(
        self, event: Optional[Any] = None, reset: bool = False
    ) -> None:
        """
        Update the A_Spatial image with the selected cells.

        Parameters:
            event (Any, optional): The event that triggered the update. Defaults to None.
            reset (bool, optional): Whether to reset the A_Spatial image. Defaults to False.

        Returns:
        None
        """
        self.print_flush_statements(UPDATE_ASPAT=True)
        if reset:
            self._update_ASP_wDSimage_at_Start()
        A_Spatial = self.segDict["A_Spatial"]
        selected_cells = self.get_selected_cells(self.ASP_VAR.get(), A_Spatial.shape[1])
        for cell in selected_cells:
            self.plot_select_cell_ASpat(
                A_Spatial=A_Spatial, cell_num=cell, ax=self.ax_ASP, cmap=self.cmaps[0]
            )
        self.canvas_ASP.draw()
        print(done_gui)
        print("Plotting Cell #s:")
        self.print_cell_num_plotted(selected_cells=selected_cells)
        print()
        if reset:
            self.update_CTP_plot()

    def update_CTP_plot(self, event: Optional[Any] = None, reset: bool = False) -> None:
        """
        Update the CTP plot based on the selected cell.

        Parameters:
            event (optional): The event that triggered the update. Defaults to None.
            reset (bool, optional): Whether to reset the plot. Defaults to False.
        """
        self.print_flush_statements(UPDATE_CTP=True)
        if reset:
            self._update_ASP_wDSimage_at_Start()

        # Clear CTP ax & pk trace
        self.ax_CTP.clear()
        self.ax_CTP_trc.clear()

        # Initialize matrices to plot
        A_Spatial = self.segDict["A_Spatial"]
        C_Temporal = self.segDict["C_Temporal"]
        selected_cell = int(self.CTP_VAR.get())

        # Plot selected cell in A_Spatial
        self.plot_select_cell_ASpat(
            A_Spatial=A_Spatial,
            cell_num=selected_cell,
            ax=self.ax_ASP,
            cmap=self.cmaps[1],
            alpha=0.8,
        )

        # Plot selected cell in CTP
        self.plot_select_cell_CTP(
            CTP=C_Temporal,
            pks=self.pksDict,
            cell_num=selected_cell,
            ax_CTP=self.ax_CTP,
            ax_pk_trace=self.ax_CTP_trc,
            color_from_cmap=self.cmaps[1](0.5),
        )

        # Redraw all canvas
        for can in self.canvas_list:
            can.draw()

        print(done_gui)
        print("Plotting Cell #s:")

        # For print, need to pass selected_cell as a list
        self.print_cell_num_plotted(selected_cells=[selected_cell])
        print()

        if reset:
            # Because it clears the image
            self.update_ASpat_image_wDS_Entry()

    def on_slider_move(self, value: Any) -> None:
        """
        Update the StringVar when the slider is moved and update the CTP plot.

        Parameters:
        - value: The new value of the slider.

        Returns:
        None
        """
        self.CTP_VAR.set(value)
        self.update_CTP_plot(reset=True)

    def on_var_change(self, *args: Any) -> None:
        """
        Callback function triggered when the variable associated with the Entry widget changes.

        This function updates the Scale widget to reflect the Entry's value and calls the
        update_CTP_plot method with the reset parameter set to True.

        Parameters:
        *args: Variable-length argument list.

        Returns:
        None
        """
        # Update the Scale widget to reflect the Entry's value
        value = self.CTP_VAR.get()
        if value.isdigit():
            self.SCALE.set(int(value))
            self.update_CTP_plot(reset=True)

    def print_post_loadORreset_msg(self) -> None:
        """
        Prints a message indicating the status of the loaded ID and the entry selection for cells.

        If a file ID is present, it indicates that the file was loaded and the status is set to "created".
        If no file ID is present, it indicates that the file was removed and the status is set to "removed".
        """
        status = "created" if self.fileID else "removed"
        print(
            dedent(
                f"""\
    {done_gui}
    {FR}Loaded ID: {self.fileID}
    {FR}Entry selection for cells was {status}
    """
            )
        )


######################################################
#  run function when script is called directly
######################################################
if __name__ == "__main__":
    segDictGUI(enable_console=True)

import os
import tkinter as tk
from textwrap import dedent

import numpy as np
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from typing import Any, Optional

from CLAH_ImageAnalysis.GUI import BaseGUI
from CLAH_ImageAnalysis.unitAnalysis import UA_enum
from CLAH_ImageAnalysis.unitAnalysis import pks_utils


class segDictGUI(BaseGUI):
    """
    A class representing the graphical user interface for segDict.

    Attributes:
        segDict (dict): A dictionary containing segmentation data.
        cueShiftTuning (dict): A dictionary containing cue shift tuning data.
        pksDict (dict): A dictionary containing peak data.
        subjID (str): The ID of the loaded file.
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
        # init BaseGUI with 1400x1400+100+100 geometry
        super().__init__(
            enable_console=enable_console,
            tot_column_used=15,
            x=1400,
            y=1400,
            x_offset=100,
            y_offset=100,
        )

        print(self.GUI_ELE["GUI_TITLE_START"].format(self.GUI_ELE["SD_TITLE"]))

        # set GUI title
        self.set_GUI_title(self.GUI_ELE["SD_TITLE"])

        # init global vars for specific GUI
        self.print_wFrm(f"{self.GUICLASS['BU_GLOBALVARS']}")
        self.init_global_vars_SD()

        # init widgets
        self.print_wFrm(f"{self.GUICLASS['BU_INITWIDGETS']}")
        self.init_widgets()

        # display menubar
        self.print_wFrm(f"{self.GUICLASS['BU_MENUBAR']}")
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
        - subjID: The file ID.
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
        self.subjID = None
        self.ax_list = []
        self.Frame_List = []
        self.LB_list = []
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
        self.figspan_ASP = int(self.tot_column_used // 2)
        self.figspan_CTP = int(self.tot_column_used // (3 / 2))
        self.figspan_ACTP = int(self.tot_column_used // (3))
        self.cspan4list_ratio = 2 / 3
        self.cspan4AEntry_ratio = 1 - self.cspan4list_ratio

        self.PKSkey = self.enum_utils.enum2dict(UA_enum.PKS)
        # self.cSS_str = self.enum_utils.enum2dict(UA_enum.CSS)

        self.fps = UA_enum.Parser4QT.ARG_DICT.value[("fps", "f")]["DEFAULT"]
        self.sdThresh = UA_enum.Parser4QT.ARG_DICT.value[("sdThresh", "sdt")]["DEFAULT"]
        self.timeout = UA_enum.Parser4QT.ARG_DICT.value[("timeout", "to")]["DEFAULT"]

        self.PKSUtils = pks_utils(
            fps=self.fps,
            sdThresh=self.sdThresh,
            timeout=self.timeout,
        )

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
        self._create_listbox()
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
            self.GUI_ELE["LOAD_SESS"]: self.load_sessions4GUI,
            self.GUI_ELE["SAV_FIG"]: [],
            "separator": None,
            "Quit": self.quit_app,
        }
        self.MenuBarUtils.create_menu(self.GUI_ELE["MMENU"], file_commands)
        self.MenuBarUtils.update_menu_item_state(
            self.GUI_ELE["MMENU"], self.GUI_ELE["SAV_FIG"], tk.DISABLED
        )

        # Add Edit menu
        edit_commands = {self.GUI_ELE["RESET"]: self.reset_gui}
        self.MenuBarUtils.create_menu(self.GUI_ELE["EDIT"], edit_commands)

        if self.enable_console:
            # Add View menu
            view_commands = {
                self.GUI_ELE["CONSOLE"]: self.toggle_console,
            }
            self.MenuBarUtils.create_menu(self.GUI_ELE["VIEW"], view_commands)

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

    def _create_listbox(self) -> None:
        self.cspan4listbox = int(self.figspan_ASP * self.cspan4list_ratio)
        # create label for listbox
        listbox_label_config = self.config_maker(
            text="Sessions",
            row=0,
            column=self.figspan_ASP,
            columnspan=self.cspan4listbox,
            sticky="nsew",
        )
        self.WidgetUtils.create_tk_widget(
            widget_type="LABEL", name="SESSIONS_LABEL", config=listbox_label_config
        )
        # Create a frame to hold listbox and button
        listbox_frame_config = self.config_maker(
            row=1,
            column=self.figspan_ASP,
            columnspan=self.cspan4listbox,
            sticky="nsew",
        )
        self.WidgetUtils.create_tk_widget(
            widget_type="FRAME",
            name="LISTBOX_FRAME",
            config=listbox_frame_config,
        )
        self.LISTBOX_FRAME = self.WidgetUtils.frames["LISTBOX_FRAME"]

        # Listbox to the right of ASP image
        listbox_config = self.config_maker(
            row=0,
            column=0,
            sticky="nsew",
            height=20,
            width=30,
            in_=self.LISTBOX_FRAME,
        )

        self.WidgetUtils.create_tk_widget(
            widget_type="LISTBOX",
            name="DISPLAY_SESSIONS",
            config=listbox_config,
            diff_parent=self.LISTBOX_FRAME,
        )

        # add button to select session
        load_button_config = self.config_maker(
            text=self.GUI_ELE["UNUSABLE"],
            row=1,
            column=0,
            sticky="nsew",
            command=self.load_file4GUI,
            in_=self.LISTBOX_FRAME,
            state=tk.DISABLED,
        )
        self.WidgetUtils.create_tk_widget(
            widget_type="BUTTON",
            name="LOAD_SESSION",
            config=load_button_config,
            diff_parent=self.LISTBOX_FRAME,
        )

        self.LISTBOX = self.WidgetUtils.listboxes["DISPLAY_SESSIONS"]

        self.LB_list.append(self.LISTBOX)

        self.Frame_List.append(self.LISTBOX_FRAME)

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
            row=1,
            column=int(self.figspan_ASP + self.cspan4listbox),
            rowspan=1,
            sticky="nsew",
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
        self.Frame_List.append(self.ASP_FIG_FRAME)
        self.Frame_List.append(self.CTP_FIG_FRAME)
        self.Frame_List.append(self.CTP_FIG_TRACE_FRAME)
        self.Frame_List.append(self.ASP_ENTRY)
        self.Frame_List.append(self.CTP_ENTRY)

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
        self.ax_ASP.set_facecolor(self.color_dict["black"])
        self.ax_ASP.set_xticks([])
        self.ax_ASP.set_yticks([])

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
        for lb in self.LB_list:
            self.config_rowsNcols2resize(columns=True, rows=True, parent=lb)
        for can in self.canvas_list:
            can.draw()
        self.root.update()

    ######################################################
    #  Load file funcs
    ######################################################

    def load_sessions4GUI(self) -> None:
        """
        Loads a file for the GUI.

        If a file is already loaded, it resets the GUI.
        It then performs various update operations and prints messages.
        """
        if self.subjID:
            self.reset_gui()
        self._load_sessionsNfill_listbox()

    def load_file4GUI(self) -> None:
        self._load_selected_session()
        self._load_segDictNRelatedData()
        self._update_load_label_post_load()
        self._format_post_load_entries()
        self._update_ASP_wDSimage_at_Start()
        self.print_post_loadORreset_msg()
        self.update_CTP_plot(reset=True)
        print()
        # self.update_ASpat_image_wDS_Entry()
        pass

    def _load_selected_session(self) -> None:
        """
        Event handler for the listbox selection event.
        """
        if not self.LISTBOX.curselection():
            return
        selected_idx = self.LISTBOX.curselection()[0]
        self.selected_sess = self.sess_list2use[selected_idx]

        self.full_sessPath = self.selected_sess[0]
        self.sess_basename = self.selected_sess[1]
        self.pkl_fname = self.selected_sess[2]
        self.date = self.selected_sess[3]
        self.subjID = self.selected_sess[4]
        self.sessType = self.selected_sess[5]

        # change directory to the selected session
        os.chdir(self.full_sessPath)

        print(f"Selected session: {self.sess_basename}")
        self.print_wFrm(f"Full Session Path: {self.full_sessPath}")
        self.print_wFrm(f"PKL File Name: {os.path.basename(self.pkl_fname)}")
        self.print_wFrm(f"Date: {self.date}", frame_num=1)
        self.print_wFrm(f"Subject ID: {self.subjID}", frame_num=1)
        self.print_wFrm(f"Session Type: {self.sessType}", frame_num=1)

    def _load_sessionsNfill_listbox(self) -> None:
        self.dataDir, self.sess_list2use = self.get_eligible_sessions(
            ql_file_tag="SD_PKL", date_first=True
        )
        self.LISTBOX.delete(0, tk.END)
        for sess in self.sess_list2use:
            self.LISTBOX.insert(tk.END, sess[1])

        # activate load button
        load_button_config = self.config_maker(
            text=self.GUI_ELE["LOAD_SESS_SELECTED"],
            row=1,
            column=0,
            sticky="nsew",
            command=self.load_file4GUI,
            in_=self.LISTBOX_FRAME,
            state=tk.NORMAL,
        )
        self.WidgetUtils.create_tk_widget(
            widget_type="BUTTON",
            name="LOAD_SESSION",
            config=load_button_config,
            diff_parent=self.LISTBOX_FRAME,
        )
        # print()

    def _load_segDictNRelatedData(self) -> None:
        """
        Load utilities for CSS (Cue Shift Structure).

        This method loads the necessary utilities for CSS, including the pkl file name,
        file ID, date, session, cueShiftStruc, segDict, and pksDict. It also removes the
        cueShiftStruc after extracting the required information.

        Returns:
            None
        """
        self.print_wFrm("Extracting segDict", end="", flush=True)
        self.segDict = self.saveNloadUtils.load_file(
            self.pkl_fname,
        )
        self.C_Temporal = self.segDict["C_Temporal"]
        self.A_Spatial = self.segDict["A_Spatial"]
        self.max_cell_num = self.A_Spatial.shape[1]

        self.utils.print_done_small_proc(new_line=False)

        self.print_wFrm("Finding pks", end="", flush=True, frame_num=1)
        self.pks = {}
        for seg in range(self.C_Temporal.shape[0]):
            pks, _, _ = self.PKSUtils.find_CaTransients(
                self.C_Temporal[seg, :].copy(), cell_num=seg
            )
            self.pks[f"{self.PKSkey['SEG']}{seg}"] = pks

        self.utils.print_done_small_proc(new_line=False)

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
        load_label_wsubjID_config = self.config_maker(
            text=self.GUI_ELE["LOAD_ID"].format(self.subjID, self.date, self.sessType)
        )
        self.WidgetUtils.tk_update(
            name=self.GUICLASS["BU_LOAD"], config=load_label_wsubjID_config
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
        range_label_config = self.config_maker(
            text=f" Available Range: 0 - {self.max_cell_num} ",
            row=0,
            column=0,
            sticky="nsew",
            in_=self.ASP_ENTRY,
        )
        self.WidgetUtils.create_tk_widget(
            widget_type="LABEL",
            name=self.GUICLASS["BU_RANGE"],
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

        label_CTP_config = self.config_maker(
            text=f"Select Cell # for Signal Trace (0-{self.max_cell_num})",
            row=0,
            column=0,
            columnspan=1,
            sticky="nsew",
            in_=self.CTP_ENTRY,
        )
        self.WidgetUtils.create_tk_widget(
            widget_type="LABEL",
            name="CTP_ENTRY_LABEL",
            config=label_CTP_config,
            diff_parent=self.CTP_ENTRY,
        )

        scale_CTP_config = self.config_maker(
            from_=0,
            to=self.max_cell_num,
            orient="horizontal",
            variable=self.CTP_VAR,
            command=self.on_slider_move,
            row=1,
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
        vcmd = (
            self.root.register(lambda P: self.validate_entry(P, self.max_cell_num)),
            "%P",
        )
        entry_CTP_config = self.config_maker(
            textvariable=self.CTP_VAR,
            validate="key",
            validatecommand=vcmd,
            row=2,
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
            axes.clf()

        self.ax_ASP.set_facecolor(self.color_dict["black"])

        for can in self.canvas_list:
            can.figure.clear()
            can.draw()

        for entry in self.Frame_List:
            if entry in [self.ASP_ENTRY, self.CTP_ENTRY] and entry is not None:
                entry.destroy()
        # after destroying must rebuild entry frames
        self.create_base_gui_header()
        self.init_widgets()

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
        import matplotlib.pyplot as plt

        self.ax_ASP.clear()

        image_file = self.findLatest(self.image_utils.get_DSImage_filename(wCMAP=True))
        if image_file:
            img = plt.imread(image_file)
            # self.print_wFrm("Loaded DS image with CMap", frame_num=1)
            self.ax_ASP.imshow(img)
        else:
            image_file = self.findLatest(self.image_utils.get_DSImage_filename())
            img = self.image_utils.read_image(image_file)
            # self.print_wFrm("Loaded DS image", frame_num=1)
            self.ax_ASP.imshow(img, cmap="gray", aspect="equal")

        self.ax_ASP.set_xticks([])
        self.ax_ASP.set_yticks([])
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
        print(self.done_gui)
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
        selected_cell = int(self.CTP_VAR.get())

        # Plot selected cell in A_Spatial
        self.plot_select_cell_ASpat(
            A_Spatial=self.A_Spatial,
            cell_num=selected_cell,
            ax=self.ax_ASP,
            cmap=self.cmaps[1],
            alpha=0.8,
        )

        # Plot selected cell in CTP
        self.plot_select_cell_CTP(
            CTP=self.C_Temporal,
            pks=self.pks,
            cell_num=selected_cell,
            ax_CTP=self.ax_CTP,
            ax_pk_trace=self.ax_CTP_trc,
            color_from_cmap=self.cmaps[1](0.5),
        )

        # Redraw all canvas
        for can in self.canvas_list:
            can.draw()

        print(self.done_gui)
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
        status = "created" if self.subjID else "removed"
        print(
            dedent(
                f"""\
    {self.done_gui}
    {self.FR}Loaded ID: {self.subjID}
    {self.FR}Entry selection for cells was {status}
    """
            )
        )


######################################################
#  run function when script is called directly
######################################################
if __name__ == "__main__":
    segDictGUI(enable_console=True)

import tkinter as tk
from textwrap import dedent
from tkinter import filedialog
from CLAH_ImageAnalysis.GUI import BaseGUI
from CLAH_ImageAnalysis.utils import load_file, text_dict, color_dict
from matplotlib.colors import rgb2hex
from rich import print


text_lib = text_dict()
FR, EFR = text_lib["frames"]["FR"], text_lib["frames"]["EFR"]
color_dict = color_dict()
file_tag = text_lib["file_tag"]
dict_name = text_lib["dict_name"]
GUICLASS = text_lib["GUICLASSUTILS"]
GUI_ELE = text_lib["GUI_ELEMENTS"]
done_str = text_lib["completion"]["small_proc"]
done_gui = text_lib["completion"]["GUI"]


class multiSessASpatCheck(BaseGUI):
    """
    A class that represents a GUI for multi-session spatial analysis.

    Attributes:
        multSessSegStruc (None or dict): A dictionary containing multi-session segmentation structures.
        fileID (None or str): The ID of the loaded file.
        figsize (tuple): The size of the figure in inches.

    Methods:
        __init__(self, enable_console=False): Initializes the multiSessASpatCheck object.
        _init_global_vars_MSS(self, figsize=(8, 8)): Initializes the global variables for multi-session spatial analysis.
        init_widgets(self): Initializes the widgets in the GUI.
        _create_buttons(self): Creates the buttons in the GUI.
        _create_plts(self): Creates the canvas and plots in the GUI.
        _fill_in_menu_bar(self): Fills in the menu bar in the GUI.
        on_resize(self): Handles the resizing of the GUI.
        load_file(self): Loads a file and performs post-load GUI setup.
        cell_selector_setup(self): Sets up the cell selector section in the GUI.
        reset_gui(self): Resets the GUI to its initial state.
    """

    def __init__(self, enable_console: bool = False) -> None:
        """
        Initializes an instance of the multiSessASpatCheck class.

        Parameters:
            enable_console (bool): Flag to enable console output. Default is False.
        """
        self.multSessSegStruc = None
        self.fileID = None

        print(GUI_ELE["GUI_TITLE_START"].format(GUI_ELE["MSS_ASPAT_TITLE"]))
        super().__init__(enable_console=enable_console)

        # Set title of GUI
        self.set_GUI_title(GUI_ELE["MSS_ASPAT_TITLE"])

        print(f"{FR}{GUICLASS['BU_GLOBALVARS']}")
        self._init_global_vars_MSS()

        print(f"{FR}{GUICLASS['BU_INITWIDGETS']}")
        self.init_widgets()

        print(f"{FR}{GUICLASS['BU_MENUBAR']}")
        self.display_menu_bar()

        # Enable window resizing
        self.root.bind("<Configure>", self.on_resize())
        # run GUI
        self.run_GUI()

    def _init_global_vars_MSS(self, figsize: tuple = (8, 8)) -> None:
        """
        Initializes global variables for the multiSessASpatCheck class.

        Parameters:
        - figsize (tuple): The size of the figure (default is (8, 8)).

        Returns:
        None
        """
        self.col_entry = {}
        self.col_range_label = {}
        self.col_sess_label = {}
        self.col_vars = {}
        self.cell_select_frame = None
        self.figsize = figsize

    def init_widgets(self) -> None:
        """
        Initializes the widgets in the GUI.

        This method creates buttons, plots, and fills in the menu bar.
        It also resizes the GUI accordingly.

        Parameters:
            None

        Returns:
            None
        """
        self._create_buttons()
        self._create_plts()
        self._fill_in_menu_bar()
        # resize GUI accordingly
        self.on_resize()

    def _create_buttons(self) -> None:
        """
        Creates and initializes the buttons in the GUI.

        This method creates the update button and the reset button in the GUI.
        The update button is initially disabled and has a black background and white text color.
        The reset button is created with a specified row and a reset command.

        Parameters:
            None

        Returns:
            None
        """
        # create update button
        update_button_config = self.config_maker(
            text=GUI_ELE["UNUSABLE"],
            command=[],
            bg=color_dict["black"],
            fg=color_dict["white"],
            row=5,
            column=0,
            columnspan=self.tot_column_used,
            state=tk.DISABLED,
        )
        self.WidgetUtils.create_tk_widget(
            widget_type="BUTTON",
            name=GUICLASS["BU_UPDATE"],
            config=update_button_config,
        )

        # Reset Button
        self.create_reset_button(row=7, reset_command=self.reset_gui)

    def _create_plts(self):
        """
        Create plots for ASP image.

        This method creates a canvas for the ASP image and sets the face color of the plot to black.

        Parameters:
        - master: The master widget (typically a Tkinter root or Toplevel widget) that will contain the canvas.
        - figsize: The size of the figure (width, height) in inches.
        - row: The row index of the grid where the canvas will be placed.
        - column: The column index of the grid where the canvas will be placed.
        - columnspan: The number of columns that the canvas will span.

        Returns:
        - fig: The matplotlib figure object.
        - ax: The matplotlib axes object.
        - canvas: The Tkinter canvas object.
        """
        self.fig, self.ax, self.canvas = self.create_canvas_plt(
            master=self.root,
            figsize=self.figsize,
            row=6,
            column=0,
            columnspan=self.tot_column_used,
        )
        self.ax.set_facecolor(color_dict["black"])

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
            GUI_ELE["LOAD"]: self.load_file,
            GUI_ELE["SAV_FIG"]: self.save_figure,
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

    def on_resize(self) -> None:
        """
        Callback method triggered when the window is resized.
        It resizes the rows and columns of the GUI and redraws the canvas.
        """
        self.config_rowsNcols2resize(columns=True, rows=True)
        self.canvas.draw()

    ######################################################
    #  functions used by GUI widgets
    ######################################################

    def load_file(self) -> None:
        """
        Loads a file and performs necessary GUI setup after loading.

        If a fileID already exists, the GUI is reset before completing the load process.

        Parameters:
            None

        Returns:
            None
        """
        # if fileID already exists, reset GUI before completing load process
        if self.fileID:
            print("File already loaded... so resetting GUI...")
            self.reset_gui()

        pkl_fname, self.fileID = self.get_pkl_fnameNfileID(
            dict_name["MSS"], file_tag["PKL"]
        )
        # load pkl file
        self.multSessSegStruc = load_file(pkl_fname, file_tag["PKL"])
        # update load label
        self.WidgetUtils.tk_update(
            GUICLASS["BU_LOAD"], {"text": GUI_ELE["LOAD_ID"].format(self.fileID)}
        )
        # further post load GUI setup
        # creates cell selector frame/section
        self.cell_selector_setup()
        # update Class vars
        # update update button
        upd_update_button_post_load_config = self.config_maker(
            text=GUI_ELE["UPDATE"],
            command=self.update_ASpat_image,
            bg=color_dict["white"],
            fg=color_dict["black"],
            state=tk.NORMAL,
        )
        self.WidgetUtils.tk_update(
            GUICLASS["BU_UPDATE"], upd_update_button_post_load_config
        )
        # turn on save fig option in menu
        self.MenuBarUtils.update_menu_item_state(
            GUI_ELE["MMENU"], GUI_ELE["SAV_FIG"], tk.NORMAL
        )
        self.on_resize()
        # print message to confirm operation
        self.print_post_loadORreset_msg()

    def cell_selector_setup(self) -> None:
        """
        Sets up the cell selector section of the GUI.

        This method creates and configures the necessary widgets for selecting cells in the GUI.
        It creates a frame for the cell selector section and adds labels, entry fields, and range labels
        for each session.

        Returns:
            None
        """
        # global tot_column_used
        # tot_column_used = len(multSessSegStruc.keys()) * 2
        # num_sessions = len(self.multSessSegStruc.keys())
        total_columns = self.tot_column_used
        # columnspan = total_columns // num_sessions
        cell_select_frame_config = self.config_maker(
            row=1, column=0, columnspan=total_columns, sticky="nsew"
        )
        self.WidgetUtils.create_tk_widget(
            widget_type="FRAME",
            name=GUICLASS["BU_CELL_SEL"],
            config=cell_select_frame_config,
        )
        self.cell_select_frame = self.WidgetUtils.frames[GUICLASS["BU_CELL_SEL"]]
        parent_frame = self.cell_select_frame

        # create label for selecting cells section
        cell_select_label_config = self.config_maker(
            text=GUI_ELE["CELL_SEL"], font=("bold"), row=0, column=0
        )
        self.WidgetUtils.create_tk_widget(
            widget_type="LABEL",
            name=GUICLASS["BU_CELL_SEL"],
            config=cell_select_label_config,
            diff_parent=parent_frame,
        )
        for i, numSess in enumerate(self.multSessSegStruc.keys()):
            max_col = self.multSessSegStruc[numSess]["A_SPATIAL"].shape[1] - 1
            # Create and place session label using WidgetUtils
            row_num = i + 1
            sess_label_config = self.config_maker(
                text=f" Session {row_num} ",
                bg=rgb2hex(self.cmaps[i % len(self.cmaps)](0.5)),
                font=("bold"),
                row=row_num,
                column=0,
                sticky="nsew",
                in_=parent_frame,
            )
            self.WidgetUtils.create_tk_widget(
                widget_type="LABEL",
                name=GUICLASS["BU_SESS"],
                config=sess_label_config,
                diff_parent=parent_frame,
            )
            # Create and place range label using WidgetUtils
            range_label_config = self.config_maker(
                text=f" Available Range: 0 - {max_col} ",
                row=row_num,
                column=2,
                sticky="nsew",
                in_=parent_frame,
            )
            self.WidgetUtils.create_tk_widget(
                widget_type="LABEL",
                name=GUICLASS["BU_RANGE"],
                config=range_label_config,
                diff_parent=parent_frame,
            )
            # Create and place entry using WidgetUtils
            col_var = tk.StringVar(self.root)
            col_var.set("0")
            entry_config = self.config_maker(
                textvariable=col_var,
                row=row_num,
                column=1,
                bindings={"<Return>": lambda event: self.update_ASpat_image()},
                in_=parent_frame,
                justify="center",
            )
            entry_identifier = f"{GUICLASS['BU_UPDATE']}_sess_{i}"
            self.WidgetUtils.create_tk_widget(
                widget_type="ENTRY",
                name=entry_identifier,
                config=entry_config,
                diff_parent=parent_frame,
            )
            # store entry & labels into dict for later use
            self.col_entry[numSess] = self.WidgetUtils.entries[entry_identifier]
            self.col_vars[numSess] = col_var
            self.col_range_label[numSess] = self.WidgetUtils.labels[
                GUICLASS["BU_RANGE"]
            ]
            self.col_sess_label[numSess] = self.WidgetUtils.labels[GUICLASS["BU_SESS"]]
        # resize columns for parent frame
        self.config_rowsNcols2resize(parent_frame, rows=True)

    def reset_gui(self) -> None:
        """
        Resets the GUI to its initial state.

        This method clears the axes and figure, destroys the cell selector section,
        resets values in dictionaries, updates labels and buttons, disables save
        figure option in the menu, and prints a confirmation message.

        Parameters:
            None

        Returns:
            None
        """
        self.print_flush_statements(RESET=True)
        # clear axes & figure
        self.ax.clear()
        self.canvas.draw()
        self.multSessSegStruc = None
        self.fileID = None

        # destroy cell selector section
        if self.cell_select_frame is not None:
            self.cell_select_frame.destroy()
        # reset values in dictionaries
        self.col_entry, self.col_range_label, self.col_range_label, self.col_vars = (
            {},
            {},
            {},
            {},
        )
        # reseting values w/in Class vars
        # reset load label
        self.WidgetUtils.tk_update(
            GUICLASS["BU_LOAD"], {"text": GUI_ELE["LOAD_ID_EMPTY"]}
        )
        # reset update button
        upd_update_button_post_reset_config = self.config_maker(
            text=GUI_ELE["UNUSABLE"],
            bg=color_dict["black"],
            fg=color_dict["white"],
            command=[],
            state=tk.DISABLED,
        )
        self.WidgetUtils.tk_update(
            GUICLASS["BU_UPDATE"], upd_update_button_post_reset_config
        )
        # turn off save fig option in menu
        self.MenuBarUtils.update_menu_item_state(
            GUI_ELE["MMENU"], GUI_ELE["SAV_FIG"], tk.DISABLED
        )
        self.on_resize()
        # print message to confirm operation
        self.print_post_loadORreset_msg()

    def update_ASpat_image(self) -> None:
        """
        Updates the A_Spatial image in the GUI.

        This method clears the axes, sets the background color to black, and displays the A_Spatial images for each numSess.
        It also generates a title for the plot based on the selected columns.

        Returns:
            None
        """
        self.print_flush_statements(UPDATE_ASPAT=True)
        # Clear axes
        self.ax.clear()

        # Set background color to black
        self.ax.set_facecolor(color_dict["black"])

        # selected cols dict for later printing
        selCol_dict = {}

        title = GUI_ELE["PLT_TITLE"].format(self.fileID) + "\n"
        # Get selected cols and display images for each numSess
        for i, (numSess, col_var) in enumerate(self.col_vars.items()):
            # Get A_Spatial array for numSess
            A_Spatial = self.multSessSegStruc[numSess]["A_SPATIAL"]
            selected_cols = self.get_selected_cells(col_var.get(), A_Spatial.shape[1])
            selCol_dict[numSess] = selected_cols
            for col in selected_cols:
                self.plot_select_cell_ASpat(
                    A_Spatial=A_Spatial,
                    cell_num=col,
                    ax=self.ax,
                    cmap=self.cmaps[i % len(self.cmaps)],
                )
            # automate title addition
            if len(selected_cols) > 1:
                title = (
                    title
                    + EFR[0:-2]
                    + GUI_ELE["PLT_CELLS"].format(
                        i + 1, selected_cols[0], selected_cols[-1]
                    )
                    + " "
                )
            else:
                title = (
                    title
                    + EFR[0:-2]
                    + GUI_ELE["PLT_CELL"].format(i + 1, selected_cols[0])
                    + " "
                )
        title = title + EFR[0]
        self.ax.set_title(title, fontsize=10)
        self.canvas.draw()
        print(done_gui)
        print("Plotting Cell #s:")
        for i, (numSess, col_var) in enumerate(self.col_vars.items()):
            current_col = selCol_dict[numSess]
            self.print_cell_num_plotted_ASPat(
                selected_cells=current_col, sess_num=i + 1
            )
            # if len(current_col) > 1:
            #     print(f"{FR}Session {i+1}: Cells {current_col[0]}-{current_col[-1]}")
            # else:
            #     print(f"{FR}Session {i+1}: Cell {current_col[0]}")
        print()

    def print_post_loadORreset_msg(self) -> None:
        """
        Prints a message indicating the status of the loaded ID, entry selection for each session,
        and the usability and availability of the Update Plot button and Save Figure option.
        """
        status = "created" if self.fileID else "removed"
        usability = "usable" if self.fileID else "unusable"
        availability = "available" if self.fileID else "unavailable"

        print(
            dedent(
                f"""\
    {done_gui}
    {FR}Loaded ID: {self.fileID}
    {FR}Entry selection for each session was {status}
    {FR}Update Plot button is now {usability}
    {FR}Save Figure option is now {availability}
    """
            )
        )

    def save_figure(self) -> None:
        """
        Saves the current figure as a PNG file.

        This method opens a dialog for file saving and saves the current figure as a PNG file
        with the user-specified file name. If the user cancels the file saving dialog, the
        figure is not saved.

        Returns:
            None
        """
        self.print_flush_statements(SAVE_FIG=True)
        # Open a dialog for file saving
        file_path = filedialog.asksaveasfilename(defaultextension=".png")
        if file_path:
            self.fig.savefig(file_path)
        print(done_gui)
        print(f"{FR}Figure saved as: {file_path}")
        print()


######################################################
#  run function when script is called directly
######################################################
if __name__ == "__main__":
    multiSessASpatCheck(enable_console=True)

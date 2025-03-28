"""
This module contains the implementation of a basic GUI structure using tkinter.
The BaseGUI class provides a foundation for creating GUI applications with common features such as a main window, console window, menu bar, and widgets.
"""

import os
import sys
import tkinter as tk
import easygui
from rich import print
from tkinter import scrolledtext
from ttkthemes import ThemedTk
from CLAH_ImageAnalysis.GUI import GUI_Utils
from CLAH_ImageAnalysis import utils
from CLAH_ImageAnalysis import dependencies
from matplotlib.colors import ListedColormap
from typing import Callable


class BaseGUI(GUI_Utils):
    """
    A base class for creating GUI applications.

    Args:
        enable_console (bool, optional): Flag to enable the console window. Defaults to False.
        tot_column_used (int, optional): The total number of columns used in the GUI layout. Defaults to 12.
        x (int, optional): The x-coordinate of the main window. Defaults to None.
        y (int, optional): The y-coordinate of the main window. Defaults to None.
        x_offset (int, optional): The x-offset of the main window. Defaults to None.
        y_offset (int, optional): The y-offset of the main window. Defaults to None.

    Attributes:
        root (ThemedTk): The main Tkinter window.
        enable_console (bool): Flag indicating if the console window is enabled.
        WidgetUtils (WidgetFactory): An instance of the WidgetFactory class for creating GUI widgets.
        menubar (tk.Menu): The menu bar of the main window.
        MenuBarUtils (MenuBarFactory): An instance of the MenuBarFactory class for creating menu bars.
        console_window (tk.Toplevel): The console window.
        console_text (scrolledtext.ScrolledText): The text widget for displaying console output.
        StdOutFuncs (StdoutRedirector): An instance of the StdoutRedirector class for redirecting stdout to the console.
        geometry (str): The geometry of the main window.
        tot_column_used (int): The total number of columns used in the GUI layout.
        cmaps (List[ListedColormap]): A list of color maps used in the GUI.
    """

    def __init__(
        self,
        enable_console: bool = False,
        tot_column_used: int = 12,
        x: int | None = None,
        y: int | None = None,
        x_offset: int | None = None,
        y_offset: int | None = None,
    ) -> None:
        """
        Initializes the BasicGUIStruc class.

        Parameters:
            enable_console (bool, optional): Determines if the console window is enabled. Defaults to False.
            tot_column_used (int, optional): The total number of columns used in the GUI. Defaults to 12.
            x (int, optional): The x-coordinate of the main window. Defaults to None.
            y (int, optional): The y-coordinate of the main window. Defaults to None.
            x_offset (int, optional): The x-offset of the main window. Defaults to None.
            y_offset (int, optional): The y-offset of the main window. Defaults to None.
        Returns:
            None
        """
        # init GUI_Utils
        super().__init__()
        # init Tkinter (themed)
        self.root = ThemedTk()
        # set theme
        self.root.set_theme("breeze")

        # init utilsNconstants
        #! IMPORTANT: must be called before init_global_vars_basic
        self.init_utilsNconstants()

        # set geometry of main window
        if not x and not y and not x_offset and not y_offset:
            # will revert to default geometry
            self.init_global_vars_basic(tot_column_used)
        else:
            self.init_global_vars_basic(
                tot_column_used, x=x, y=y, x_offset=x_offset, y_offset=y_offset
            )

        self.print_wFrm(f"Creating main window with geometry: {self.geometry}")

        # determine if console window is enabled
        self.enable_console = enable_console
        if self.enable_console:
            # if True, all console output goes to console window
            # if false, all console output goes to terminal
            self.print_wFrm("Console window enabled")
            self.print_wFrm(
                f"{self.FR}Click View -> Console to toggle console visibility"
            )
            self.create_console_window()
        # init WidgetUtils and MenuBarUtils
        self.print_wFrm("GUI Module is ready to be processed")

        self.print_wFrm("Initializing Widget Utils")
        self.WidgetUtils = self.WidgetFactory(self.root)

        self.print_wFrm("Initializing MenuBar Utils")
        self.menubar = tk.Menu(self.root)
        self.MenuBarUtils = self.MenuBarFactory(self.menubar)

        # can also adjust geometry here need be

        # set window size and position
        self.root.geometry(self.geometry)

        # set up generic load file header for GUI
        self.create_base_gui_header()

        # intercepts the close button of the main window
        self.root.protocol("WM_DELETE_WINDOW", self.quit_app)

    def init_global_vars_basic(
        self,
        tot_column_used: int,
        x: int = 800,
        y: int = 800,
        x_offset: int = 100,
        y_offset: int = 100,
    ) -> None:
        """
        Initializes global variables for the basic GUI structure.

        Parameters:
        - tot_column_used (int): The total number of columns used.
        - x (int): The width of the GUI window (default: 800).
        - y (int): The height of the GUI window (default: 800).
        - x_offset (int): The x-coordinate offset of the GUI window (default: 100).
        - y_offset (int): The y-coordinate offset of the GUI window (default: 100).

        Returns:
            None
        """
        self.tot_column_used = tot_column_used
        self.geometry = f"{x}x{y}+{x_offset}+{y_offset}"
        self.setup_colormaps()

    def init_utilsNconstants(self) -> None:
        """
        Initializes utility functions and constants.
        """
        # funcs
        self.utils = utils
        self.color_dict = self.utils.color_dict()
        self.enum_utils = self.utils.enum_utils
        self.findLatest = self.utils.findLatest
        self.image_utils = self.utils.image_utils
        self.fig_tools = self.utils.fig_tools
        self.saveNloadUtils = self.utils.saveNloadUtils
        self.print_wFrm = self.utils.print_wFrame
        self.dep = dependencies

        # strings
        self.text_lib = self.utils.text_dict()
        self.file_tag = self.text_lib["file_tag"]
        self.dict_name = self.text_lib["dict_name"]
        self.FR, self.EFR = (
            self.text_lib["frames"]["FR"],
            self.text_lib["frames"]["EFR"],
        )
        self.GUI_ELE = self.text_lib["GUI_ELEMENTS"]
        self.GUICLASS = self.text_lib["GUICLASSUTILS"]
        self.GUI_str = self.text_lib["GUI_ELEMENTS"]
        self.breaker = self.text_lib["breaker"]["hash_half"]
        self.done_gui = self.text_lib["completion"]["GUI"]

    def create_console_window(self, height: int = 30, width: int = 100) -> None:
        """
        Create a console window for displaying standard output.

        Parameters:
            height (int): The height of the console window. Default is 30.
            width (int): The width of the console window. Default is 100.
        """

        # Create a Toplevel window for the console
        self.console_window = tk.Toplevel(self.root)
        self.console_window.title("Console")
        self.console_window.deiconify()  # console window visible by default

        # Prevents console window from being deleted permanently
        self.console_window.protocol("WM_DELETE_WINDOW", self.toggle_console)
        self.console_text = scrolledtext.ScrolledText(
            self.console_window,
            state="disabled",
            height=height,
            width=width,
            bg=self.color_dict["black"],
            fg=self.color_dict["white"],
        )
        self.console_text.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.StdOutFuncs = self.StdoutRedirector(self.console_text)
        sys.stdout = self.StdOutFuncs

    def display_menu_bar(self) -> None:
        """
        Configures the root window to display the menu bar.

        This method sets the menu bar of the root window to the `menubar` attribute of the class.

        Parameters:
        - self: The instance of the class.

        Returns:
        - None
        """
        self.root.config(menu=self.menubar)
        self.utils.print_done_small_proc()

    def toggle_console(self) -> None:
        """
        Toggles the visibility of the console window.

        If the console window is currently withdrawn (hidden), it will be shown.
        If the console window is currently shown, it will be hidden.
        """
        if self.console_window.state() == "withdrawn":
            self.console_window.deiconify()  # Show the console window
        else:
            self.console_window.withdraw()  # Hide the console window

    def create_base_gui_header(self) -> None:
        """
        Creates the base GUI header.

        This function creates a label for displaying information, such as file load status.

        Returns:
            None
        """
        cspan = self.tot_column_used // 2
        # Create place a label for displaying information (e.g., file load status)
        load_file_frame = tk.Frame(self.root)
        load_file_frame.grid(row=0, column=0, columnspan=cspan)

        load_label_config = self.config_maker(
            text=self.GUI_str["LOAD_ID_EMPTY"],
            row=0,
            column=0,
            sticky="w",
            in_=load_file_frame,
            columnspan=cspan,
        )
        self.WidgetUtils.create_tk_widget(
            widget_type="LABEL", name=self.GUICLASS["BU_LOAD"], config=load_label_config
        )

    def setup_colormaps(self):
        """
        Set up the color maps for the GUI.

        This method initializes a list of color maps using the color_dict dictionary.
        Each color map is a ListedColormap object with two colors: "none" and a specific color from the color_dict.

        Returns:
            None
        """
        self.cmaps = [
            ListedColormap(["none", self.color_dict["red"]]),
            ListedColormap(["none", self.color_dict["blue"]]),
            ListedColormap(["none", self.color_dict["green"]]),
            ListedColormap(["none", self.color_dict["yellow"]]),
            ListedColormap(["none", self.color_dict["violet"]]),
            ListedColormap(["none", self.color_dict["orange"]]),
        ]

    def config_rowsNcols2resize(
        self, parent: tk.Widget | None = None, **kwargs
    ) -> None:
        """
        Configures the rows and columns of a parent widget for resizing.

        Parameters:
            parent: The parent widget to configure. If None, the default parent widget will be used.
            **kwargs: Additional keyword arguments.
                rows: The number of rows to configure.
                columns: The number of columns to configure.

        Returns:
            None
        """
        total_columns, total_rows = None, None
        parent_to_use = self.WidgetUtils.parent_determiner(parent)

        rows = kwargs.get("rows")
        columns = kwargs.get("columns")

        if rows:
            total_rows = parent_to_use.grid_size()[1]
            for row in range(total_rows):
                parent_to_use.rowconfigure(row, weight=1)
        if columns:
            total_columns = parent_to_use.grid_size()[0]
            for col in range(total_columns):
                parent_to_use.columnconfigure(col, weight=1)

    def _prerun_GUI_proc(self) -> None:
        """
        Perform pre-run tasks for the GUI.

        This method updates the window and positions the console next to the main window if enabled.
        """
        # update window
        self.root.update()
        # set console to right of main window
        if self.enable_console:
            self.StdOutFuncs.position_next_to_main(self.root)

    def run_GUI(self) -> None:
        """
        Runs the GUI and displays the output below the print statements.
        """
        self._prerun_GUI_proc()
        print("GUI is operational!")
        print("Output will be displayed below this line")
        print(self.breaker)
        self.root.mainloop()

    def quit_app(self) -> None:
        """
        Quits the application and closes the GUI.

        This method restores the original standard output, prints a closing message,
        and quits the application.

        Returns:
            None
        """
        sys.stdout = self.StdOutFuncs.original_stdout
        print("Closing GUI...")
        if not self.enable_console:
            print(self.breaker)
        self.root.quit()

    def set_GUI_title(self, title: str) -> None:
        """
        Set the title of the GUI window.

        Parameters:
        - title (str): The title to set for the GUI window.

        Returns:
        - None
        """
        self.root.title(title)

    def create_reset_button(self, row: int, reset_command: Callable) -> None:
        """
        Create a reset button widget.

        Args:
            row (int): The row number where the button should be placed.
            reset_command (function): The function to be called when the button is clicked.

        Returns:
            None
        """
        reset_button_config = self.config_maker(
            text=self.GUI_ELE["RESET"],
            command=reset_command,
            bg=self.color_dict["red"],
            fg=self.color_dict["white"],
            row=row,
            column=0,
            columnspan=self.tot_column_used,
        )
        self.WidgetUtils.create_tk_widget(
            widget_type="BUTTON",
            name=self.GUICLASS["BU_RESET"],
            config=reset_button_config,
        )

    def get_eligible_sessions(self, ql_file_tag: str, date_first: bool = False) -> None:
        dataDir = None
        sess_list = []
        sess_list2use = []

        QL = self.text_lib["Folders"]["QL"]
        QL_FT = self.text_lib["QL_LNAMES"]
        egu_msg = self.GUI_ELE["EGU_MSG_PAR"]

        dataDir = easygui.diropenbox(msg=egu_msg)
        os.chdir(dataDir)

        sess_list = [os.path.join(dataDir, sess) for sess in os.listdir(dataDir)]
        sess_list2use = []

        for idx, sess in enumerate(sess_list):
            ql_folder = os.path.join(sess, QL)
            fileOFint = os.path.join(ql_folder, QL_FT[ql_file_tag])
            if os.path.lexists(fileOFint):
                file_path = os.path.realpath(fileOFint)
                sess_basename = os.path.basename(sess)
                if date_first:
                    date = sess_basename.split("_")[0]
                    subjID = sess_basename.split("_")[1]
                    sessType = sess_basename.split("_")[2]
                else:
                    date = None
                    subjID = sess_basename.split("_")[0]
                    sessType = None
                sess_list2use.append(
                    (sess, sess_basename, file_path, date, subjID, sessType)
                )

        return dataDir, sess_list2use

    # def get_eligible_sessions(
    #     self, dict_name: str, file_tag: str, date_first: bool = False
    # ) -> tuple[str, str, str | None, str | None]:
    #     """
    #     Function to get the pickle file name and file ID.

    #     Parameters:
    #         dict_name (str): The name of the dictionary.
    #         file_tag (str): The file tag to filter the file selection.
    #         date_first (bool, optional): Whether the date is the first part of the file name. Defaults to False.

    #     Returns:
    #         tuple: A tuple containing the pickle file name and file ID. If `date_first` is True, the tuple also contains
    #         the date and session type.

    #     """
    #     QL = self.text_lib["Folders"]["QL"]
    #     print("Loading folder...", end="", flush=True)
    #     egu_msg = self.text_lib["EGU_MSG_SESS"].format(dict_name)

    #     sess_dir = easygui.diropenbox(msg=egu_msg)

    #     # change directory to session folder
    #     os.chdir(sess_dir)

    #     # get file id
    #     if date_first:
    #         fileID = fname.split("_")[1]
    #         date = fname.split("_")[0]
    #         session_type = fname.split("_")[2]
    #         return pkl_fname, fileID, date, session_type
    #     else:
    #         fileID = fname.split("_")[0]
    #         return pkl_fname, fileID

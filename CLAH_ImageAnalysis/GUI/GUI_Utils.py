import numpy as np
import re
import sys
import tkinter as tk
import matplotlib
from scipy.sparse import csr_matrix, spmatrix
import matplotlib.pyplot as plt
from rich import print
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from CLAH_ImageAnalysis import utils
from matplotlib.colors import rgb2hex
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk

matplotlib.use("TkAgg")
plt.ioff()

text_lib = utils.text_dict()
color_dict = utils.color_dict()
GUI_ELE = text_lib["GUI_ELEMENTS"]
GUICLASS = text_lib["GUICLASSUTILS"]
GUI_str = text_lib["GUI_ELEMENTS"]
breaker = text_lib["breaker"]["hash_half"]

######################################################
#  Widgets
######################################################


class WidgetFactory:
    """
    A class that provides methods for creating and configuring Tkinter widgets.
    """

    def __init__(self, parent: tk.Widget) -> None:
        """
        Initializes an instance of the GUI_Utils class.

        Parameters:
            parent: The parent widget.

        Returns:
            None
        """
        self.parent = parent
        self._init_default_values()
        self._init_empty_widget_dicts()

    def _init_empty_widget_dicts(self) -> None:
        """
        Initializes empty dictionaries for buttons, labels, entries, frames, scales, and grid configurations.
        """
        self.buttons = {}
        self.labels = {}
        self.entries = {}
        self.frames = {}
        self.scales = {}
        self.listboxes = {}
        self.grid_configs = {}

    def _init_default_values(self) -> None:
        """
        Initializes the default values for the GUI_Utils class.

        Sets the default state to tk.NORMAL, default coordinate to 0,
        default blank string to an empty string, and default padding to 10.
        """
        self.default_state = tk.NORMAL
        self.default_coord = 0
        self.default_blank_str = ""
        self.default_padding = 10

    def parent_determiner(self, parent: tk.Widget | None = None) -> tk.Widget:
        """
        Determines the parent to use for the current operation.

        If a specific parent is provided, it will be used. Otherwise, the default parent
        associated with the instance will be used.

        Parameters:
            parent (object): The specific parent object to use (optional).

        Returns:
            object: The parent object to use for the current operation.
        """
        if parent is not None:
            parent_to_use = parent
        else:
            parent_to_use = self.parent
        return parent_to_use

    def create_tk_widget(
        self,
        widget_type: str,
        name: str,
        config: dict,
        diff_parent: tk.Widget | None = None,
    ) -> tk.Widget:
        """
        Creates a new tk_widget with the given name and configuration.

        Parameters:
            widget_type (str): The type of widget to create. Valid values are "LABEL", "BUTTON", "ENTRY", "FRAME", and "SCALE".
            name (str): The name of the widget.
            config (dict): The configuration options for the widget.
            diff_parent (tk.Widget, optional): The parent widget to use for creating the widget. Defaults to None.

        Returns:
            tk.Widget: The created tk_widget.

        Raises:
            ValueError: If an invalid widget_type is provided.

        """
        widget_type = widget_type.upper()
        parent_to_use = self.parent_determiner(diff_parent)
        if widget_type == "LABEL":
            tk_widget = tk.Label(parent_to_use)
        elif widget_type == "BUTTON":
            tk_widget = tk.Button(parent_to_use)
        elif widget_type == "ENTRY":
            tk_widget = tk.Entry(parent_to_use)
        elif widget_type == "FRAME":
            tk_widget = tk.Frame(parent_to_use)
        elif widget_type == "SCALE":
            tk_widget = tk.Scale(parent_to_use)
        elif widget_type == "LISTBOX":
            tk_widget = tk.Listbox(parent_to_use)
        else:
            raise ValueError(f"Invalid widget_type: {widget_type}")

        self.tk_configuration(tk_widget, config)

        # Store the tk_widget with its name
        if widget_type == "LABEL":
            self.labels[name] = tk_widget
        elif widget_type == "BUTTON":
            self.buttons[name] = tk_widget
        elif widget_type == "ENTRY":
            self.entries[name] = tk_widget
        elif widget_type == "FRAME":
            self.frames[name] = tk_widget
        elif widget_type == "SCALE":
            self.scales[name] = tk_widget
        elif widget_type == "LISTBOX":
            self.listboxes[name] = tk_widget

        return tk_widget

    def tk_configuration(self, tk_object: tk.Widget, config: dict) -> None:
        """
        Configures a Tkinter widget based on the provided configuration.

        Parameters:
            tk_object (tkinter.Widget): The Tkinter widget to be configured.
            config (dict): The configuration options for the widget.

        Returns:
            None
        """

        # Common configuration for all widgets
        if not isinstance(tk_object, tk.Frame):
            tk_object.config(
                bg=config.get("bg"),
                fg=config.get("fg"),
                state=config.get("state", self.default_state),
            )

        # Add Listbox-specific configuration
        if isinstance(tk_object, tk.Listbox):
            tk_object.config(
                height=config.get("height"),
                width=config.get("width"),
                selectmode=config.get("selectmode", tk.SINGLE),
                exportselection=config.get("exportselection", False),
            )

        # Handling for 'text' attribute, typically for Labels
        if "text" in config:
            tk_object.config(text=config.get("text"))
            tk_object.config(font=config.get("font"))

        # Special handling for 'command' in case of Buttons
        if isinstance(tk_object, tk.Button):
            tk_object.config(command=config.get("command"))

        # Handling for 'textvariable' in case of Entry widgets
        if isinstance(tk_object, tk.Entry):
            tk_object.config(textvariable=config.get("textvariable"))
            tk_object.config(validate=config.get("validate"))
            tk_object.config(validatecommand=config.get("validatecommand"))
            tk_object.config(justify=config.get("justify"))

        if isinstance(tk_object, tk.Scale):
            tk_object.config(
                from_=config.get("from_"),
                to=config.get("to"),
                orient=config.get("orient"),
                command=config.get("command"),
            )

        # Grid placement configuration
        grid_config = config.get("grid", {})
        self.grid_configs[id(tk_object)] = grid_config
        tk_object.grid(
            row=grid_config.get("row", self.default_coord),
            column=grid_config.get("column", self.default_coord),
            columnspan=grid_config.get("columnspan"),
            rowspan=grid_config.get("rowspan"),
            sticky=grid_config.get("sticky"),
            in_=grid_config.get("in_"),
            padx=grid_config.get("padx", self.default_padding),
            pady=grid_config.get("pady", self.default_padding),
        )

        # Handle optional event bindings
        bindings = config.get("bindings")
        if bindings:
            for event, handler in bindings.items():
                tk_object.bind(event, handler)

    def tk_update(self, name: str, config: dict) -> None:
        """
        Update the configuration of a Tkinter object based on the provided name and configuration.

        Parameters:
            name (str): The name of the Tkinter object to update.
            config (dict): The new configuration settings to apply.

        Returns:
            None

        Raises:
            None
        """
        tk_object = None
        if name in self.labels:
            tk_object = self.labels[name]
        elif name in self.buttons:
            tk_object = self.buttons[name]
        elif name in self.entries:
            tk_object = self.entries[name]
        elif name in self.listboxes:
            tk_object = self.listboxes[name]

        if tk_object:
            # Get current configuration
            current_config = {
                "text": tk_object.cget("text"),
                "bg": tk_object.cget("bg"),
                "fg": tk_object.cget("fg"),
                "state": tk_object.cget("state"),
                "grid": self.grid_configs.get(id(tk_object), {}),
            }

            # Merge current configuration with new settings
            updated_config = {**current_config, **config}

            # Apply the updated configuration
            self.tk_configuration(tk_object, updated_config)
        else:
            print(f"{name} not found")

    @staticmethod
    def destroy_widget_dict(widget_in_dict: dict) -> dict:
        """
        Destroy all widgets in the given dictionary and clear the dictionary.

        Parameters:
            widget_in_dict (dict): A dictionary containing widgets.

        Returns:
            dict: An empty dictionary after destroying all widgets.
        """
        for widget in widget_in_dict.values():
            widget.destroy()
        widget_in_dict.clear()
        return widget_in_dict


######################################################
#  Menu Bar
#######################################################


class MenuBarFactory:
    """
    A class that represents a factory for creating menu bars in a GUI.

    Attributes:
        parent: The parent widget to which the menu bar will be added.
        menus: A dictionary that stores the created menus.
        default_state: The default state for menu items.

    Methods:
        create_menu: Creates a new menu with the specified name, commands, and state.
        update_menu_item_state: Updates the state of a menu item.

    """

    def __init__(self, parent: tk.Widget) -> None:
        """
        Initializes a new instance of the MenuBarFactory class.

        Parameters:
            parent: The parent widget to which the menu bar will be added.

        """
        self.parent = parent
        self.menus = {}
        self.default_state = tk.NORMAL

    def create_menu(self, menu_name: str, commands: dict, state: list = []) -> None:
        """
        Creates a new menu with the specified name, commands, and state.

        Parameters:
            menu_name (str): The name of the menu.
            commands (dict): A dictionary that maps menu item names to their corresponding commands.
            state (list, optional): The state of the menu items. Defaults to an empty list.

        """
        menu = tk.Menu(self.parent, tearoff=0)
        self.menus[menu_name] = menu

        if state:
            state_to_use = state
        else:
            state_to_use = self.default_state

        # Dynamically add menu items based on the commands dictionary
        for name, command in commands.items():
            if name == "separator":
                menu.add_separator()
            else:
                menu.add_command(label=name, command=command, state=state_to_use)

        # Add the menu to the parent menu bar
        self.parent.add_cascade(label=menu_name, menu=menu)

    def update_menu_item_state(
        self, menu_name: str, item_name: str, state: str
    ) -> None:
        """
        Updates the state of a menu item.

        Parameters:
            menu_name (str): The name of the menu.
            item_name (str): The name of the menu item.
            state (str): The new state of the menu item.

        """
        menu = self.menus.get(menu_name)
        if menu and menu.index(item_name) is not None:
            menu.entryconfig(item_name, state=state)
        else:
            print(f"Menu item '{item_name}' in menu '{menu_name}' not found")


######################################################
#  Console Window functions
######################################################


class StdoutRedirector(object):
    """
    A class that redirects the standard output to a text widget in a GUI.

    Parameters:
        text_widget (Tkinter.Text): The text widget to redirect the output to.

    Methods:
        write(text): Writes the given text to the text widget.
        flush(): Does nothing.
        position_next_to_main(main_window, offset=25): Positions the console window next to the main window.
        strip_ansi(text): Strips ANSI escape codes from the given text.
    """

    def __init__(self, text_widget: tk.Text) -> None:
        self.text_widget = text_widget
        self.original_stdout = sys.stdout

    def write(self, text: str) -> None:
        """
        Writes the given text to the text widget.

        Parameters:
            text (str): The text to write to the text widget.
        """
        # Check if the widget still exists
        if self.text_widget.winfo_exists():
            # Strip ANSI escape codes
            text = self.strip_ansi(text)

            self.text_widget.configure(state="normal")
            self.text_widget.insert("end", text)
            self.text_widget.configure(state="disabled")
            self.text_widget.see("end")

    def flush(self) -> None:
        pass

    def position_next_to_main(self, main_window: tk.Widget, offset: int = 25) -> None:
        """
        Positions the console window next to the main window.

        Parameters:
            main_window (Tkinter.Tk): The main window of the GUI.
            offset (int, optional): The offset in pixels between the main window and the console window.
        """

        # Get the Toplevel window containing the console
        console_window = self.text_widget.master.winfo_toplevel()
        console_window.update_idletasks()  # Update the window's layout

        main_x = main_window.winfo_x()
        main_y = main_window.winfo_y()
        main_width = main_window.winfo_width()

        console_x = main_x + main_width + offset
        console_y = main_y

        console_window.geometry(f"+{console_x}+{console_y}")

    @staticmethod
    def strip_ansi(text: str) -> str:
        """
        Strips ANSI escape codes from the given text.

        Parameters:
            text (str): The text to strip ANSI escape codes from.

        Returns:
            str: The text with ANSI escape codes stripped.
        """

        # Regular expression to match ANSI escape codes
        ansi_escape = re.compile(r"(?:\x1B[@-_]|[\x80-\x9F])[0-?]*[ -/]*[@-~]")
        return ansi_escape.sub("", text)


######################################################
#  GUI Utils class called up by BasicGUI
######################################################


class GUI_Utils:
    """
    A utility class for GUI-related functions and configurations.
    """

    def __init__(self) -> None:
        """
        Initializes an instance of the GUI_Utils class.
        """
        self.WidgetFactory = WidgetFactory
        self.MenuBarFactory = MenuBarFactory
        self.StdoutRedirector = StdoutRedirector

    @staticmethod
    def config_maker(**kwargs) -> dict:
        """
        Creates a configuration dictionary for GUI elements.

        Parameters:
            **kwargs: Keyword arguments representing the configuration options for the GUI element.

        Returns:
            dict: A dictionary containing the configuration options for the GUI element.

        Example:
            config = config_maker(text="Click Me", font=("Arial", 12), command=button_click)
        """
        config = {
            "text": kwargs.get("text"),
            "font": kwargs.get("font"),
            "command": kwargs.get("command"),
            "bg": kwargs.get("bg"),
            "fg": kwargs.get("fg"),
            "state": kwargs.get("state"),
            "textvariable": kwargs.get("textvariable"),
            "bindings": kwargs.get("bindings"),
            "from_": kwargs.get("from_"),
            "to": kwargs.get("to"),
            "orient": kwargs.get("orient"),
            "variable": kwargs.get("variable"),
            "justify": kwargs.get("justify"),
            "height": kwargs.get("height"),
            "width": kwargs.get("width"),
            "selectmode": kwargs.get("selectmode"),
            "exportselection": kwargs.get("exportselection"),
            "grid": {
                "row": kwargs.get("row"),
                "column": kwargs.get("column"),
                "columnspan": kwargs.get("columnspan"),
                "rowspan": kwargs.get("rowspan"),
                "sticky": kwargs.get("sticky"),
                "in_": kwargs.get("in_"),
                "padx": kwargs.get("padx"),
                "pady": kwargs.get("pady"),
            },
        }
        return config

    @staticmethod
    def create_canvas_plt(master, figsize, row, column, columnspan):
        """
        Create a canvas with a matplotlib figure and axes.

        Parameters:
            master (tk.Widget): The master widget (usually a tkinter.Tk or tkinter.Toplevel instance) to which the canvas will be added.
            figsize (tuple): A tuple specifying the size of the figure in inches.
            row (int): The row index in the grid where the canvas will be placed.
            column (int): The column index in the grid where the canvas will be placed.
            columnspan (int): The number of columns the canvas will span.

        Returns:
        - fig: The matplotlib figure object.
        - ax: The matplotlib axes object.
        - canvas: The tkinter canvas widget containing the figure.

        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)
        canvas = FigureCanvasTkAgg(fig, master=master)
        canvas.get_tk_widget().grid(
            row=row, column=column, columnspan=columnspan, sticky="nsew"
        )

        plt.close(fig)
        return fig, ax, canvas

    @staticmethod
    def add_toolbar(
        canvas_to_use: FigureCanvasTkAgg, fig_frame: tk.Frame
    ) -> NavigationToolbar2Tk:
        """
        Adds a toolbar to a given canvas within a figure frame.

        Parameters:
            canvas_to_use (FigureCanvasTkAgg): The canvas to which the toolbar will be added.
            fig_frame (tk.Frame): The frame containing the figure.

        Returns:
            NavigationToolbar2Tk: The created toolbar.

        """
        toolbar = NavigationToolbar2Tk(canvas_to_use, fig_frame)
        return toolbar

    @staticmethod
    def get_selected_cells(input_str: str, max_col: int) -> list[int]:
        """
        Parse the input string to get a list of selected columns/cells.

        Parameters:
            input_str (str): The input string representing the selected cells.
            max_col (int): The maximum number of columns.

        Returns:
            list: A list of selected columns/cells.

        Notes:
            - If the input string is "ALL" or "A", all cells are selected.
            - If the input string contains a hyphen "-", it is treated as a range of cells.
            - If the input string is a single number, it represents a single selected cell.
        """
        if input_str.upper() == "ALL" or input_str.upper() == "A":
            return list(range(max_col))
        elif "-" in input_str:
            start, end = map(int, input_str.split("-"))
            return list(range(start, min(end, max_col - 1) + 1))
        else:
            return [int(input_str)]

    @staticmethod
    def print_flush_statements(**kwargs) -> None:
        """
        Prints and flushes different statements based on the provided keyword arguments.

        Parameters:
            **kwargs: Keyword arguments indicating the type of statement to print and flush.
                Possible keyword arguments:
                    - RESET: If True, prints "Resetting GUI...".
                    - LOAD: If True, prints "Loading file...".
                    - SAVE_FIG: If True, prints "Saving file...".
                    - UPDATE_ASPAT: If True, prints "Updating A_Spatial image...".
                    - UPDATE_CTP: If True, prints "Updating CTP image...".
        """
        if kwargs.get("RESET"):
            msg = "Resetting GUI..."
        elif kwargs.get("LOAD"):
            msg = "Loading file..."
        elif kwargs.get("SAVE_FIG"):
            msg = "Saving file..."
        elif kwargs.get("UPDATE_ASPAT"):
            msg = "Updating A_Spatial image..."
        elif kwargs.get("UPDATE_CTP"):
            msg = "Updating CTP image..."
        # print flush state
        print(msg, end="", flush=True)

    @staticmethod
    def validate_entry(P: str, max_val: int) -> bool:
        """
        Validates an entry by checking if it is a digit and falls within the specified range.

        Parameters:
            P (str): The entry to be validated.
            max_val (int): The maximum allowed value for the entry.

        Returns:
            bool: True if the entry is valid, False otherwise.
        """
        if P.isdigit() and 0 <= int(P) <= max_val:
            return True
        elif P == "":
            return True  # Allow empty string, which can be interpreted as zero
        return False

    @staticmethod
    def plot_select_cell_ASpat(
        A_Spatial: csr_matrix | np.ndarray,
        cell_num: int,
        ax: plt.Axes,
        cmap: str,
        alpha: float = 0.5,
    ) -> None:
        """
        Select cells from A_Spatial matrix and reshape to 2D matrix

        Parameters:
        - A_Spatial: The input matrix from which cells are selected
        - cell_num: The index of the cell to be selected
        - ax: The matplotlib axis object on which the image will be plotted
        - cmap: The colormap to be used for the image
        - nlen: The length of the reshaped matrix (default: 256)
        - alpha: The transparency of the plotted image (default: 0.5)
        """
        # convert from sparse matrix to dense to list
        if isinstance(A_Spatial, (csr_matrix, spmatrix)):
            spatial_data = A_Spatial[:, cell_num].toarray()
        elif isinstance(A_Spatial, np.ndarray):
            spatial_data = A_Spatial[:, cell_num]
        else:
            raise ValueError(
                "A_Spatial must be a scipy.sparse.csr_matrix or np.ndarray"
            )
        # spatial_data = A_Spatial[:, cell_num].toarray().tolist()
        # remove any empty dict entries
        spatial_data = np.array([0 if x == {} else x for x in spatial_data])
        nlen = int(np.sqrt(len(spatial_data)))
        # convert back to array & reshape
        spatial_data = spatial_data.reshape(nlen, nlen)
        ax.imshow(spatial_data.T, cmap=cmap, alpha=alpha)

    @staticmethod
    def plot_select_cell_CTP(
        CTP: np.ndarray,
        pks: dict,
        cell_num: int,
        ax_CTP: plt.Axes,
        ax_pk_trace: plt.Axes,
        color_from_cmap: str,
        pre_peak: int = 50,
        post_peak: int = 200,
        lw: float = 0.5,
        markersize: int = 2,
        fs: int = 8,
        ymax: float = 1.1,
        ymin: float = -0.4,
    ):
        """
        Select cells from CTP matrix w/ pk overlay
        Posts pk trace in ax_pk_trace fig

        Parameters:
            CTP (np.ndarray): The CTP matrix containing the temporal data for all cells.
            pks (dict): A dictionary containing the peak information for each cell.
            cell_num (int): The index of the selected cell.
            ax_CTP (plt.Axes): The Axes object to plot the CTP.
            ax_pk_trace (plt.Axes): The Axes object to plot the peak trace.
            color_from_cmap (str): The color for the CTP plot.
            pre_peak (int, optional): The number of time points to include before the peak, default is 50.
            post_peak (int, optional): The number of time points to include after the peak, default is 200.
            lw (float, optional): The linewidth for the plots, default is 0.5.
            markersize (float, optional): The size of the markers for the peak points, default is 2.
            fs (int, optional): The fontsize for the plot titles, default is 8.
            ymax (float, optional): The maximum y-axis value for the plots, default is 1.1.
            ymin (float, optional): The minimum y-axis value for the plots, default is -0.4.

        Returns:
            None
        """
        # Extract CTP data for selected cell
        temporal_data = CTP[cell_num, :]
        # Find max CTP value
        max_CTP_val = np.max(temporal_data)
        # normalize to max
        temporal_data /= max_CTP_val
        # find pks for selected cell
        selected_pks = pks[f"seg{cell_num}"].astype(int)
        # set color for CTP plot
        color = rgb2hex(color_from_cmap)

        # plot CTP
        ax_CTP.plot(temporal_data, color=color, linewidth=lw)
        ax_CTP.set_xlim(0, len(temporal_data))

        # plot pks on CTP & plot pk trace pre & post in ax_pk_trace
        if selected_pks.any():
            ax_CTP.plot(
                selected_pks,
                temporal_data[selected_pks],
                marker="o",
                color=color_dict["violet"],
                markersize=markersize,
                linestyle="none",
            )
            for pk in selected_pks:
                pk_slice = slice(pk - pre_peak, pk + post_peak + 1)
                x_pk_idx = np.arange(-pre_peak, post_peak + 1)
                y_pk_trace = temporal_data[pk_slice]
                if len(x_pk_idx) > len(y_pk_trace):
                    x_pk_idx = x_pk_idx[: len(y_pk_trace)]
                ax_pk_trace.plot(
                    x_pk_idx,
                    y_pk_trace,
                    linewidth=lw,
                )
        ax_CTP.set_title(f"Ca2+ Transient Profile: Cell {cell_num}", fontsize=fs)
        ax_CTP.set_ylabel("Normalized to Max", fontsize=fs)
        ax_pk_trace.set_title(f"Pk Transient Profile: Cell {cell_num}", fontsize=fs)
        # ax_pk_trace.set_ylabel("Normalized to Max", fontsize=fs)
        ax_CTP.set_ylim(ymin, ymax)
        ax_pk_trace.set_ylim(ymin, ymax)

    @staticmethod
    def print_cell_num_plotted(default_str: str = "", **kwargs) -> None:
        """
        Prints the session number and selected cells range.

        Parameters:
            default_str (str): Default string to print if no selected cells or session number is provided.
            **kwargs: Additional keyword arguments.
                selected_cells (list): List of selected cells.
                session_num (int): Session number.

        Returns:
            None
        """
        selected_cells = kwargs.get("selected_cells")
        session_num = kwargs.get("session_num")
        sess_to_print = default_str
        cells_to_print = default_str
        if selected_cells:
            if len(selected_cells) > 1:
                cells_to_print = f"Cells {selected_cells[0]} - {selected_cells[-1]}"
            else:
                cells_to_print = f"Cell {selected_cells[0]}"

        if session_num:
            sess_to_print = f"Session {session_num}: "

        utils.print_wFrame(f"{sess_to_print}{cells_to_print}")

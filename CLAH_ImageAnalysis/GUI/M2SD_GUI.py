import sys
import os
import tkinter as tk
from cairosvg import svg2png
import io
from PIL import Image, ImageTk
import multiprocessing
from tkinter import ttk
from tkinter import filedialog
from ttkthemes import ThemedTk
from CLAH_ImageAnalysis.tifStackFunc.TSF_enum import Parser4M2SD
from CLAH_ImageAnalysis.utils import folder_tools
from CLAH_ImageAnalysis.utils import color_dict

# from tkinter import font


class MoCo2segDictGUI:
    def __init__(self, root):
        """Initialize the GUI"""

        self.root = root
        self.root.title("Motion Correction & Segmentation (M2SD)")

        self.setup_icon()

        # set terminal title
        # _set_terminal_title()

        self.color_lib = color_dict()

        # font settings
        self.setup_font()

        # store selected paths
        self.paths = []

        # Create main frame with padding
        self.create_main_frame()
        # Path selection frame
        self.path_selection_setup()
        # Create variables for all parameters using defaults from Parser4M2SD
        self.create_parameter_vars()
        # Create parameter sections
        # start with basic options
        current_row = self.create_main_options(start_row=1)
        # then advanced options
        current_row = self.create_advanced_options(start_row=current_row)
        # Create Run button
        self.create_run_btn(current_row=current_row)

        # Status
        self.create_status_bar(current_row=current_row)

    def setup_icon(self):
        """Setup the icon for the GUI"""
        # Get the path to the icon relative to this script
        current_dir = os.path.dirname(os.path.abspath(__file__))  # Gets GUI directory
        package_root = os.path.dirname(
            os.path.dirname(current_dir)
        )  # Go up two levels to CLAH_IA
        icon_path = os.path.join(package_root, "AppDesktopSettings", "M2SD_icon.svg")

        # Set the window icon
        if os.path.exists(icon_path):
            try:
                # Convert SVG to PNG in memory
                png_data = svg2png(url=icon_path, output_width=64, output_height=64)
                # Convert to PIL Image
                pil_img = Image.open(io.BytesIO(png_data))
                icon_photo = ImageTk.PhotoImage(pil_img)
                self.root.iconphoto(True, icon_photo)
            except Exception as e:
                print(f"Could not load icon: {e}")

    def setup_font(self):
        """Setup the font for the GUI"""

        def _get_system_font():
            """Determine font based on OS"""
            if sys.platform.startswith("win"):
                return "Helvetica"
            elif sys.platform.startswith("darwin"):
                return "Arial"
            else:
                # return "liberation mono"
                return "liberation mono"

        self.fontGUI = _get_system_font()
        self.fontSize = 18
        self.fontSize_small = 15
        self.root.tk.call("tk", "scaling", 1.5)

    def create_main_frame(self):
        """Create the main frame"""
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    def path_selection_setup(self):
        """Create the path selection section"""

        btn_width = 15

        path_frame = ttk.LabelFrame(
            self.main_frame, text="Directories of sessions to process", padding="5"
        )
        path_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)

        # Listbox for paths with scrollbar
        path_list_frame = ttk.Frame(path_frame)
        path_list_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E))

        self.path_listbox = tk.Listbox(path_list_frame, width=75, height=10)
        scrollbar = ttk.Scrollbar(
            path_list_frame, orient="vertical", command=self.path_listbox.yview
        )
        self.path_listbox.configure(yscrollcommand=scrollbar.set)

        self.path_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Path buttons
        path_btn_frame = ttk.Frame(path_frame)
        path_btn_frame.grid(row=0, column=2, padx=5)

        path_btns = [
            ("Add Directory", self.add_path),
            ("Remove Selected", self.remove_path),
            ("Clear All", self.clear_paths),
        ]

        for btn_text, btn_command in path_btns:
            ttk.Button(
                path_btn_frame,
                text=btn_text,
                command=btn_command,
                width=btn_width,
            ).pack(pady=2)

    def add_path(self):
        """Add a path to the list"""
        directory = filedialog.askdirectory()
        if directory and directory not in self.paths:
            self.paths.append(directory)
            self.path_listbox.insert(tk.END, directory)
            self.update_status()

    def remove_path(self):
        """Remove selected path from the list"""
        selection = self.path_listbox.curselection()
        if selection:
            index = selection[0]
            self.paths.pop(index)
            self.path_listbox.delete(index)
            self.update_status()

    def clear_paths(self):
        """Clear all paths"""
        self.paths.clear()
        self.path_listbox.delete(0, tk.END)
        self.update_status()

    def create_run_btn(self, current_row):
        """Create the run button"""
        self.run_btn = ttk.Button(
            self.main_frame,
            text="Run Analysis",
            command=self.run_analysis,
            state=tk.DISABLED,
            # style="Blue.TButton",
        )
        self.run_btn.grid(row=current_row, column=0, columnspan=3, pady=(10, 5))

    def create_status_bar(self, current_row):
        """Create the status bar"""
        self.status_var = tk.StringVar(value="Select at least one directory to start")
        ttk.Label(
            self.main_frame,
            textvariable=self.status_var,
        ).grid(row=current_row + 1, column=0, columnspan=3, pady=(5, 10))

    def update_status(self):
        """Update status bar and run button state based on number of selected paths"""
        num_paths = len(self.paths)
        if num_paths == 0:
            self.status_var.set("Select at least one directory to start")
            self.run_btn.configure(state=tk.DISABLED)
        else:
            self.run_btn.configure(state=tk.NORMAL)
            status_str = f"Ready to run - {num_paths} paths selected. Click 'Run Analysis' to start."
            if num_paths > 1:
                status_str += (
                    "\nNOTE - settings will be applied to all selected directories)"
                )
            self.status_var.set(status_str)

        self.root.update()

    def create_parameter_vars(self):
        """Create variables for all parameters using defaults from Parser4M2SD"""
        self.param_vars = {}
        for (param_name, _), param_info in Parser4M2SD.ARG_DICT.value.items():
            if param_info["TYPE"] == "bool":
                self.param_vars[param_name] = tk.BooleanVar(value=param_info["DEFAULT"])
            elif param_info["TYPE"] == "int" and "n_proc" not in param_name:
                self.param_vars[param_name] = tk.IntVar(
                    value=param_info["DEFAULT"]
                    if param_info["DEFAULT"] is not None
                    else 1
                )
            elif param_info["TYPE"] == "int" and "n_proc" in param_name:
                self.param_vars[param_name] = tk.IntVar(
                    value=param_info["DEFAULT"]
                    if param_info["DEFAULT"] is not None
                    else multiprocessing.cpu_count()
                )
        self.param_vars["process_all"] = tk.BooleanVar(value=False)

    def create_main_options(self, start_row):
        """Create the main options section"""

        ttk.Label(
            self.main_frame,
            text="Main Options",
            font=(self.fontGUI, self.fontSize, "bold"),
        ).grid(row=start_row, column=0, columnspan=3, sticky=tk.W, pady=(10, 5))

        main_params = [
            ("motion_correct", "Motion Correction (MC)"),
            (
                "segment",
                "Segment (SG): NOTE - Can be used without MC if MC was already performed in a previous run",
            ),
            (
                "process_all",
                "Process All Sessions (if unchecked, will prompt for user selection in the console)",
            ),
            (
                "overwrite",
                "Overwrite: NOTE - Only use this if you want to start analysis from scratch\n & delete all existing M2SD output files from previous run",
            ),
        ]

        row = start_row + 1
        for param_name, text2use in main_params:
            ttk.Checkbutton(
                self.main_frame,
                text=text2use,
                variable=self.param_vars[param_name],
            ).grid(row=row, column=0, sticky=tk.W)
            row += 1

        return row

    # def create_processor_options(self, start_row):
    #     """Create the processor options section"""
    #     ttk.Label(
    #         self.main_frame,
    #         text="Processor Settings",
    #         font=(self.fontGUI, self.fontSize, "bold"),
    #     ).grid(row=start_row, column=0, columnspan=3, sticky=tk.W, pady=(10, 5))

    #     row = start_row + 1
    #     for param_name, label in [
    #         ("n_proc4MOCO", "MOCO Processors:"),
    #         ("n_proc4CNMF", "CNMF Processors:"),
    #     ]:
    #         ttk.Label(self.main_frame, text=label).grid(row=row, column=0, sticky=tk.W)
    #         ttk.Entry(
    #             self.main_frame, textvariable=self.param_vars[param_name], width=5
    #         ).grid(row=row, column=1, sticky=tk.W)
    #         row += 1

    #     return row

    def create_advanced_options(self, start_row):
        """Create the advanced options section"""
        ttk.Label(
            self.main_frame,
            text="Advanced Options (ONLY USE IF YOU KNOW WHAT YOU ARE DOING)",
            font=(self.fontGUI, self.fontSize, "bold"),
            background=self.color_lib["darkcoral"],
        ).grid(row=start_row, column=0, columnspan=3, sticky=tk.W, pady=(10, 5))

        advanced_params = [
            (
                "concatenate",
                "Concatenate: Concatenate sessions if both sessions happened on the same day",
            ),
            (
                "prev_sd_varnames",
                "Prev SD Varnames: Saves variable names in segDict (SG output) using single letters like 'C' instead of 'C_Temporal'",
            ),
            (
                "compute_metrics",
                "Compute Metrics: Export metrics related to motion correction (will add time to analysis)",
            ),
            (
                "use_cropper",
                "Use Cropper: Use to crop dimensions of recording (only for 1photon/.isxd files)",
            ),
            (
                "separate_channels",
                "Separate Channels: Motion correct channels separately. Only applicable for 2photon data with 2 channels.",
            ),
        ]

        mc_iter_txt = "Number of MC iterations (Default: 1; each additional iteration will add time to analysis)"

        row = start_row + 1
        for param_name, text2use in advanced_params:
            ttk.Checkbutton(
                self.main_frame,
                text=text2use,
                variable=self.param_vars[param_name],
            ).grid(row=row, column=0, sticky=tk.W)
            row += 1

        # MC iterations
        ttk.Label(
            self.main_frame,
            text=mc_iter_txt,
        ).grid(row=row, column=0, sticky=tk.W)
        ttk.Entry(
            self.main_frame, textvariable=self.param_vars["mc_iter"], width=5
        ).grid(row=row, column=1, sticky=tk.W)
        row += 1

        return row

    def browse_path(self):
        directory = filedialog.askdirectory()
        if directory:
            self.path_var.set(directory)

    def run_analysis(self):
        from CLAH_ImageAnalysis.tifStackFunc import MoCo2segDict

        total_paths = len(self.paths)

        for p_idx, path in enumerate(self.paths):
            path_string_status = f"({p_idx + 1}/{total_paths})"
            # try:
            basename = folder_tools.get_basename(path)

            self.status_var.set(
                f"Running analysis for {basename} {path_string_status} - See console for progress/messages"
            )
            self.root.update()

            sess2process = "all" if self.param_vars["process_all"].get() else []

            # Create instance with GUI parameters
            moco = MoCo2segDict(
                path=path,
                sess2process=sess2process,
                **{name: var.get() for name, var in self.param_vars.items()},
            )

            # Run the analysis
            moco.run_whole_proc()

            self.status_var.set(
                f"Analysis complete for {basename} {path_string_status}"
            )
            # except Exception as e:
            #     self.status_var.set(f"Error: {str(e)}")
            #     self.root.update()
            #     print(f"Error: {str(e)}")
            #     sys.exit(1)

        self.status_var.set(
            "Analysis complete! Click 'Clear All' to start a new analysis or click the red 'X' to close the window"
        )
        self.root.update()


def main():
    def _set_terminal_title():
        title = "Motion Correction & Segmentation (M2SD)"
        # set terminal title
        print("\033]30;" + title + "\007", end="", flush=True)  # Konsole
        print("\33]0;" + title + "\a", end="", flush=True)  # other terminals

    _set_terminal_title()

    root = ThemedTk(theme="adapta")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    app = MoCo2segDictGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

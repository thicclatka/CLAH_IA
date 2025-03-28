import sys
import os
import tkinter as tk
from cairosvg import svg2png
import io
import getpass
from PIL import Image, ImageTk
import multiprocessing
from tkinter import ttk
from tkinter import filedialog
from ttkthemes import ThemedTk
import sqljobscheduler as sqljs
from pathlib import Path
from CLAH_ImageAnalysis.tifStackFunc.TSF_enum import Parser4M2SD
from CLAH_ImageAnalysis.utils import folder_tools
from CLAH_ImageAnalysis.utils import color_dict
from CLAH_ImageAnalysis.utils import paths
from CLAH_ImageAnalysis.utils import enum_utils
from CLAH_ImageAnalysis.tifStackFunc import TSF_enum


class MoCo2segDictGUI:
    def __init__(self, root):
        """Initialize the GUI"""

        self.root = root
        self.root.title("Motion Correction & Segmentation (M2SD)")

        self.user = getpass.getuser()

        self.setup_icon()

        # set terminal title
        # _set_terminal_title()

        self.color_lib = color_dict()

        self.strings2print = {
            "email_required": "Email address is required to queue jobs",
            "no_paths_selected": "Need to select at least one directory",
            "paths_selectedWnoemail": "{num_paths} path(s) selected. Still need email address to enable 'Run Analysis' button",
            "paths_selectedWemail": "Ready to run - {num_paths} path(s) selected.\nEmail: {email_address}\nClick 'Run Analysis' to add job to the queue.",
            "paths_selected_multiple": "NOTE - settings will be applied to all selected directories)",
        }

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

        # create field to add email address
        current_row = self.get_email_address(start_row=current_row)
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
        self.status_var = tk.StringVar(
            value=f"{self.strings2print['email_required']}\n{self.strings2print['no_paths_selected']}"
        )
        ttk.Label(
            self.main_frame,
            textvariable=self.status_var,
        ).grid(row=current_row + 1, column=0, columnspan=3, pady=(5, 10))

    def update_status(self):
        """Update status bar and run button state based on number of selected paths"""
        num_paths = len(self.paths)
        email_address = self.email_var.get()

        if not email_address or "@" not in email_address:
            str_status = self.strings2print["email_required"] + "\n"
        else:
            str_status = ""

        if num_paths == 0:
            self.status_var.set(
                f"{str_status}{self.strings2print['no_paths_selected']}"
            )
            self.run_btn.configure(state=tk.DISABLED)
        elif num_paths > 0:
            if not email_address or "@" not in email_address:
                status_str = self.strings2print["paths_selectedWnoemail"].format(
                    num_paths=num_paths
                )
                self.run_btn.configure(state=tk.DISABLED)
            else:
                status_str = self.strings2print["paths_selectedWemail"].format(
                    num_paths=num_paths, email_address=email_address
                )
                self.run_btn.configure(state=tk.NORMAL)
            if num_paths > 1:
                status_str += f"\n{self.strings2print['paths_selected_multiple']}"
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
        # self.param_vars["process_all"] = tk.BooleanVar(value=False)

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
            # (
            #     "process_all",
            #     "Process All Sessions (if unchecked, will prompt for user selection in the console)",
            # ),
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

    def open_mc_params_window(self):
        """Open the MC parameters window"""
        if not self.paths:
            self.status_var.set(self.strings2print["no_paths_selected"])
            return
        MCParamsWindow(self.root, self.paths, param_type="MC", onePhotonStatus=None)

    def open_cnmf_params_window(self):
        """Open the CNMF parameters window"""
        if not self.paths:
            self.status_var.set(self.strings2print["no_paths_selected"])
            return
        MCParamsWindow(self.root, self.paths, param_type="CNMF", onePhotonStatus=None)

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

        row = start_row + 1
        ttk.Button(
            self.main_frame,
            text="Adjust MC Parameters",
            command=self.open_mc_params_window,
        ).grid(row=row, column=0, sticky=tk.W)
        row += 1

        ttk.Button(
            self.main_frame,
            text="Adjust CNMF Parameters",
            command=self.open_cnmf_params_window,
        ).grid(row=row, column=0, sticky=tk.W)
        row += 1

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

        # row = start_row + 1
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
        # row += 1

        return row

    def get_email_address(self, start_row):
        """Create field to add email address"""
        row = start_row + 1
        # Add email field
        ttk.Label(
            self.main_frame,
            text="Email Address (for job notifications):",
        ).grid(row=row, column=0, sticky=tk.W)
        self.email_var = tk.StringVar()
        self.email_var.trace_add("write", self.on_email_change)
        ttk.Entry(self.main_frame, textvariable=self.email_var, width=30).grid(
            row=row, column=1, sticky=tk.W
        )
        row += 1

        return row

    def on_email_change(self, *args):
        """Update status bar when email address is changed"""
        self.update_status()

    def browse_path(self):
        directory = filedialog.askdirectory()
        if directory:
            self.path_var.set(directory)

    def run_analysis(self):
        try:
            sqlUtils = sqljs.JobQueue()
            M2SD_path = os.path.join(
                paths.get_code_dir_path("tifStackFunc"), "MoCo2segDict.py"
            )
            python_exec = str(paths.get_python_exec_path())
            email_address = self.email_var.get()

            for p_idx, path in enumerate(self.paths):
                parameters = {
                    "path": path,
                    "sess2process": "all",
                    **{name: var.get() for name, var in self.param_vars.items()},
                }
                sqlUtils.add_job(
                    programPath=M2SD_path,
                    path2python_exec=python_exec,
                    parameters=parameters,
                    email_address=email_address,
                    user=self.user,
                    python_env=paths.get_python_env(),
                )

            self.status_var.set(
                "Analysis added to the queue. Click 'Clear All' to start a new analysis to queue or click the red 'X' to close the window"
            )
            self.root.update()
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            self.root.update()
            print(f"Error: {str(e)}")
            sys.exit(1)


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


class MCParamsWindow:
    def __init__(self, parent, paths, param_type="MC", onePhotonStatus=None) -> None:
        """
        Create a new window for adjusting motion correction parameters.

        Parameters:
            parent: Parent window
            paths: List of paths to process
            onePhotonStatus: Dict mapping paths to their one-photon status, or None to auto-detect
        """
        self.window = tk.Toplevel(parent)
        self.param_type = param_type

        self.title = (
            f"{'Motion Correction' if param_type == 'MC' else 'CNMF'} Parameters (2p)"
        )
        self.window.title(self.title)

        self.paths = paths
        self.onePhotonStatus = onePhotonStatus or self._detect_imaging_type()

        # Create main frame
        self.main_frame = ttk.Frame(self.window, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Create parameter variables for each path
        self.param_vars = {}
        self._create_param_vars()

        # Create the UI
        self._create_header()
        self._create_parameter_table()
        self._create_control_buttons()

    def _detect_imaging_type(self):
        """Auto-detect if each path is one-photon or two-photon"""
        status = {}
        for path in self.paths:
            # Add your detection logic here
            # For now, defaulting to False (2-photon)
            status[path] = False
        return status

    def _create_param_vars(self):
        """Create parameter variables for each path"""
        for path in self.paths:
            is_onep = self.onePhotonStatus[path]
            # Initialize with default values based on imaging type
            self.param_vars[path] = {}

            if self.param_type == "MC":
                reference_enum = (
                    TSF_enum.MOCO_Params4OnePhoton if is_onep else TSF_enum.MOCO_Params
                )
            else:  # CNMF
                reference_enum = (
                    TSF_enum.CNMF_Params_1p if is_onep else TSF_enum.CNMF_Params
                )

            reference_dict = enum_utils.enum2dict(reference_enum)

            for param_name in reference_dict.keys():
                self.param_vars[path][param_name] = tk.StringVar(
                    value=str(reference_dict[param_name])
                )

    def _create_header(self):
        """Create the header row with parameter names"""
        ttk.Label(self.main_frame, text="Path").grid(row=0, column=0, sticky=tk.W)

        # Get parameter names from first path (they're all the same)
        first_path = list(self.param_vars.keys())[0]
        params = list(self.param_vars[first_path].keys())

        for col, param in enumerate(params, 1):
            ttk.Label(self.main_frame, text=param).grid(row=0, column=col)

    def _create_parameter_table(self):
        """Create the table of parameters"""
        for row, path in enumerate(self.paths, 1):
            # Path label
            ttk.Label(self.main_frame, text=os.path.basename(path)).grid(
                row=row, column=0, sticky=tk.W
            )

            # Parameter entries
            for col, (param_name, var) in enumerate(self.param_vars[path].items(), 1):
                ttk.Entry(self.main_frame, textvariable=var, width=10).grid(
                    row=row, column=col
                )

    def _create_control_buttons(self):
        """Create control buttons"""
        button_frame = ttk.Frame(self.main_frame)
        button_frame.grid(
            row=len(self.paths) + 1,
            column=0,
            columnspan=len(self.param_vars[self.paths[0]]) + 1,
            pady=10,
        )

        ttk.Button(
            button_frame, text="Toggle 1P/2P Defaults", command=self._toggle_defaults
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            button_frame, text="Save Parameters", command=self._save_parameters
        ).pack(side=tk.LEFT, padx=5)

    def _toggle_defaults(self):
        """Toggle between 1P and 2P defaults for all paths"""
        for path in self.paths:
            self.onePhotonStatus[path] = not self.onePhotonStatus[path]

            if self.param_type == "MC":
                reference_enum = (
                    TSF_enum.MOCO_Params4OnePhoton
                    if self.onePhotonStatus[path]
                    else TSF_enum.MOCO_Params
                )
            else:  # CNMF
                reference_enum = (
                    TSF_enum.CNMF_Params_1p
                    if self.onePhotonStatus[path]
                    else TSF_enum.CNMF_Params
                )

            reference_dict = enum_utils.enum2dict(reference_enum)

            for param_name, value in reference_dict.items():
                self.param_vars[path][param_name].set(str(value))

        if "2p" in self.title:
            self.title = self.title.replace("2p", "1p")
        else:
            self.title = self.title.replace("1p", "2p")

        self.window.title(self.title)
        self.window.update()

    def _save_parameters(self):
        """Save parameters for each path"""
        for path in self.paths:
            os.chdir(path)
            # Convert string values back to appropriate types
            params = {}
            for param_name, var in self.param_vars[path].items():
                value = var.get()
                # Convert string to appropriate type
                if value.lower() == "true":
                    params[param_name] = True
                elif value.lower() == "false":
                    params[param_name] = False
                elif value.lower() == "none":
                    params[param_name] = None
                else:
                    try:
                        params[param_name] = int(value)
                    except ValueError:
                        try:
                            params[param_name] = float(value)
                        except ValueError:
                            params[param_name] = value

            sess_folders = [
                os.path.abspath(d) for d in os.listdir() if os.path.isdir(d)
            ]
            sess_folders.sort()
            # Generate and export parameters
            for sf in sess_folders:
                os.chdir(sf)
                TSF_enum.generateNexport_user_set_params(
                    onePhotonCheck=self.onePhotonStatus[path],
                    param_type=self.param_type,
                    **params,
                )
            os.chdir(path)

        self.window.destroy()


if __name__ == "__main__":
    main()

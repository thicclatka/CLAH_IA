import customtkinter as ctk
import tkinter as tk
import multiprocessing
from CLAH_ImageAnalysis.tifStackFunc.TSF_enum import Parser4M2SD
from CLAH_ImageAnalysis.utils import folder_tools


class MoCo2segDictGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Motion Correction & Segmentation (M2SD)")

        # Set theme
        ctk.set_appearance_mode("dark")  # Options: "dark", "light"
        ctk.set_default_color_theme("blue")  # Options: "blue", "green", "dark-blue"

        self.fontGUI = "liberation mono"
        self.fontSize = 12

        # store selected paths
        self.paths = []

        # Create main frame with padding
        self.main_frame = ctk.CTkFrame(self.root, corner_radius=10)
        self.main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # Path selection frame
        path_frame = ctk.CTkFrame(self.main_frame)
        path_frame.grid(row=0, column=0, columnspan=3, sticky="ew", pady=5)

        ctk.CTkLabel(path_frame, text="Directories of sessions to process").grid(
            row=0, column=0, sticky="w", padx=5
        )

        # Listbox for paths with scrollbar
        path_list_frame = ctk.CTkFrame(path_frame)
        path_list_frame.grid(row=1, column=0, columnspan=2, sticky="ew")

        self.path_listbox = tk.Listbox(path_list_frame, width=50, height=4)
        scrollbar = ctk.CTkScrollbar(path_list_frame, command=self.path_listbox.yview)
        self.path_listbox.configure(yscrollcommand=scrollbar.set)

        self.path_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Path buttons
        path_btn_frame = ctk.CTkFrame(path_frame)
        path_btn_frame.grid(row=1, column=2, padx=5)

        ctk.CTkButton(path_btn_frame, text="Add Path", command=self.add_path).pack(
            pady=2
        )
        ctk.CTkButton(
            path_btn_frame, text="Remove Selected", command=self.remove_path
        ).pack(pady=2)
        ctk.CTkButton(path_btn_frame, text="Clear All", command=self.clear_paths).pack(
            pady=2
        )

        # Create variables for all parameters using defaults from Parser4M2SD
        self.create_parameter_vars()

        # Create parameter sections
        current_row = self.create_main_options(start_row=1)
        current_row = self.create_advanced_options(start_row=current_row)

        # Run button
        self.run_btn = ctk.CTkButton(
            self.main_frame,
            text="Run Analysis",
            command=self.run_analysis,
            state="disabled",
        )
        self.run_btn.grid(row=current_row, column=0, columnspan=3, pady=(10, 5))

        # Status
        self.status_var = ctk.StringVar(value="Select at least one directory to start")
        ctk.CTkLabel(self.main_frame, textvariable=self.status_var).grid(
            row=current_row + 1, column=0, columnspan=3, pady=(5, 10)
        )

    def add_path(self):
        """Add a path to the list"""
        directory = ctk.filedialog.askdirectory()
        if directory and directory not in self.paths:
            self.paths.append(directory)
            self.path_listbox.insert(ctk.END, directory)
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
        self.path_listbox.delete(0, ctk.END)
        self.update_status()

    def update_status(self):
        """Update status bar and run button state based on number of selected paths"""
        num_paths = len(self.paths)
        if num_paths == 0:
            self.status_var.set("Select at least one directory to start")
            self.run_btn.configure(state="disabled")
        else:
            self.status_var.set(
                f"Ready to run - {num_paths} paths selected. Click 'Run Analysis' to start."
            )
            self.run_btn.configure(state="normal")

        self.root.update()

    def create_parameter_vars(self):
        """Create variables for all parameters using defaults from Parser4M2SD"""
        self.param_vars = {}
        for (param_name, _), param_info in Parser4M2SD.ARG_DICT.value.items():
            if param_info["TYPE"] == "bool":
                self.param_vars[param_name] = ctk.BooleanVar(
                    value=param_info["DEFAULT"]
                )
            elif param_info["TYPE"] == "int" and "n_proc" not in param_name:
                self.param_vars[param_name] = ctk.IntVar(
                    value=param_info["DEFAULT"]
                    if param_info["DEFAULT"] is not None
                    else 1
                )
            elif param_info["TYPE"] == "int" and "n_proc" in param_name:
                self.param_vars[param_name] = ctk.IntVar(
                    value=param_info["DEFAULT"]
                    if param_info["DEFAULT"] is not None
                    else multiprocessing.cpu_count()
                )
        self.param_vars["process_all"] = ctk.BooleanVar(value=False)

    def create_main_options(self, start_row):
        """Create the main options section"""
        ctk.CTkLabel(
            self.main_frame,
            text="Main Options",
            font=(self.fontGUI, self.fontSize, "bold"),
        ).grid(row=start_row, column=0, columnspan=3, sticky="w", pady=(10, 5))

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
            ctk.CTkCheckBox(
                self.main_frame,
                text=text2use,
                variable=self.param_vars[param_name],
                checkbox_height=20,
                checkbox_width=20,
                corner_radius=3,
                text_color="white",
                checkmark_color="white",
            ).grid(row=row, column=0, sticky="w")
            row += 1

        return row

    def create_advanced_options(self, start_row):
        """Create the advanced options section"""
        ctk.CTkLabel(
            self.main_frame,
            text="Advanced Options (ONLY USE IF YOU KNOW WHAT YOU ARE DOING)",
            font=(self.fontGUI, self.fontSize, "bold"),
        ).grid(row=start_row, column=0, columnspan=3, sticky="w", pady=(10, 5))

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
        ]

        mc_iter_txt = "Number of MC iterations (Default: 1; each additional iteration will add time to analysis)"

        row = start_row + 1
        for param_name, text2use in advanced_params:
            ctk.CTkCheckBox(
                self.main_frame,
                text=text2use,
                variable=self.param_vars[param_name],
            ).grid(row=row, column=0, sticky="w")
            row += 1

        # MC iterations
        ctk.CTkLabel(self.main_frame, text=mc_iter_txt).grid(
            row=row, column=0, sticky="w"
        )
        ctk.CTkEntry(
            self.main_frame, textvariable=self.param_vars["mc_iter"], width=75
        ).grid(row=row, column=1, sticky="w")
        row += 1

        return row

    def run_analysis(self):
        from CLAH_ImageAnalysis.tifStackFunc import MoCo2segDict

        total_paths = len(self.paths)

        for p_idx, path in enumerate(self.paths):
            path_string_status = f"({p_idx + 1}/{total_paths})"
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

        self.status_var.set(
            "Analysis complete! Click 'Clear All' to start a new analysis or click the red 'X' to close the window"
        )
        self.root.update()


def main():
    root = ctk.CTk()
    app = MoCo2segDictGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

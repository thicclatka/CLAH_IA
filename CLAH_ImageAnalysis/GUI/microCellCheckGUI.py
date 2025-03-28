import os
import tkinter as tk
from tkinter import filedialog
from ttkthemes import ThemedTk
import matplotlib.pyplot as plt
import easygui
from matplotlib.colors import ListedColormap, rgb2hex
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from CLAH_ImageAnalysis.utils import load_file, text_dict, color_dict

text_lib = text_dict()
file_tag = text_lib["file_tag"]
dict_name = text_lib["dict_name"]
color_dict = color_dict()
blue = color_dict["blue"]
red = color_dict["red"]
green = color_dict["green"]
GUI_str = text_lib["GUI"]


root = ThemedTk()
root.set_theme("breeze")
root.title(GUI_str["GT"])

# init global variables
column_width = 10
col_entry = {}
col_range_labels = {}
col_sess_label = {}


def destroy_widget(widget_in_dict: dict) -> dict:
    """
    Destroy all widgets in the given dictionary and clear the dictionary.
    """
    for widget in widget_in_dict.values():
        widget.destroy()
    widget_in_dict.clear()
    return widget_in_dict


# Load File
def load_file4GUI() -> None:
    """
    Load a file and setup the GUI.
    """
    egu_msg = GUI_str["EGU_MSG"]
    pkl_fname = easygui.fileopenbox(msg=egu_msg, default=f"*{file_tag['PKL']}")
    global multSessSegStruc
    multSessSegStruc = load_file(pkl_fname, file_tag["PKL"])
    setup_gui()

    global file_id
    file_id = os.path.basename(pkl_fname).split("_")[0]
    load_label.config(text=GUI_str["LOAD_ID"].format(file_id))
    update_button.config(
        text=GUI_str["UPDATE"],
        command=update_image,
        bg="white",
        fg="black",
        state=tk.NORMAL,
    )
    # turn on save fig option in menu
    filemenu.entryconfig("Save Figure", state=tk.NORMAL)


def setup_gui() -> None:
    """
    Setup the GUI.
    """
    # global column_width
    # column_width = len(multSessSegStruc.keys()) * 2
    global cmaps
    cmaps = [
        ListedColormap(["none", red]),
        ListedColormap(["none", blue]),
        ListedColormap(["none", green]),
    ]

    # init global col_ars
    global col_vars, col_entry, col_range_labels, col_sess_label
    # Create a dictionary to store col_var for each numSess
    col_vars = {}
    col_entry = {}
    col_range_labels = {}
    col_sess_label = {}
    # Create dropdown menu for col selection for each numSess
    for i, numSess in enumerate(multSessSegStruc.keys()):
        max_col = multSessSegStruc[numSess]["A_SPATIAL"].shape[1] - 1

        # Create a label for the entry
        cmap = cmaps[i % len(cmaps)]
        color = cmap(0.5)  # Get color from color map
        color = rgb2hex(color)  # Convert color to hex format

        sess_label = tk.Label(root, text=f"Session {i+1}", bg=color)
        column = (i + 1) + i * 2
        sess_label.grid(row=2, column=column)

        # Create a label for range
        range_label = tk.Label(root, text=f"Range: 0 - {max_col}")
        range_label.grid(row=3, column=column)

        col_var = tk.StringVar(root)
        col_var.set("0")  # default value
        entry = tk.Entry(root, textvariable=col_var)
        entry.grid(row=4, column=column)  # Arrange entries in the second row

        # Bind Enter key to update_image function
        entry.bind("<Return>", lambda event: update_image())

        col_entry[numSess] = entry
        col_vars[numSess] = col_var
        col_range_labels[numSess] = range_label
        col_sess_label[numSess] = sess_label


# reset app
def reset_gui() -> None:
    """
    Reset the GUI.
    """
    global col_entry, col_range_labels, col_sess_label
    # clear axes & figure
    ax.clear()
    canvas.draw()
    # clear multSessSegStruc
    global multSessSegStruc
    multSessSegStruc = None
    # clear session labels
    # destroy session labels & entry widgets
    col_entry = destroy_widget(col_entry)
    col_range_labels = destroy_widget(col_range_labels)
    col_sess_label = destroy_widget(col_sess_label)
    # reset col_vars
    for var in col_vars.values():
        var.set(0)
    load_label.config(text=GUI_str["LOAD_ID_EMPTY"])
    update_button.config(
        text=GUI_str["UNUSABLE"], bg="black", fg="white", command=[], state=tk.DISABLED
    )
    # turn off save fig option in menu
    filemenu.entryconfig("Save Figure", state=tk.DISABLED)


# Update update_image function to use col_vars
def update_image() -> None:
    """
    Update the image display based on the selected columns.
    """
    # Clear axes
    ax.clear()

    # Set background color to black
    ax.set_facecolor("black")

    title = GUI_str["PLT_TITLE"].format(file_id) + "\n"
    # Get selected cols and display images for each numSess
    for i, (numSess, col_var) in enumerate(col_vars.items()):
        # Get A_Spatial array for numSess
        A_Spatial = multSessSegStruc[numSess]["A_SPATIAL"]

        if col_var.get().upper() == "ALL" or col_var.get().upper() == "A":
            # If input is "all" or "ALL", set cols to full range of columns
            cols = list(range(A_Spatial.shape[1]))
        elif "-" in col_var.get():
            # Split col_var by "-" and convert to integers
            start, end = map(int, col_var.get().split("-"))

            # Ensure end does not exceed number of columns
            end = min(end, A_Spatial.shape[1] - 1)

            # Get all columns from start to end
            cols = list(range(start, end + 1))
        else:
            # Convert col_var to integer
            cols = [int(col_var.get())]

        for col in cols:
            image_data = A_Spatial[:, col].reshape(256, 256)

            cmap = cmaps[i % len(cmaps)]
            # Display image
            # transpose image_data to match orientation of image
            ax.imshow(image_data.T, cmap=cmap, alpha=0.5)  # Set alpha for overlay

        # automate title addition
        if len(cols) > 1:
            title = (
                title
                + "| "
                + GUI_str["PLT_CELLS"].format(i + 1, cols[0], cols[-1])
                + " "
            )
        else:
            title = title + "| " + GUI_str["PLT_CELL"].format(i + 1, cols[0]) + " "

    title = title + "|"
    ax.set_title(title, fontsize=10)
    canvas.draw()


def save_figure() -> None:
    """
    Save the current figure to a file.
    """
    # Open a dialog for file saving
    file_path = filedialog.asksaveasfilename(defaultextension=".png")

    if file_path:
        fig.savefig(file_path)


# Create label to display loaded file ID
load_label = tk.Label(root, text=GUI_str["LOAD_ID_EMPTY"])
load_label.grid(row=0, column=0)


# Button to update images
update_button = tk.Button(
    root,
    text=GUI_str["UNUSABLE"],
    command=[],
    bg="black",
    fg="white",
    state=tk.DISABLED,
)
update_button.grid(row=5, column=0, columnspan=column_width)


# Matplotlib figure to display images
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.set_facecolor("black")
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().grid(row=6, column=0, columnspan=column_width)

reset_button = tk.Button(
    root, text=GUI_str["RESET"], command=reset_gui, bg=red, fg="white"
)
reset_button.grid(row=7, column=0, columnspan=column_width)

# Create a menu bar
menubar = tk.Menu(root)

# Create a File menu
filemenu = tk.Menu(menubar, tearoff=0)
filemenu.add_command(label=GUI_str["LOAD"], command=load_file4GUI)
filemenu.add_command(label=GUI_str["SAV_FIG"], command=save_figure)
filemenu.add_command(label=GUI_str["QUIT"], command=root.quit)
menubar.add_cascade(label=GUI_str["MENU"], menu=filemenu)

# disable save figure until file is loaded
filemenu.entryconfig(GUI_str["SAV_FIG"], state=tk.DISABLED)

# Create an Edit menu
editmenu = tk.Menu(menubar, tearoff=0)
editmenu.add_command(label=GUI_str["RESET"], command=reset_gui)
menubar.add_cascade(label=GUI_str["EDIT"], menu=editmenu)

# Display the menu
root.config(menu=menubar)


root.mainloop()

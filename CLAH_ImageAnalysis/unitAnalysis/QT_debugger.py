import os
import sys
from rich import print
from CLAH_ImageAnalysis.unitAnalysis import quickTuning
from CLAH_ImageAnalysis.utils.text_formatting import text_dict


os.system("clear")

text_lib = text_dict()
breakers = text_lib["breaker"]
hash = breakers["hash"]

path_FW = "/mnt/MegaDrive/Practice_Data"
path_HS2 = "/mnt/DataDrive1/alex/Data/Data_Clay/"

# Check if the directories exist and select the appropriate path
if os.path.exists(path_FW):
    correct_path = path_FW
elif os.path.exists(path_HS2):
    correct_path = path_HS2
else:
    print("Neither path exists")
    sys.exit()

working_on = [
    "2cues_CA3",
    "2odor_DG",
    "3cue_DG",
    "Alzheimers_multiWeek",
    "CA3",
    "DG-CA3-wBeh",
    "DG-CA3-wBeh2use",
    "SingleSess",
    "agedDock10_Mar24",
    "nonAged_Dk_WT_MS",
    "testing_ridge",
    "miniscope_wTDML",
]

while True:
    print(hash)
    print("ENTERING DEBUG MODE")
    print(f"Current path: {correct_path}")
    print("Available folders to debug script on:")
    for i, option in enumerate(working_on, 1):
        print(f"{i}: {option}")

    selection = (
        int(
            input(
                "(Note: typing 0 will quit the program)\nEnter the number of the folder you want to process: "
            )
        )
        - 1
    )

    # if I type 0, it will lead to -1, which will exit the program
    if selection == -1:
        print("Exiting program...")
        sys.exit()

    # Validate the selection
    if 0 <= selection < len(working_on):
        debug_path = f"{correct_path}/{working_on[selection]}"
        print(f"Running script on: {debug_path}")
        print(hash)
        print("\n")
        QT = quickTuning(path=debug_path)
        QT.run()
        break
    else:
        print("Invalid selection")
        print(
            "Process will repeat until a valid selection is made or 0 is entered to quit\n"
        )

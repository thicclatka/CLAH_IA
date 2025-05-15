from itertools import chain
import numpy as np
import inquirer
import re
from CLAH_ImageAnalysis.utils import create_multiline_string
from CLAH_ImageAnalysis.utils import folder_tools
from CLAH_ImageAnalysis.utils import ProcessStatusPrinter
from CLAH_ImageAnalysis.utils import text_dict
from CLAH_ImageAnalysis.utils import debug_utils
from CLAH_ImageAnalysis.utils import print_wFrame
from CLAH_ImageAnalysis.utils import file_finder

######################################################
#  util class function
######################################################


class subj_selector_utils:
    def __init__(
        self,
        dayPath: str,
        dayDir: list,
        file_of_interest: str,
        pre_sel: list,
        select_by_ID: bool,
        **kwargs,
    ):
        """Initialize the utility class for subject selection.

        Parameters:
            dayPath (str): Path to the day's data.
            dayDir (list): List of directories within the day's path.
            file_of_interest (str): Type of file of interest (e.g., 'H5', 'MSS').
            pre_sel (list): List of pre-selected folders.
        """

        # need this to differentiate between picking all or picking by ID for CSS
        self.select_by_ID = select_by_ID

        # initiate some text_dict values
        self.text_lib = text_dict()
        self.file_tag = self.text_lib["file_tag"]
        self.sel_tag = self.text_lib["selector"]["tags"]
        self.subj_sel_str = self.text_lib["selector"]["strings"]
        self.breaker = self.text_lib["breaker"]["hash"]
        self.breaker_lean = self.text_lib["breaker"]["lean"]

        self.what_selected = self.subj_sel_str["what_selected"]
        self.note1 = self.subj_sel_str["note_info_1"]
        self.note2 = self.subj_sel_str["note_dash_2"]
        self.note3 = self.subj_sel_str["note_dash_3"]
        self.note_all = self.subj_sel_str["note_all"]
        self.note_emc = self.subj_sel_str["note_allnoemc"]
        self.note_end = self.subj_sel_str["note_exit"]
        self.input_str = self.subj_sel_str["input_str"]
        self.wrong_sel = self.subj_sel_str["wrong_sel"]

        self.dayPath = dayPath
        self.dir = dayDir
        self.dirByNum = list(range(1, len(self.dir) + 1))
        self.file_of_interest = file_of_interest
        self.Y_int = f"Y_{self.file_of_interest}"
        self.N_int = f"N_{self.file_of_interest}"
        self.bool_dict = {}
        self.sess_dict = {
            "ID": [],
            "DATE": [],
            "TYPE": [],
        }
        self.IDDict = {}
        self.all4CSS = False
        self.file_types = self.text_lib["selector"]["file_types"]
        self.singleSess = self.text_lib["selector"]["singleSess"]
        self.multiSess = self.text_lib["selector"]["multiSess"]
        self.ChoiceBank = self.text_lib["selector"]["choices"]
        self.conditionsAll = {}
        tags = ["EMC", "TDML", "SD", "CI", "H5", "MSS", "ISXD"]

        if not self.select_by_ID:
            tags.extend(["CSS"])

        for tag in tags:
            self.conditionsAll[f"Y_{tag}"] = [
                f"Y_{self.sel_tag[tag]}",
                f"fold_{tag.lower()}",
            ]
            self.conditionsAll[f"N_{tag}"] = [
                f"N_{self.sel_tag[tag]}",
                f"fold_{tag.lower()}",
                f"fold_need_{tag.lower()}",
            ]

        self.file_check = {key: {"Yes": [], "No": []} for key in self.file_types}
        self.processAll = []
        self.processAll_noemc = []
        self.processAll_just_emc = []

        self.noTDML4SD = kwargs.get("noTDML4SD", False)

        if pre_sel:
            self._preSelection(pre_sel)

    def _preSelection(self, pre_sel: list) -> list:
        """
        Pre-selects folders based on user input.

        Parameters:
            pre_sel (list): The pre-selected folders.

        Returns:
            list: The selected folders.
        """

        self.find_eligible_files()
        self.print_eligible_file_results()
        pre_sel_msg = "\nPre-selection was made!"
        if isinstance(pre_sel, int):
            pre_sel = [pre_sel]
        elif isinstance(pre_sel, tuple):
            pre_sel = list(chain(*pre_sel))
            self._preSelection(pre_sel)
        elif isinstance(pre_sel, str):
            if pre_sel.lower() == "all":
                pre_sel = self.processAll
            else:
                pre_sel = pre_sel.split(",")
                pre_sel = self._dash_utility(pre_sel)

        if pre_sel == self.processAll:
            pre_sel_msg += " All available eligible files were selected."

        self.selected_folders = pre_sel
        print(pre_sel_msg)
        self._print_sess_forLoop()

    def find_eligible_files(self) -> None:
        """Find eligible files based on the specified file of interest."""
        FOI = self.file_of_interest
        for sess_num, sess in enumerate(self.dir, start=1):
            if sess == "~GroupData":
                continue

            current_sess = f"{self.dayPath}/{sess}"

            date, subj_id, etype = self._parse_sessInfo(sess)
            self._update_sess_dict(date, subj_id, etype)

            for key, file_type in self.file_types.items():
                _, status = file_finder(
                    current_sess, file_type, notInclude=self._get_exclusions(key)
                )
                self.file_check[key][status].append(sess_num)

        self._fill_bool_dict()
        self._create_IDDict() if self.select_by_ID else None

    def _create_IDDict(self) -> None:
        """Organize session and multi-session dictionaries for CSS selection."""
        FOI = self.file_of_interest
        for key in ["DATE", "ID", "TYPE"]:
            self.sess_dict[key] = np.array(self.sess_dict[key])

        self.ID = np.unique(self.sess_dict["ID"])

        for id in self.ID:
            self.IDDict[id] = {}
            subj_idx = np.argwhere(self.sess_dict["ID"] == id).flatten()
            if len(subj_idx) > 1:
                for key in ["DATE", "TYPE"]:
                    self.IDDict[id][key] = self.sess_dict[key][subj_idx]
                self.IDDict[id]["IDX"] = subj_idx + 1
                self.IDDict[id][f"ALL_{FOI.upper()}"] = all(
                    idx in self.bool_dict[f"Y_{FOI}"] for idx in subj_idx + 1
                )
            else:
                self.IDDict[id]["DATE"] = self.sess_dict["DATE"][subj_idx]
                self.IDDict[id]["TYPE"] = self.sess_dict["TYPE"][subj_idx]
                self.IDDict[id]["IDX"] = subj_idx + 1
                self.IDDict[id][f"ALL_{FOI.upper()}"] = (
                    subj_idx + 1 in self.bool_dict[f"Y_{FOI}"]
                )

    def _fill_bool_dict(self) -> None:
        """Populate the boolean dictionary based on the file check results."""

        for key in self.file_types:
            self.bool_dict[f"Y_{key}"] = self.file_check[key]["Yes"]
            self.bool_dict[f"N_{key}"] = self.file_check[key]["No"]

    def _parse_sessInfo(self, sess: str) -> tuple[str | None, str, str]:
        """Parse session string to extract date, id, and type.

        Parameters:
            sess (str): The session string to parse.

        Returns:
            tuple: A tuple containing the date, id, and type.
        """

        split_check = len(sess.split("_"))
        if split_check > 2:
            date = sess.split("_")[0]
            subj_id = sess.split("_")[1]
            # etype here is the experiment
            etype = sess.split("_")[-1]
        elif split_check == 2:
            date = None
            subj_id = sess.split("_")[0]
            # etype here is numSess
            etype = sess.split("_")[-1]

        return date, subj_id, etype

    def _update_sess_dict(
        self, date: str | None = None, id: str | None = None, etype: str | None = None
    ) -> None:
        """
        Update session dictionary with parsed values.

        Parameters:
            date (str | None): The date of the session.
            id (str | None): The ID of the session.
            etype (str | None): The type of the session.
        """
        self.sess_dict["DATE"].append(date) if date is not None else None
        self.sess_dict["ID"].append(id) if id is not None else None
        self.sess_dict["TYPE"].append(etype) if etype is not None else None

    def _get_exclusions(self, key: str) -> list:
        """Get list of file types to exclude during file search.

        Parameters:
            key (str): The key to get exclusions for.

        Returns:
            list: The list of file types to exclude.
        """
        return (
            [self.file_tag["EMC"], self.file_tag["SQZ"]]
            if key == self.sel_tag["H5"]
            else []
        )

    def print_eligible_file_results(self) -> None:
        """Print results of eligible files based on the file of interest."""
        print(self.breaker)
        FOI = self.file_of_interest
        if FOI == self.sel_tag["EMC"]:
            condkeys2use = self._eligibleFileSetup4EMC()
        elif FOI == self.sel_tag["SD"]:
            condkeys2use = self._eligibleFileSetup4SD()
        elif FOI == self.sel_tag["MSS"]:
            condkeys2use = ["Y_MSS", "N_MSS"]
            self.eligible_folders = self.bool_dict["Y_MSS"]
        elif FOI == self.sel_tag["CI"]:
            condkeys2use = ["Y_CI", "N_CI"]
            self.eligible_folders = self.bool_dict["Y_CI"]
        elif FOI == self.sel_tag["CSS"]:
            condkeys2use = ["Y_CSS", "N_CSS"]
            self.eligible_folders = self.bool_dict["Y_CSS"]

        if self.select_by_ID:
            # special case for CSS
            self._eligibleFileSetup4selectionByID(condkeys2use)

        # if FOI != self.sel_tag["CSS"]:
        #     self.processAll = self.eligible_folders
        # else:
        #     self.processAll = self._flatten_list(
        #         self._remove_empty_entries(self.eligible_folders)
        #     )
        self.processAll = self.eligible_folders
        self._extractEligibleFiles_fromCondDictNprintResults(condkeys2use)

    def _extractEligibleFiles_fromCondDictNprintResults(
        self, condkeys2use: list
    ) -> None:
        """
        Extracts eligible files from the condition dictionary and prints the results.

        Parameters:
            condkeys2use (list): A list of condition keys to use for extraction.
        """
        messages = []
        for condition, tags in self.conditionsAll.items():
            if condition in condkeys2use:
                if "Y" == condition.split("_")[0]:
                    fold_str = self.subj_sel_str[tags[1]].format("") + " {}"
                    need_str = None
                elif "N" == condition.split("_")[0]:
                    fold_str = self.subj_sel_str[tags[1]].format("NO") + " {}"
                    need_str = self.subj_sel_str[tags[2]].format("MISSING")
                nums = [num for num in self.dirByNum if num in self.bool_dict[tags[0]]]
                if nums:
                    messages.append(fold_str.format(nums))
                    messages.append(need_str) if need_str is not None else None
        file_msgs = [msg for msg in messages if "MISSING" not in msg]
        need_msgs = [msg for msg in messages if "MISSING" in msg]
        for msg in file_msgs + need_msgs:
            print(msg)

    @staticmethod
    def _create_condkeys2use(tags_needed: list) -> list:
        """
        Creates a list of condition keys to use based on the given tags.

        Parameters:
            tags_needed (list): A list of tags needed.

        Returns:
            list:   A list of condition keys to use, which includes 'Y_' followed by each tag
                    in `tags_needed`, and 'N_' followed by each tag in `tags_needed`.
        """
        return [f"Y_{tag}" for tag in tags_needed] + [f"N_{tag}" for tag in tags_needed]

    def _eligibleFileSetup4EMC(self) -> list:
        """
        Sets up the eligible folders for EMC analysis.

        This method creates a list of eligible folders based on the conditions specified by the user.
        It uses the boolean dictionary to determine the folders that meet the conditions for both EMC and H5.
        It also sets up a special case for EMC where the processAll_noemc list is populated with folders that meet the condition for H5 but not EMC.

        Returns:
            condkeys2use (list): A list of condition keys to be used for further processing.
        """

        if self.bool_dict[f"Y_{self.sel_tag['H5']}"]:
            # if H5 is present, use H5 (2 photon)
            boolDict2use = self.bool_dict[f"Y_{self.sel_tag['H5']}"]
            key2use = "H5"
        else:
            # if H5 is not present, use ISXD (1 photon)
            boolDict2use = self.bool_dict[f"Y_{self.sel_tag['ISXD']}"]
            key2use = "ISXD"

        condkeys2use = self._create_condkeys2use([key2use, "EMC"])

        Y_b2u_set = set(boolDict2use)
        Y_EMC_set = set(self.bool_dict[f"Y_{self.sel_tag['EMC']}"])

        self.eligible_folders = list(Y_EMC_set.union(Y_b2u_set))
        self.processAll_noemc = list(Y_b2u_set - Y_EMC_set)
        self.processAll_just_emc = list(Y_EMC_set)

        return condkeys2use

    def _eligibleFileSetup4SD(self) -> list:
        """
        Sets up the eligible files for SD analysis based on the selected tags.

        Returns:
            list: The list of condition keys to use for SD analysis.
        """
        if not self.noTDML4SD:
            condkeys2use = self._create_condkeys2use(["SD", "TDML", "EMC"])
        else:
            condkeys2use = self._create_condkeys2use(["SD"])

        Y_SD_set = set(self.bool_dict[f"Y_{self.sel_tag['SD']}"])

        if not self.noTDML4SD:
            Y_TDML_set = set(self.bool_dict[f"Y_{self.sel_tag['TDML']}"])
            self.eligible_folders = list(Y_SD_set & Y_TDML_set)
        else:
            self.eligible_folders = list(Y_SD_set)

        return condkeys2use

    def _eligibleFileSetup4selectionByID(self, condkeys2use: list) -> None:
        """Print information about eligible cueShiftStruc (CSS) files."""
        first_key = condkeys2use[0]
        parts = first_key.split("_")
        ft2use = parts[1]

        if ft2use not in ["CSS", "SD"]:
            raise ValueError(f"Invalid file type: {ft2use}")

        ID_key = self.IDDict.keys()
        init_empty = [[] for _ in range(len(ID_key))]
        self.FT_ID = {"Y": [], "N": []}
        self.bool_dict[f"Y_{ft2use}_ALL"] = init_empty.copy()
        self.bool_dict[f"N_{ft2use}_ALL"] = init_empty.copy()

        for idx, id in enumerate(self.ID):
            sess_arr = self.IDDict[id]["IDX"]
            date_arr = self.IDDict[id]["DATE"]
            str2fill = "["
            for sess, date in zip(sess_arr, date_arr):
                str2fill += f" {sess:02} ({date}),"
            str2fill = str2fill.rstrip(",") + "]"
            if self.IDDict[id][f"ALL_{ft2use}"]:
                key2use = "Y"
            else:
                key2use = "N"
            self.FT_ID[key2use].append(f"{id:<5} {str2fill}")
            self.bool_dict[f"{key2use}_{ft2use}_ALL"][idx] = sess_arr

        yes_output = "none" if not self.FT_ID["Y"] else str(self.FT_ID["Y"])
        no_output = "none" if not self.FT_ID["N"] else str(self.FT_ID["N"])

        output = [(yes_output, "yes", "Y"), (no_output, "no", "N")]

        for out, ssskey, csskey in output:
            if out == "none":
                print(f"{self.subj_sel_str[f'subj_{ssskey}_{ft2use}_all']} {out}")
            else:
                print(self.subj_sel_str[f"subj_{ssskey}_{ft2use}_all"])
                for idx, idNarr in enumerate(self.FT_ID[csskey]):
                    print_wFrame(f"{idNarr}")

        # raise error if no CSS files are found
        if not self.FT_ID["Y"] and not self.FT_ID["N"]:
            debug_utils.raiseVE_SysExit1(self.subj_sel_str[f"subj_need_mult{ft2use}"])

        # adjust eligible_folders for CSS to be list by session num vs group/ID num
        self.eligible_folders = [
            item
            for sublist in self.bool_dict[f"Y_{ft2use}_ALL"]
            if isinstance(sublist, np.ndarray)
            for item in sublist.tolist()
        ]

    # def _eligibleFileSetup4CSS(self) -> None:
    #     """Print information about eligible cueShiftStruc (CSS) files."""
    #     if self.select_by_ID:
    #         ID_key = self.IDDict.keys()
    #         init_empty = [[] for _ in range(len(ID_key))]
    #         self.CSS_ID = {"Y": [], "N": []}
    #         self.bool_dict["Y_CSS_ALL"] = init_empty.copy()
    #         self.bool_dict["N_CSS_ALL"] = init_empty.copy()

    #         for idx, id in enumerate(self.ID):
    #             sess_arr = self.IDDict[id]["IDX"]
    #             date_arr = self.IDDict[id]["DATE"]
    #             str2fill = "["
    #             for sess, date in zip(sess_arr, date_arr):
    #                 str2fill += f" {sess:02} ({date}),"
    #             str2fill = str2fill.rstrip(",") + "]"
    #             if self.IDDict[id]["ALL_CSS"]:
    #                 key2use = "Y"
    #             else:
    #                 key2use = "N"
    #             self.CSS_ID[key2use].append(f"{id:<5} {str2fill}")
    #             self.bool_dict[f"{key2use}_CSS_ALL"][idx] = sess_arr

    #         yes_output = "none" if not self.CSS_ID["Y"] else str(self.CSS_ID["Y"])
    #         no_output = "none" if not self.CSS_ID["N"] else str(self.CSS_ID["N"])

    #         output = [(yes_output, "yes", "Y"), (no_output, "no", "N")]

    #         for out, ssskey, csskey in output:
    #             if out == "none":
    #                 print(f"{self.subj_sel_str[f'subj_{ssskey}_CSS_all']} {out}")
    #             else:
    #                 print(self.subj_sel_str[f"subj_{ssskey}_CSS_all"])
    #                 for idx, idNarr in enumerate(self.CSS_ID[csskey]):
    #                     print_wFrame(f"{idNarr}")

    #         # raise error if no CSS files are found
    #         if not self.CSS_ID["Y"] and not self.CSS_ID["N"]:
    #             debug_utils.raiseVE_SysExit1(self.subj_sel_str["subj_need_multCSS"])

    #         # adjust eligible_folders for CSS to be list by session num vs group/ID num
    #         self.eligible_folders = [
    #             item
    #             for sublist in self.bool_dict["Y_CSS_ALL"]
    #             if isinstance(sublist, np.ndarray)
    #             for item in sublist.tolist()
    #         ]

    def MakeChoice_printSelectionResults(self) -> list:
        """
        Makes a selection based on the selection tag and eligible folders,
        adjusts the choice bank, and prints the selected folders.

        Returns:
            list: The selected folders.
        """
        # based on selection tag & eligible folders, will adjust choice bank from which selection is made
        self._determine_choices2use()
        print(self.breaker_lean)
        # make selection with user input
        self._makeSelection()
        self._processSelection()
        self._printSelectedFolders()
        print(self.breaker_lean, "\n")

        # return selection to outside class
        return self.selected_folders

    def _determine_choices2use(self) -> None:
        """Determines the choice bank based on the selection tags and process options."""

        def append_user_selection(choice_list: list) -> list:
            """Appends 'User selection' with the correct number to the given choice list.

            Parameters:
                choice_list (list): The choice list to append the user selection to.

            Returns:
                list: The updated choice list with the user selection appended.
            """

            next_number = len(choice_list) + 1
            choice_list.append(f"{next_number}| User selection")
            return choice_list

        check_dict = {}
        emc_check = bool(self.processAll_just_emc)

        SD2use = "SD" if not self.noTDML4SD else "SD_NO_TDML"

        choiceKey_conditon = [
            (
                "EMC_ABS",
                lambda tag: tag == self.sel_tag["EMC"] and not emc_check,
            ),
            (
                "EMC_PRES",
                lambda tag: tag == self.sel_tag["EMC"] and emc_check,
            ),
            ("MULTI", lambda tag: tag in self.multiSess),
            (SD2use, lambda tag: tag == self.sel_tag["SD"]),
            ("CSSbyID", lambda tag: tag == self.sel_tag["CSS"] and self.select_by_ID),
            ("CSS", lambda tag: tag == self.sel_tag["CSS"] and not self.select_by_ID),
        ]

        for choice_key, condition in choiceKey_conditon:
            check_dict[choice_key] = condition(self.file_of_interest)

        for key, value in check_dict.items():
            if value:
                if key == "EMC_PRES" and emc_check:
                    # Format the entries of the "EMC_PRES" list with self.processAll_noemc and self.processAll
                    self.ChoiceBank[key][0] = self.ChoiceBank[key][0].format(
                        self.processAll_noemc
                    )
                    self.ChoiceBank[key][1] = self.ChoiceBank[key][1].format(
                        self.processAll
                    )
                else:
                    # Format the first entry of the selected choice list with self.processAll
                    self.ChoiceBank[key][0] = self.ChoiceBank[key][0].format(
                        self.processAll
                    )
                self.choices2use = append_user_selection(self.ChoiceBank[key])

    def _makeSelection(self) -> None:
        """Prompt the user to select a file.

        This method displays a prompt to the user, asking them to select a file from a list of choices.
        The selected file(s) is stored in the `user_selection` attribute.
        """
        self.ques = [
            inquirer.List(
                "file2proc",
                message=self.subj_sel_str["which_sess"],
                choices=self.choices2use,
            ),
        ]
        self.user_selection = inquirer.prompt(self.ques)

    def _processSelection(self) -> None:
        """Process the user's selection."""

        user_sel = self.user_selection["file2proc"]
        if len(self.choices2use) == 2:
            # all selection
            if user_sel == self.choices2use[0]:
                self._handle_all_selection_2Choices()
            # user selection
            elif user_sel == self.choices2use[-1]:
                self._handle_user_selection()
        elif len(self.choices2use) == 3:
            # this is only for if eMC is present when eMC is selected
            # all selection but no emc
            if user_sel == self.choices2use[0]:
                print(self.subj_sel_str["all_selected_noemc"])
                self.selected_folders = self.processAll_noemc
            # all selection but with emc
            elif user_sel == self.choices2use[1]:
                print(self.subj_sel_str["all_selected_yesemc"])
                self.selected_folders = self.processAll
            # user selection
            elif user_sel == self.choices2use[-1]:
                self._handle_user_selection()
        # need to do this for CSS selection to create CSS_IDX for later exporting
        if self.select_by_ID:
            self._process_selection_by_ID_sel()

    def _handle_all_selection_2Choices(self) -> None:
        """Handle selection when 'All' is chosen."""
        FOI = self.file_of_interest
        if FOI == self.sel_tag["EMC"]:
            print(self.subj_sel_str["all_selected_noemc"])
        else:
            if not self.noTDML4SD:
                FOI2use = FOI.lower()
            elif FOI == self.sel_tag["SD"] and self.noTDML4SD:
                FOI2use = f"{FOI.lower()}_no_tdml"
            print(self.subj_sel_str[f"all_selected_{FOI2use}"])
        self.selected_folders = self.processAll

    def _handle_user_selection(self) -> None:
        """Handle selection when user manually selects files."""
        # print user selection header
        # this is also where user input is save as self.selected_folders
        self._print_UserSelectionHeader()

        # set up masks for case when all or allnoemc is input
        all_case = self.selected_folders == "all"
        allnoemc_case = (
            self.selected_folders == "allnoemc"
            and self.file_of_interest == self.sel_tag["EMC"]
        )
        if all_case:
            self.selected_folders = self.processAll
        elif allnoemc_case:
            self.selected_folders = self.processAll_noemc
        else:
            if len(self.selected_folders) == 1:
                if self.file_of_interest == self.sel_tag["CSS"] and self.select_by_ID:
                    self.selected_folders = self.bool_dict[
                        f"Y_{self.sel_tag['CSS']}_ALL"
                    ][int(self.selected_folders) - 1]
                    self.selected_folders = self._flatten_list(self.selected_folders)
                else:
                    self.selected_folders = [int(self.selected_folders)]
            elif len(self.selected_folders) > 1:
                sf_input = self.selected_folders.split(",")
                self.selected_folders = self._dash_utility(sf_input)

    def _process_selection_by_ID_sel(self) -> None:
        """Process selection specifically for CSS ID selection."""
        if len(self.selected_folders) == 1:
            self.idx2findID = [int(self.selected_folders)]
        else:
            self.idx2findID = self.selected_folders

        # creates self.CSS_ID_selected
        self._extract_IDs_from_sess()

    def _print_UserSelectionHeader(self) -> None:
        """
        Prints the header for user selection.

        This method prints the header for user selection, displaying the eligible subjects/sessions to select.
        It iterates through the eligible folders and prints them in a group or normal list way, depending on the type of session.
        It also prints additional notes and details specific to CSS selection.
        """
        FOI = self.file_of_interest
        print(self.breaker_lean)
        print("Eligible subjects/sessions to select:")
        if self.select_by_ID:
            for id in self.ID:
                if self.IDDict[id][f"ALL_{FOI}"]:
                    print(f"--{id}:")
                    for sess in self.IDDict[id]["IDX"]:
                        self._print_sessNum_fname(sess)
        else:
            for sess in self.eligible_folders:
                # Print in normal list way
                self._print_sessNum_fname(sess)
        notelist2print = [
            f"\n{self.note1}",
            self.note2,
            self.note3,
            self.note_all,
            self.note_end,
        ]
        if FOI == self.sel_tag["EMC"]:
            notelist2print.insert(-1, self.note_emc)

        notes2print = create_multiline_string(notelist2print)
        print(notes2print)
        # # need to print a detail for CSS selection given how it is organized
        # if FOI == self.sel_tag["CSS"]:
        #     print(self.subj_sel_str["CSS_DETAIL"])

        # store input from user of what folders that are selected
        self.selected_folders = input(self.input_str)

        # if 0, exit, or quit input, program quits
        self._abort_check()

    def _printSelectedFolders(self) -> None:
        """
        Print the folders that were selected by the user.

        If an incorrect selection is made, it will note it and restart the process.
        If there are ineligible folders selected, it will print a message and show the eligible file results.
        Otherwise, it will print the selected folder information.
        """
        # given how CSS is selected and organized, need to handle eligibility of choice differenly
        # otherwise eligible will just be self.eligible_folders
        if self.file_of_interest == self.sel_tag["CSS"]:
            eligible = self.processAll
        else:
            eligible = self.eligible_folders
        ineligible_folders = [
            folder for folder in self.selected_folders if folder not in eligible
        ]
        if ineligible_folders:
            print(self.wrong_sel.format(ineligible_folders))
            self.print_eligible_file_results()
            # restarting process
            self._processSelection()
        # otherwise, script progresses & prints selected folder info
        self._print_sess_forLoop()

    def _print_sess_forLoop(self) -> None:
        """
        Prints the selected folders and their corresponding directories.

        This method prints the value of `self.what_selected` and then iterates over the `self.selected_folders` list.
        For each session in the list, it prints the session number and its corresponding directory from the `self.dir` list.

        Example:
            what_selected = 'example'
            selected_folders = [1, 2, 3]
            dir = ['dir1', 'dir2', 'dir3']

            Output:
            example
            1| - dir1
            2| - dir2
            3| - dir3
        """
        print(self.what_selected)
        for sess in self.selected_folders:
            self._print_sessNum_fname(sess)
        print()

    def _print_sessNum_fname(self, sess: int) -> None:
        """Print the session number and its corresponding directory.

        Parameters:
            sess (int): The session number to print.
        """

        print(f"{sess:02d}| - {self.dir[sess - 1]}")

    def _extract_IDs_from_sess(self) -> None:
        """Extract IDs from sessions."""
        self.ID_selected = []
        for sess in self.idx2findID:
            for id in self.ID:
                if sess in self.IDDict[id]["IDX"]:
                    self.ID_selected.append(id)
        self.ID_selected = list(set(self.ID_selected))
        self.ID_selected.sort()

    def export_IDDictVars(self) -> dict:
        """Export Multi-Session Variables after filtering based on selection.

        Returns:
            dict: The filtered ID dictionary.
        """

        def filt_vals_via_selection(values: list, indices: list) -> list:
            """Filter values based on selected indices.

            Parameters:
                values (list): The values to filter.
                indices (list): The indices to filter by.

            Returns:
                list: The filtered values.
            """
            return [
                values[i]
                for i, idx in enumerate(indices)
                if idx in self.selected_folders
            ]

        filtered_IDDict = {
            k: {
                "DATE": filt_vals_via_selection(v["DATE"], v["IDX"]),
                "TYPE": filt_vals_via_selection(v["TYPE"], v["IDX"]),
                "IDX": filt_vals_via_selection(v["IDX"], v["IDX"]),
            }
            for k, v in self.IDDict.items()
            if k in self.ID_selected
        }
        return filtered_IDDict

    @staticmethod
    def _flatten_list(list_to_flat: list) -> list:
        """Flatten a nested list.

        Parameters:
            list_to_flat (list): List potentially containing nested lists.

        Returns:
            list: The flattened list.
        """
        flattened_list = []
        for item in list_to_flat:
            # Check if the item is iterable (but not a string)
            if isinstance(item, (list, np.ndarray)):
                # Extend the flattened list with the iterable item
                flattened_list.extend(item)
            else:
                # Append non-iterable items directly
                flattened_list.append(item)
        return flattened_list

    @staticmethod
    def _dash_utility(input: list) -> list:
        """Convert a range specified with dashes into a list of numbers.

        Parameters:
            input (list): List of strings potentially containing dashes to denote ranges.

        If selected input contains dashes will work, otherwise dash_holder will be empty
        """
        dash_holder = []
        non_dash_entries = []

        for entry in input:
            if "-" in entry:
                start, end = map(int, entry.split("-"))
                dash_holder.extend(range(start, end + 1))
            else:
                non_dash_entries.append(int(entry))

        output = non_dash_entries + dash_holder

        # sort output
        output.sort()

        return output

    @staticmethod
    def _remove_empty_entries(list_to_change: list) -> list:
        """Remove empty entries from a list.

        Parameters:
            list_to_change (list): List from which empty entries are to be removed.

        Returns:
            list: The list with empty entries removed.
        """
        list_to_output = [arr for arr in list_to_change if len(arr) > 0]
        return list_to_output

    def _abort_check(self) -> None:
        """
        Checks if the selected folders are None or match the exit regex pattern.
        If so, prints a user abort message and exits the program.
        """
        if self.selected_folders is None or re.match(
            self.text_lib["REGEX"]["EXIT"],
            str(self.selected_folders),
        ):
            debug_utils.raiseVE_SysExit1(self.subj_sel_str["user_abort"])


######################################################
#  main function
######################################################
def subj_selector(
    dayPath: str, dayDir: list, file_of_interest: str, **kwargs
) -> tuple[list, dict]:
    """Main function for subject selection.

    Parameters:
        path (str): Path to the day's data.
        dir (list): List of directories within the day's path.
        file_of_interest (str): Type of file of interest (e.g., 'H5', 'MSS').
        **kwargs: Additional keyword arguments.
            selection_made (bool): Whether selection was made.
            select_by_ID (bool): Whether to select by ID.

    Returns:
        tuple: A tuple containing the selected folders and the ID dictionary.
    """
    selection_made = kwargs.get("selection_made", False)
    select_by_ID = kwargs.get("select_by_ID", False)
    noTDML4SD = kwargs.get("noTDML4SD", False)

    SSU = subj_selector_utils(
        dayPath=dayPath,
        dayDir=dayDir,
        file_of_interest=file_of_interest,
        pre_sel=selection_made,
        select_by_ID=select_by_ID,
        noTDML4SD=noTDML4SD,
    )

    if not selection_made:
        # dict full of whether there are h5s, eMC, or segDicts in folder
        SSU.find_eligible_files()

        # start selector
        # printing eligible files based on file of interest
        SSU.print_eligible_file_results()

        # this does multiple things:
        # - create file results for selection
        # - show results
        # - take in selection
        # - process selection & print results
        selected_folders = SSU.MakeChoice_printSelectionResults()
    else:
        selected_folders = SSU.selected_folders

    # sel_tag = SSU.sel_tag
    IDDict = None
    if select_by_ID:
        IDDict = SSU.export_IDDictVars()

    return selected_folders, IDDict


def overwrite_selector(
    file_check: bool, file_list: list, init_prompt: str, user_confirm_prompt: str
) -> bool:
    """
    Selects whether to overwrite files or use existing files based on user confirmation.

    Parameters:
        file_check (bool): Indicates whether files need to be checked.
        file_list (list): List of files to be checked.
        init_prompt (str): Initial prompt to be displayed.
        user_confirm_prompt (str): Prompt to ask for user confirmation.

    Returns:
        bool: True if overwrite is selected, False otherwise.
    """
    if file_check:
        print(f"{init_prompt}")
        folder_tools.print_folder_contents(file_list)

        overwrite = ProcessStatusPrinter.get_user_confirmation(
            prompt=user_confirm_prompt
        )

        if overwrite:
            selection_str = "Selection to overwite files was saved"
        else:
            selection_str = "Selection to use existing files was saved."
        print(f"{selection_str} Applying process to all files.")
        return overwrite

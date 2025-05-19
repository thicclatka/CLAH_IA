"""
This module contains utility functions and classes for extracting data from TDML files and creating the treadBehDict dictionary.

Classes:
- TDML_extractor: A class that extracts settings and lap data from TDML files.
- setup_vars: A class that sets up the variables for the treadBehDict dictionary.

Functions:
- _determine_lap_type: Determines the type of lap based on the valves and locations.
- _determine_ctype: Determines the cue type based on the valve.
- _init_lapDict_entry: Initializes an entry in the lapDict dictionary.

"""

import numpy as np

from CLAH_ImageAnalysis.behavior import behavior_utils
from CLAH_ImageAnalysis.core import BaseClass as BC
from CLAH_ImageAnalysis.dependencies import runmean

######################################################
#  vars for keys & subkeys for treadBehDict & tdml file
######################################################


#############################################################
# Extractors
#############################################################
class TDML_extractor(BC):
    """
    A class that extracts information from TDML data and populates a treadBehDict.

    Args:
        TDMLsettings (object): An instance of the TDML_settings class.

    Attributes:
        TDMLsettings (object): An instance of the TDML_settings class.

    Methods:
        settings(data_dict, treadBehDict): Finds settings within tdml_data and updates the treadBehDict.
        lapDict(tdml_data): Finds lap types, valves, odors, cues, context, etc., fills lapDict, and saves corresponding mat.
    """

    def __init__(self, TDMLkey: dict, TDMLsettings: object) -> None:
        self.program_name = "TDML2tBD"
        self.class_type = "utils"
        BC.__init__(self, self.program_name, mode=self.class_type)

        self.file_tag = self.text_lib["file_tag"]
        self.dict_name = self.text_lib["dict_name"]

        # initiate TDMLkey & TDMLsettings
        self.TDMLkey = TDMLkey
        self.TDMLsettings = TDMLsettings
        self._unpack_TDMLkey()

    def _unpack_TDMLkey(self) -> None:
        """
        Unpacks the TDMLkey dictionary and assigns the values to corresponding attributes.

        This method extracts the values from the TDMLkey dictionary and assigns them to the
        corresponding attributes of the class instance. The TDMLkey dictionary contains various
        keys representing different settings, contexts, events, and other parameters used in
        the TDML2tBD utility.

        Note:
        - The TDMLkey dictionary should be initialized before calling this method.

        """
        self.SETTINGS = self.TDMLkey["SETTINGS_KEY"]
        self.SETTINGS_KEY = self.TDMLkey["SETTINGS_SUBKEY"]
        self.SETTINGS_VALVE = self.TDMLkey["SETTINGS_VALVE"]
        self.CONTEXT = self.TDMLkey["CONTEXT_KEY"]
        self.CONTEXT_VNAME = self.TDMLkey["CONTEXT_VNAME"]
        self.RFID_VNAME = self.TDMLkey["RFID_VNAME"]
        self.EVENT_VALVE = self.TDMLkey["EVENT_VALVE"]
        self.CUE1 = self.TDMLkey["CUE1"]
        self.CUE2 = self.TDMLkey["CUE2"]
        self.UNK = self.TDMLkey["UNK"]
        self.BC1 = self.TDMLkey["BC1"]
        self.BC2 = self.TDMLkey["BC2"]
        self.BOTH = self.TDMLkey["BOTH"]
        self.BOTH_SWITCH = self.TDMLkey["BOTH_SWITCH"]
        self.ID = self.TDMLkey["CONTEXT_SUBKEY"]
        self.LOC = self.TDMLkey["LOC"]
        self.ODOR = self.TDMLkey["ODOR"]
        self.TYPE = self.TDMLkey["TYPE"]
        self.DEC = self.TDMLkey["DEC"]
        self.PIN = self.TDMLkey["PIN"]
        self.TIME = self.TDMLkey["TIME"]
        self.TIME_KEY = self.TDMLkey["TIME_KEY"]
        self.TIME_NANO = self.TDMLkey["TIME_NANO"]
        self.POS = self.TDMLkey["POSITION"]
        self.POS_KEY = self.TDMLkey["POSITION_KEY"]
        self.LAP = self.TDMLkey["LAP_KEY"]
        self.LAP_NUM = self.TDMLkey["LAP_NUM_KEY"]
        self.COUNT = self.TDMLkey["COUNT_KEY"]
        self.REW = self.TDMLkey["REWARD"]
        self.REWZONE = self.TDMLkey["REW_ZONE"]
        self.ACTION = self.TDMLkey["ACTION"]
        self.LAP_LIST = self.TDMLkey["LAP_LIST"]
        self.START = self.TDMLkey["START"]
        self.STOP = self.TDMLkey["STOP"]
        self.OPEN = self.TDMLkey["OPEN"]
        self.CLOSE = self.TDMLkey["CLOSE"]
        self.LICK = self.TDMLkey["LICK"]
        self.SESSINFO = self.TDMLkey["SESSINFO"]
        self.NAME = self.TDMLkey["NAME_KEY"]
        self.TONE = self.TDMLkey["TONE"]
        self.TRIG = self.TDMLkey["TRIG"]
        self.SYNC = self.TDMLkey["SYNC"]
        self.CUE1_VALVE = self.TDMLkey["CUE1_VALVE"]
        self.CUE2_VALVE = self.TDMLkey["CUE2_VALVE"]
        self.LED_VALVE = self.TDMLkey["LED_VALVE"]
        self.LED = self.TDMLkey["LED"]
        self.TONE_VALVE = self.TDMLkey["TONE_VALVE"]
        self.REW_VALVE = self.TDMLkey["REW_VALVE"]
        self.CONTEXT1 = self.TDMLkey["CONTEXT1"]
        self.CONTEXT2 = self.TDMLkey["CONTEXT2"]
        self.CUETYPE = self.TDMLkey["CUETYPE"]
        self.CONTEXT_3C = self.TDMLkey["CONTEXT_3C"]
        self.CONTEXT_3C_TC = self.TDMLkey["CONTEXT_3C_TC"]
        self.CONTEXT_3C_TL = self.TDMLkey["CONTEXT_3C_TL"]
        self.CONTEXT_3C_CL = self.TDMLkey["CONTEXT_3C_CL"]
        self.TCL = self.TDMLkey["TCL"]
        self.CL = self.TDMLkey["CL"]
        self.TL = self.TDMLkey["TL"]
        self.TC = self.TDMLkey["TC"]
        self.SCENT_CONTEXT = [
            self.TDMLkey["S1C1"],
            self.TDMLkey["S1C2"],
            self.TDMLkey["S2C1"],
            self.TDMLkey["S2C2"],
        ]
        self.OPTO = self.TDMLkey["OPTO"]
        self.OPTO_VALVE = self.TDMLkey["OPTO_VALVE"]
        self.LAPSYNC_VALVE = self.TDMLkey["LAPSYNC_VALVE"]

    def settings(self, data_dict: dict, treadBehDict: dict) -> dict:
        """
        Finds settings within tdml_data and updates the treadBehDict.

        Args:
            data_dict (dict): The dictionary containing the TDML data.
            treadBehDict (dict): The treadBehDict to be updated.

        Returns:
            dict: The updated treadBehDict.
        """
        # Check for presence of settings & update class instance accordingly
        if self.SETTINGS in data_dict:
            settings = data_dict[self.SETTINGS]
            self.TDMLsettings.update_settings(settings)

        # Similarly, update session info if present
        if self.SESSINFO in data_dict:
            sess_info = data_dict[self.SESSINFO]
            self.TDMLsettings.update_sess_info(sess_info)

        # Incorporate updated settings into your treadBehDict
        treadBehDict[self.TDMLkey["SYNCPIN"]] = self.TDMLsettings.syncPin
        treadBehDict[self.TDMLkey["LEDPIN"]] = self.TDMLsettings.ledPin
        treadBehDict[self.SESSINFO] = self.TDMLsettings.sessInfo

        return treadBehDict

    def lapDict(self, tdml_data: dict) -> tuple:
        """
        Finds lap types, valves, odors, cues, context, etc., fills lapDict, and saves corresponding mat.

        Args:
            tdml_data (dict): The TDML data.

        Returns:
            tuple: A tuple containing the lapDict, total_cue, and cue_valve_ind.
        """
        DEC = self.DEC
        SETTINGS = self.SETTINGS
        SETTINGS_KEY = self.SETTINGS_KEY
        SETTINGS_VALVE = self.SETTINGS_VALVE
        LOC = self.LOC
        ID = self.ID
        LAP_LIST = self.LAP_LIST
        CUETYPE = self.CUETYPE
        TYPE = self.TYPE

        self.rprint("Creating/processing lapDict", end="", flush=True)
        context_info = tdml_data[1][SETTINGS][SETTINGS_KEY]
        lapDict = {}

        xml_fname = self.findLatest(self.file_tag["XML"])
        gpio_fname = self.findLatest(
            [self.file_tag["GPIO_SUFFIX"], self.file_tag["CSV"]]
        )
        # will have a .xml which will be removed later
        if xml_fname:
            lapDict_fname = xml_fname
            ftag2remove = self.file_tag["XML"]
        elif gpio_fname:
            lapDict_fname = gpio_fname.split(self.file_tag["GPIO_SUFFIX"])[0]
            ftag2remove = []

        # Iterate through each context in the list
        for context in context_info:
            # Extract common data from each context
            valves = context.get(SETTINGS_VALVE, [])
            location = context.get(LOC, [])
            id = context.get(ID, "")

            if valves:
                ctype = self._determine_ctype(valves[0])

            # Iterate through decorators to process lap-related data
            for decorator in context.get(DEC, []):
                lap_list = decorator.get(LAP_LIST, [])

                for lap in lap_list:
                    lap_key = f"Lap_{lap}"
                    if lap_key not in lapDict:
                        lapDict[lap_key] = self._init_lapDict_entry()
                    if ctype is not None:
                        lapDict[lap_key][CUETYPE].append(ctype)
                    lapDict[lap_key][SETTINGS_VALVE].extend(valves)
                    lapDict[lap_key][LOC].extend(location)
                    lapDict[lap_key][ID].append(id)

        cue_finder = []
        valve_finder = []
        for lap_key in lapDict:
            cue_finder.extend(lapDict[lap_key][CUETYPE])
            valve_finder.extend(lapDict[lap_key][SETTINGS_VALVE])
            valves_set = set(lapDict[lap_key][SETTINGS_VALVE])
            locations = lapDict[lap_key][LOC]
            lapDict[lap_key][TYPE] = self._determine_lap_type(valves_set, locations)

        cue_valve_ind = list(set(valve_finder))
        cue_valve_ind = [
            valve for valve in cue_valve_ind if valve != self.LAPSYNC_VALVE
        ]

        cue_arr = list(set(cue_finder))
        cue_arr = [cue for cue in cue_arr if cue is not None]

        total_cue = len(cue_arr)
        self.print_done_small_proc(new_line=False)
        # print lapDict to terminal
        behavior_utils.print_lapDict_results(lapDict, cue_arr)
        # save lapDict to file
        self.saveNloadUtils.savedict2file(
            lapDict,
            self.dict_name["LAPDICT"],
            lapDict_fname,
            file_tag_to_remove=ftag2remove,
            file_suffix=self.dict_name["LAPDICT"],
            date=True,
            filetype_to_save=[self.file_tag["MAT"], self.file_tag["PKL"]],
        )
        return lapDict, total_cue, cue_arr, cue_valve_ind

    def _determine_lap_type(self, valves_set: set, locations: list) -> str:
        """
        Determines the lap type based on the valves and locations.

        Args:
            valves_set (set): A set of valves.
            locations (list): A list of locations.

        Returns:
            str: The determined lap type.
        """
        if valves_set == {self.CUE1_VALVE, self.CUE2_VALVE}:
            if locations == self.CONTEXT1:
                return self.BOTH
            elif locations == self.CONTEXT2:
                return self.BOTH_SWITCH
        elif (
            valves_set == {self.CUE1_VALVE, self.LED_VALVE, self.TONE_VALVE}
            and locations == self.CONTEXT_3C
        ):
            return self.TCL
        elif (
            valves_set == {self.CUE1_VALVE, self.LED_VALVE}
            and locations == self.CONTEXT_3C_CL
        ):
            return self.CL
        elif (
            valves_set == {self.CUE1_VALVE, self.TONE_VALVE}
            and locations == self.CONTEXT_3C_TC
        ):
            return self.TC
        elif (
            valves_set == {self.LED_VALVE, self.TONE_VALVE}
            and locations == self.CONTEXT_3C_TL
        ):
            return self.TL
        elif valves_set == {self.CUE1_VALVE} and len(locations) == 1:
            return self.CUE1
        elif valves_set == {self.CUE2_VALVE} and len(locations) == 1:
            return self.CUE2
        elif valves_set == {self.CUE1_VALVE} and len(locations) == 2:
            return self.BC1
        elif valves_set == {self.CUE2_VALVE} and len(locations) == 2:
            return self.BC2
        else:
            return self.UNK

    def _determine_ctype(self, valve: str) -> str:
        """
        Determines the ctype based on the valve.

        Args:
            valve (str): The valve.

        Returns:
            str: The determined ctype.
        """
        if valve == self.CUE1_VALVE:
            return self.CUE1
        elif valve == self.CUE2_VALVE:
            return self.CUE2
        elif valve == self.TONE_VALVE:
            return self.TONE
        elif valve == self.LED_VALVE:
            return self.LED
        elif valve == self.OPTO_VALVE:
            return self.OPTO
        else:
            return None

    def _init_lapDict_entry(self) -> dict:
        """
        Initializes a lapDict entry.

        Returns:
            dict: The initialized lapDict entry.
        """
        return {
            self.CUETYPE: [],
            self.SETTINGS_VALVE: [],
            self.LOC: [],
            self.ID: [],
            self.TYPE: [],
        }


#############################################################
# Setup Variables
#############################################################
class setup_vars:
    """
    A class that contains utility methods for setting up variables in the treadBehDict.

    Methods:
    - empty_treadBehDict(cue_arr): Creates an empty treadBehDict to be filled.
    - rfid_ctxt(eventDictEntries, key_of_interest): Initializes proper keys, event type, and subkeys for rfid/ctxt.
    """

    def __init__(self, TDMLkey: dict) -> None:
        self.TDMLkey = TDMLkey

    #  create empty treadBehDict to be filled
    def empty_treadBehDict(self, cue_arr: list) -> dict:
        """
        Creates an empty treadBehDict to be filled.

        Parameters:
        - cue_arr (list): A list of cues.

        Returns:
        - treadBehDict (dict): An empty treadBehDict with the specified keys and structures.
        """
        TIME = self.TDMLkey["TIME"]
        TIME_KEY = self.TDMLkey["TIME_KEY"]
        SESSINFO = self.TDMLkey["SESSINFO"]
        POS_KEY = self.TDMLkey["POSITION_KEY"]
        LAP_NUM = self.TDMLkey["LAP_NUM_KEY"]
        TIME_NANO = self.TDMLkey["TIME_NANO"]
        NAME = self.TDMLkey["NAME_KEY"]
        POS = self.TDMLkey["POSITION"]
        LAP = self.TDMLkey["LAP_KEY"]
        REWZONE = self.TDMLkey["REW_ZONE"]
        REW = self.TDMLkey["REWARD"]
        LICK = self.TDMLkey["LICK"]
        RFID_VNAME = self.TDMLkey["RFID_VNAME"]
        CONTEXT_VNAME = self.TDMLkey["CONTEXT_VNAME"]
        SYNC = self.TDMLkey["SYNC"]
        TRIG = self.TDMLkey["TRIG"]

        base_keys = [TIME_KEY, POS_KEY]
        lap_keys = [key for key in base_keys if key != POS_KEY] + [LAP_NUM]
        y_keys = base_keys + [TIME_NANO]
        # frames_keys = base_keys + [COUNT]
        rfid_ctxt_keys = base_keys + [LAP_NUM, NAME]
        cue_event_keys = base_keys + [LAP_NUM]

        #  use dict_utils & self.TDMLkey vars
        #  to fill in treadBehDict to exact specs
        treadBehDict = {
            TIME: [],
            SESSINFO: [],
            POS: behavior_utils.init_empty_keys(y_keys),
            LAP: behavior_utils.init_empty_keys(lap_keys),
            REWZONE: behavior_utils.create_event_structure(),
            REW: behavior_utils.init_empty_keys(cue_event_keys),
            LICK: behavior_utils.init_empty_keys(cue_event_keys),
            RFID_VNAME: behavior_utils.init_empty_keys(rfid_ctxt_keys),
            CONTEXT_VNAME: behavior_utils.init_empty_keys(rfid_ctxt_keys),
            # self.TDMLkey["FRAME"]: init_empty_keys(frames_keys),
            self.TDMLkey["CUE_EVENTS"]: {
                cue: behavior_utils.create_event_structure([LAP_NUM]) for cue in cue_arr
            },
            SYNC: behavior_utils.init_empty_keys(
                [self.TDMLkey["ONTIME"], self.TDMLkey["OFFTIME"]]
            ),
            TRIG: behavior_utils.init_empty_keys(
                [self.TDMLkey["ON"], self.TDMLkey["OFF"]]
            ),
            self.TDMLkey["SYNCPIN"]: [],
            self.TDMLkey["LEDPIN"]: [],
        }
        return treadBehDict

    #  initialize proper keys, event type, & subkeys for rfid/ctxt
    def rfid_ctxt(self, eventDictEntries: dict, key_of_interest: str) -> tuple:
        """
        Initializes proper keys, event type, and subkeys for rfid/ctxt.

        Parameters:
        - eventDictEntries (dict): A dictionary containing event dictionary entries.
        - key_of_interest (str): The key of interest, either 'RFID_VNAME' or 'CONTEXT_VNAME'.

        Returns:
        - event_type (str): The event type.
        - key (str): The key.
        - Name (str): The name.
        """
        RFID_VNAME = self.TDMLkey["RFID_VNAME"]
        CONTEXT_VNAME = self.TDMLkey["CONTEXT_VNAME"]

        # Validate key_of_interest
        if key_of_interest not in [RFID_VNAME, CONTEXT_VNAME]:
            raise ValueError(
                f"key_of_interest must be either '{RFID_VNAME}' or '{CONTEXT_VNAME}'"
            )

        base_key = (
            self.TDMLkey["RFID_UPP"]
            if key_of_interest == RFID_VNAME
            else self.TDMLkey["CONTEXT_UPP"]
        )

        EVENT_TYPE = self.TDMLkey[f"{base_key}_VNAME"]
        KEY = self.TDMLkey[f"{base_key}_KEY"]
        NAME_KEY = eventDictEntries[KEY][self.TDMLkey[f"{base_key}_SUBKEY"]]

        key = KEY
        event_type = EVENT_TYPE
        Name = NAME_KEY
        return event_type, key, Name


#############################################################
# Append Data
#############################################################
class dataAppender:
    """
    A class that provides methods to append event data to a dictionary.

    Methods:
    - events: Append event data for all types.
    - rfid_ctxt: Append event data addition needed for rfid/ctxt.
    """

    def __init__(self, TDMLkey: dict) -> None:
        self.TDMLkey = TDMLkey

    def events(
        self,
        event_dict: dict,
        event_type: str,
        currTime: float,
        currYPos: float = None,
        lap: int = None,
        TimesNano: float = None,
        Name: str = None,
    ) -> None:
        """
        Append event data for all types to the given event dictionary.

        Parameters:
        - event_dict (dict): The event dictionary to append the data to.
        - event_type (str): The type of the event.
        - currTime (float): The current time of the event.
        - currYPos (float, optional): The current Y position of the event. Default is None.
        - lap (int, optional): The lap number of the event. Default is None.
        - TimesNano (float, optional): The nano time of the event. Default is None.
        - Name (str, optional): The name of the event. Default is None.
        """
        TIME_KEY = self.TDMLkey["TIME_KEY"]
        POS_KEY = self.TDMLkey["POSITION_KEY"]
        START = self.TDMLkey["START"]
        STOP = self.TDMLkey["STOP"]
        LAP_NUM = self.TDMLkey["LAP_NUM_KEY"]
        TIME_NANO = self.TDMLkey["TIME_NANO"]
        NAME = self.TDMLkey["NAME_KEY"]

        event_dict[event_type][TIME_KEY].append(currTime)
        if currYPos is not None:
            event_dict[event_type][POS_KEY].append(currYPos)
        if lap is not None:
            if event_type not in [START, STOP]:
                event_dict[event_type][LAP_NUM].append(lap)
            else:
                event_dict[LAP_NUM].append(lap)
        if TimesNano is not None:
            event_dict[event_type][TIME_NANO].append(TimesNano)
        if Name is not None:
            event_dict[event_type][NAME].append(Name)

    def rfid_ctxt(
        self,
        eventDictEntries,
        treadBehDict: dict,
        key: str,
        event_type: str,
        currTime: float,
        currYPos: float,
        currLap: int,
        Name: str,
    ) -> None:
        """
        Append event data addition needed for rfid/ctxt to the given tread behavior dictionary.

        Parameters:
        - eventDictEntries (list): The list of event dictionary entries.
        - treadBehDict (dict): The tread behavior dictionary to append the data to.
        - key (str): The key to check in the event dictionary entries.
        - event_type (str): The type of the event.
        - currTime (float): The current time of the event.
        - currYPos (float): The current Y position of the event.
        - currLap (int): The lap number of the event.
        - Name (str): The name of the event.
        """
        if key in eventDictEntries:
            self.events(
                treadBehDict,
                event_type,
                currTime,
                currYPos,
                lap=currLap,
                Name=Name,
            )


#############################################################
# Process Events
#############################################################
class event_processor:
    """
    A class that processes different types of events in a behavioral task.

    Args:
        TDMLsettings (object): An object containing settings for the TDML task.
        total_cue (int): The total number of cue events.
        cue_valve_ind (list): A sorted list of indices corresponding to cue valve events. This helps organize whether a cue event is a scent, tone, led, or opto event.

    Attributes:
        TDMLsettings (object): An object containing settings for the TDML task.
        dataAppender (object): An instance of the dataAppender class.
        cue_valve_ind (list): A sorted list of indices corresponding to cue valve events.
        cue_arr (list): A list of cue events based on the total number of cues.
        setup_vars (object): An instance of the setup_vars class.

    Methods:
        cue: Process cue events.
        lap: Process lap events.
        noncue: Process non-cue events.
        rfid_ctxt: Process RFID/Context events.
        special_ev: Process special events (opto/tact/lick/tone).
        time_adjuster: Adjust time based on start time and difference between nano and beMate time.
        valve: Process valve events (rew, led, sync, cue events).
    """

    def __init__(
        self,
        TDMLkey: dict,
        TDMLsettings: object,
        total_cue: int,
        cue_arr: list,
        cue_valve_ind: list,
    ) -> None:
        self.dataAppender = dataAppender(TDMLkey=TDMLkey)
        self.setup_vars = setup_vars(TDMLkey=TDMLkey)
        self.cue_valve_ind = cue_valve_ind
        self.cue_valve_ind.sort()
        self.TDMLkey = TDMLkey
        self.TDMLsettings = TDMLsettings
        self._create_cue_arr(total_cue, cue_arr)

    def _create_cue_arr(self, total_cue: int, cue_arr: list) -> None:
        """
        Create the cue array based on the total number of cues.

        Parameters:
        - total_cue (int): The total number of cues.

        Returns:
        - None

        The cue array is created based on the total number of cues provided. The cue array contains the cues to be used during the experiment. The cues are defined using the TDMLkey dictionary, which contains the keys for different cues such as CUE1, CUE2, LED, TONE, and NOCUE. The cue array is assigned to the `cue_arr` attribute of the class instance.

        Example usage:
        >>> _create_cue_arr(1)
        """
        NOCUE = self.TDMLkey["NOCUE"]

        if total_cue > 0:
            self.cue_arr = cue_arr
            self.cue_arr.sort()
        else:
            self.cue_arr = [NOCUE]

    def position(
        self,
        data_dict: dict,
        treadBehDict: dict,
        currTime: float,
        currYPos: float,
    ) -> tuple:
        """
        Update the position information in the treadBehDict based on the data_dict.

        Args:
            data_dict (dict): A dictionary containing the data.
            treadBehDict (dict): The treadBehDict to be updated.
            currTime (float): The current time.
            currYPos (float): The current Y position.

        Returns:
            tuple: A tuple containing the updated treadBehDict and currYPos.
        """
        tdml_id_position = self.TDMLkey["TDML_ID_POSITION"]
        POSITION = self.TDMLkey["POSITION"]
        TIME_MS = self.TDMLkey["TIME_MS"]

        if tdml_id_position in data_dict:
            eventDictEntries = data_dict[tdml_id_position]
            currYPos = data_dict[POSITION]
            self.dataAppender.events(
                treadBehDict,
                POSITION,
                currTime,
                currYPos,
                TimesNano=eventDictEntries[TIME_MS],
            )
        return treadBehDict, currYPos

    #  process cue events
    def cue(
        self,
        eventDictEntries: dict,
        treadBehDict: dict,
        currTime: float,
        currYPos: float,
        currLap: int,
        tone: bool = False,
    ) -> None:
        """
        Process cue events, which includes scents (CUE1 or CUE2), tones (TONE), led (LED), and opto (OPTO).

        Args:
            eventDictEntries (dict): A dictionary containing event entries.
            treadBehDict (dict): A dictionary containing behavioral data.
            currTime (float): The current time.
            currYPos (float): The current Y position.
            currLap (int): The current lap number.
            tone (bool, optional): Whether the cue event is a tone event. Defaults to False.
        """
        TONE = self.TDMLkey["TONE"]
        EVENT_VALVE = self.TDMLkey["EVENT_VALVE"]
        PIN = self.TDMLkey["PIN"]
        ACTION = self.TDMLkey["ACTION"]
        OPEN = self.TDMLkey["OPEN"]
        CLOSE = self.TDMLkey["CLOSE"]
        START = self.TDMLkey["START"]
        STOP = self.TDMLkey["STOP"]

        if tone:
            event_key = TONE
        else:
            event_key = EVENT_VALVE

        if eventDictEntries[event_key][PIN] in self.cue_valve_ind:
            idx = self.cue_valve_ind.index(eventDictEntries[event_key][PIN])
            cue_key = self.cue_arr[idx]
            action = eventDictEntries[event_key][ACTION]
            if action == OPEN:
                self.dataAppender.events(
                    treadBehDict[self.TDMLkey["CUE_EVENTS"]][cue_key],
                    START,
                    currTime,
                    currYPos,
                    lap=currLap,
                )
            elif action == CLOSE:
                self.dataAppender.events(
                    treadBehDict[self.TDMLkey["CUE_EVENTS"]][cue_key],
                    STOP,
                    currTime,
                    currYPos,
                )

    #  process laps
    def lap(
        self, data_dict: dict, treadBehDict: dict, currTime: float, currLap: int
    ) -> tuple:
        """
        Process lap events.

        Args:
            data_dict (dict): A dictionary containing data.
            treadBehDict (dict): A dictionary containing behavioral data.
            currTime (float): The current time.
            currLap (int): The current lap number.

        Returns:
            tuple: A tuple containing the updated treadBehDict and currLap.
        """
        LAP = self.TDMLkey["LAP_KEY"]
        if LAP in data_dict:
            currLap = data_dict[LAP]
            self.dataAppender.events(treadBehDict, LAP, currTime, lap=currLap)
        return treadBehDict, currLap

    #  process noncue events
    def noncue(
        self,
        eventDictEntries: dict,
        treadBehDict: dict,
        event_type: str,
        currTime: float,
        currYPos: float,
        currLap: int = [],
    ) -> None:
        """
        Process non-cue events.

        Args:
            eventDictEntries (dict): A dictionary containing event entries.
            treadBehDict (dict): A dictionary containing behavioral data.
            event_type (str): The type of the event.
            currTime (float): The current time.
            currYPos (float): The current Y position.
            currLap (int, optional): The current lap number. Defaults to [].
        """
        CONTEXT = self.TDMLkey["CONTEXT_KEY"]
        ACTION = self.TDMLkey["ACTION"]
        REWZONE = self.TDMLkey["REW_ZONE"]
        START = self.TDMLkey["START"]
        STOP = self.TDMLkey["STOP"]

        action = eventDictEntries[CONTEXT][ACTION]
        event_type, action = self.TDMLsettings.get_correct_et_act_noncue(
            event_type, action
        )

        if action == START:
            if event_type == REWZONE:
                self.dataAppender.events(
                    treadBehDict[event_type], START, currTime, currYPos
                )
            else:
                self.dataAppender.events(
                    treadBehDict[event_type], START, currTime, currYPos, lap=currLap
                )
        elif action == STOP:
            self.dataAppender.events(treadBehDict[event_type], STOP, currTime, currYPos)

    #  process rfid/ctxt events
    def rfid_ctxt(
        self,
        eventDictEntries: dict,
        treadBehDict: dict,
        key_of_interest: str,
        currTime: float,
        currYPos: float,
        currLap: int,
    ) -> dict:
        """
        Process RFID/Context events.

        Args:
            eventDictEntries (dict): A dictionary containing event entries.
            treadBehDict (dict): A dictionary containing behavioral data.
            key_of_interest (str): The key of interest.
            currTime (float): The current time.
            currYPos (float): The current Y position.
            currLap (int): The current lap number.

        Returns:
            dict: The updated treadBehDict.
        """
        CONTEXT_VNAME = self.TDMLkey["CONTEXT_VNAME"]
        REW = self.TDMLkey["REWARD"]
        SCENT_CONTEXT = [
            self.TDMLkey["S1C1"],
            self.TDMLkey["S1C2"],
            self.TDMLkey["S2C1"],
            self.TDMLkey["S2C2"],
        ]

        event_type, key, Name = self.setup_vars.rfid_ctxt(
            eventDictEntries, key_of_interest
        )
        self.dataAppender.rfid_ctxt(
            eventDictEntries,
            treadBehDict,
            key,
            event_type,
            currTime,
            currYPos,
            currLap,
            Name,
        )
        if key_of_interest == CONTEXT_VNAME:
            event_types = set()
            event_types.add(Name)
            for event_type in event_types:
                #  needed to extract rewZone
                if event_type not in SCENT_CONTEXT:
                    if event_type == REW:
                        self.noncue(
                            eventDictEntries,
                            treadBehDict,
                            event_type,
                            currTime,
                            currYPos,
                        )
                    else:
                        continue
        return treadBehDict

    #  process opto/tact/lick/tone events
    def special_ev(
        self,
        event_of_int: str,
        treadBehDict: dict,
        currTime: float,
        currYPos: float,
        currLap: int,
    ) -> dict:
        """
        Process special events (opto/tact/lick/tone).

        Args:
            event_of_int (str): The event of interest.
            treadBehDict (dict): A dictionary containing behavioral data.
            currTime (float): The current time.
            currYPos (float): The current Y position.
            currLap (int): The current lap number.

        Returns:
            dict: The updated treadBehDict.
        """
        self.dataAppender.events(
            treadBehDict, event_of_int, currTime, currYPos, lap=currLap
        )
        return treadBehDict

    #  adjust nano time based on start time
    #  adjust y.Time based on diff btw nano & beMate time
    def time_adjuster(
        self, treadBehDict: dict, thres: float = 0.02, rm_window: int = 200
    ) -> dict:
        """
        Adjust time based on start time and difference between nano and beMate time.

        Args:
            treadBehDict (dict): A dictionary containing behavioral data.
            thres (float, optional): The threshold for time adjustment. Defaults to 0.02.
            rm_window (int, optional): The window size for running mean. Defaults to 200.

        Returns:
            dict: The updated treadBehDict.
        """
        POS = self.TDMLkey["POSITION"]
        TIME_NANO = self.TDMLkey["TIME_NANO"]
        TIME_KEY = self.TDMLkey["TIME_KEY"]

        yTimesNano = np.array(treadBehDict[POS][TIME_NANO])
        yTimes = np.array(treadBehDict[POS][TIME_KEY])
        yTimesNano = (yTimesNano - yTimesNano[0]) / 1000 + yTimes[0]
        dTimes = yTimes - yTimesNano
        dTimes_smoothed = runmean(dTimes, rm_window)
        dTimes = dTimes - dTimes_smoothed
        yTimes2 = yTimes.copy()
        yTimes2[dTimes > thres] = yTimes[dTimes > thres] - dTimes[dTimes > thres]
        treadBehDict[POS][self.TDMLkey["TIME_ADJ"]] = yTimes2.tolist()

        return treadBehDict

    #  process valve events (rew, led, sync, cue events)
    def valve(
        self,
        eventDictEntries: dict,
        treadBehDict: dict,
        currTime: float,
        currYPos: float,
        currLap: int,
    ) -> None:
        """
        Process valve events (rew, led, sync, cue events).

        Args:
            eventDictEntries (dict): A dictionary containing event entries.
            treadBehDict (dict): A dictionary containing behavioral data.
            currTime (float): The current time.
            currYPos (float): The current Y position.
            currLap (int): The current lap number.
        """
        EVENT_VALVE = self.TDMLkey["EVENT_VALVE"]
        PIN = self.TDMLkey["PIN"]
        ACTION = self.TDMLkey["ACTION"]
        REW_VALVE = self.TDMLkey["REW_VALVE"]
        OPEN = self.TDMLkey["OPEN"]
        REW = self.TDMLkey["REWARD"]
        SYNC = self.TDMLkey["SYNC"]
        TRIG = self.TDMLkey["TRIG"]
        LED = self.TDMLkey["LED"]
        START = self.TDMLkey["START"]
        STOP = self.TDMLkey["STOP"]
        TONE = self.TDMLkey["TONE"]
        CLOSE = self.TDMLkey["CLOSE"]

        if EVENT_VALVE in eventDictEntries:
            valve_pin = eventDictEntries[EVENT_VALVE][PIN]
            valve_action = eventDictEntries[EVENT_VALVE][ACTION]
            #  reward pin
            if valve_pin == REW_VALVE and valve_action == OPEN:
                self.dataAppender.events(treadBehDict, REW, currTime, currYPos)
            #  syncPin
            if valve_pin == self.TDMLsettings.syncPin and valve_action == OPEN:
                treadBehDict[SYNC][self.TDMLkey["ONTIME"]] = currTime
                treadBehDict[TRIG][self.TDMLkey["ON"]] = 1
            elif valve_pin == self.TDMLsettings.syncPin and valve_action == CLOSE:
                treadBehDict[SYNC][self.TDMLkey["OFFTIME"]] = currTime
                treadBehDict[TRIG][self.TDMLkey["OFF"]] = 1
            #  ledPin
            if valve_pin == self.TDMLsettings.ledPin and valve_action == OPEN:
                self.dataAppender.events(
                    treadBehDict[LED],
                    START,
                    currTime,
                    currYPos,
                    lap=currLap,
                )
            elif valve_pin == self.TDMLsettings.ledPin and valve_action == CLOSE:
                self.dataAppender.events(treadBehDict[LED], STOP, currTime, currYPos)
            #  cue events
            self.cue(
                eventDictEntries,
                treadBehDict,
                currTime,
                currYPos,
                currLap,
            )
        elif TONE in eventDictEntries:
            self.cue(
                eventDictEntries,
                treadBehDict,
                currTime,
                currYPos,
                currLap,
                tone=True,
            )

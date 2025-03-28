from enum import Enum


class TXT(Enum):
    """
    Enum class `TXT` for defining string constants used as keys in the TDML2 tread behavior dictionary.

    Attributes:
        ACTION: String for a key representing a specific action or behavior.
        ADJ_FRAME: String for a key indicating adjustment frame times.
        BOTH: String for a key representing a combination of conditions or factors.
        BOTH_SWITCH: String for a key indicating a switch between two conditions.
        BC1, BC2: Strings for keys related to 'both' conditions.
        CLOSE: String for a key indicating a closing action or state.
        CONTEXT_ID, CONTEXT_KEY, CONTEXT_SUBKEY, CONTEXT_UPP, CONTEXT_VNAME: Strings for various keys and identifiers related to context.
        CONTEXT1, CONTEXT2, CONTEXT_3C, CONTEXT_3C_TC, CONTEXT_3C_TL, CONTEXT_3C_CL: Strings for keys representing context-related ranges or combinations.
        TCL, TC, TL, CL: Strings for keys related to tone, cue, and led combinations.
        COUNT_KEY: String for a key used for counting occurrences or instances.
        CUE1, CUE2, CUE1_VALVE, CUE2_VALVE, CUE_EVENTS, CUETYPE: Strings for keys related to cues and their types or values.
        DEC: String for a key referring to decorators or modifications.
        EVENT_VALVE: String for a key associated with a specific event valve.
        FRAME: String for a key related to frame or structure.
        LAP_KEY, LAP_LIST, LAP_NUM_KEY, LAP_TYPE: Strings for keys associated with laps or circuits.
        LED, LEDPIN, LED_VALVE: Strings for keys related to LED configuration.
        LICK: String for a key representing a licking action or sensor.
        LOC: String for a key related to location or position.
        NAME_KEY: String for a key used for naming elements or components.
        OFF, OFFTIME: Strings for keys indicating an off state or time.
        ODOR: String for a key related to odor or scent.
        ON, ONTIME: Strings for keys indicating an on state or time.
        OPEN: String for a key indicating an opening action or state.
        OPTO: String for a key related to optogenetics.
        PIN: String for a generic key representing a pin or connection.
        POSITION, POSITION_KEY: Strings for keys related to position.
        REW_VALVE, REW_ZONE, REWARD: Strings for keys related to rewards and reward zones.
        RFID_KEY, RFID_SUBKEY, RFID_UPP, RFID_VNAME: Strings for keys related to RFID tagging and identification.
        RESAMP: String for a key related to resampling.
        S1C1, S1C2, S2C1, S2C2: Strings for keys indicating different scent-context combinations.
        SESSINFO: String for a key related to session information.
        SETTINGS_KEY, SETTINGS_SUBKEY, SETTINGS_VALVE: Strings for keys related to settings, including context and valve settings.
        START, STOP: Strings for keys indicating the start and stop of actions or processes.
        SYNC, SYNCPIN: Strings for keys related to synchronization and its configuration.
        TACT: String for a key related to tactile or touch elements.
        TDML_ID_EVENT, TDML_ID_POSITION: Strings for keys identifying TDML events and position data.
        TIME, TIME_ADJ, TIME_KEY, TIME_MS, TIME_NANO: Strings for various keys related to time, including adjustments and formats.
        TONE, TONE_VALVE: Strings for keys related to tone and its valve configuration.
        TRIG: String for a key related to triggers.
        TYPE: String for a general key representing a type.
        UNK: String for a key representing an unknown or unspecified element.
        VELOCITY: String for a key related to velocity or speed.
    """

    ACTION = "action"
    ADJ_FRAME = "adjFrTimes"
    BOTH = "both"
    BOTH_SWITCH = "both_switch"
    BC1 = "both_c1"
    BC2 = "both_c2"
    CLOSE = "close"
    CONTEXT_ID = "context_id"
    CONTEXT_KEY = "context"
    CONTEXT_SUBKEY = "id"
    CONTEXT_UPP = "CONTEXT"
    CONTEXT_VNAME = "ctxt"
    CONTEXT1 = [500, 1300]
    CONTEXT2 = [1300, 500]
    CONTEXT_3C = [500, 1500, 1000]
    CONTEXT_3C_TC = [500, 1500]
    CONTEXT_3C_TL = [500, 1000]
    CONTEXT_3C_CL = [1500, 1000]
    #
    TCL = "toneNcueNled"
    TC = "toneNcue"
    TL = "toneNled"
    CL = "cueNled"
    #
    COUNT_KEY = "Count"
    CUE1 = "cue1"
    CUE1_VALVE = 3
    CUE2 = "cue2"
    CUE2_VALVE = 4
    CUETYPE = "cueType"
    CUE_EVENTS = "cueEvents"
    DEC = "decorators"
    EVENT_VALVE = "valve"
    FRAME = "frames"
    LAPSYNC = "lapSync"
    LAPSYNC_VALVE = 8
    LAP_KEY = "lap"
    LAP_LIST = "lap_list"
    LAP_NUM_KEY = "Lap"
    LAP_TYPE = "lapType"
    LED = "led"
    LEDPIN = "ledPin"
    LED_VALVE = 7
    LICK = "lick"
    LOC = "locations"
    NAME_KEY = "Name"
    NOCUE = "noCue"
    ODOR = "odors"
    OFF = "Off"
    OFFTIME = "OffTime"
    ON = "On"
    ONTIME = "OnTime"
    OPEN = "open"
    OPTO = "opto"
    OPTO_VALVE = 52
    PIN = "pin"
    POSITION = "y"
    POSITION_KEY = "Pos"
    RESAMP = "resamp"
    REWARD = "reward"
    REW_VALVE = 5
    REW_ZONE = "rewZone"
    RFID_KEY = "tag_reader"
    RFID_SUBKEY = "tag"
    RFID_UPP = "RFID"
    RFID_VNAME = "rfid"
    S1C1 = "scent1_context1"
    S1C2 = "scent1_context2"
    S2C1 = "scent2_context1"
    S2C2 = "scent2_context2"
    SESSINFO = "sessInfo"
    SETTINGS_KEY = "settings"
    SETTINGS_SUBKEY = "contexts"
    SETTINGS_VALVE = "valves"
    START = "start"
    STOP = "stop"
    SYNC = "sync"
    SYNCPIN = "syncPin"
    TACT = "tact"
    TDML_ID_EVENT = "127.0.0.1:5015"
    TDML_ID_POSITION = "127.0.0.1:5025"
    TIME = "time"
    TIME_ADJ = "TimesAdj"
    TIME_KEY = "Time"
    TIME_MS = "millis"
    TIME_NANO = "TimesNano"
    TONE = "tone"
    TONE_VALVE = 10
    TRIG = "TRIG"
    TYPE = "Type"
    UNK = "unknown"
    VELOCITY = "vel"


class SETTINGS:
    """
    Class that represents the settings for TDML2treadBehDict.
    """

    def __init__(self) -> None:
        self.syncPin = None
        self.ledPin = None
        self.sessInfo = None

    def update_settings(self, settings_dict: dict) -> None:
        """
        Updates the settings based on the provided settings dictionary.

        Args:
            settings_dict (dict): The settings dictionary.

        Returns:
            None
        """
        if "sync_pin" in settings_dict:
            self.syncPin = settings_dict["sync_pin"]
        if "led" in settings_dict:
            self.ledPin = settings_dict["led"]

    def update_sess_info(self, sess_info: str) -> None:
        """
        Updates the session information.

        Args:
            sess_info (str): The session information.

        Returns:
            None
        """
        self.sessInfo = sess_info

    @staticmethod
    def get_correct_et_act_noncue(original_type: str, original_action: str) -> tuple:
        """
        Gets the correct event type and act values for non-cue events.

        Args:
            original_type (str): The original event type.
            original_action (str): The original event action.

        Returns:
            tuple: The corrected event type and action.
        """
        type2return = original_type
        act2return = original_action
        if original_type == TXT.REWARD.value:
            type2return = TXT.REW_ZONE.value
        if original_action == TXT.OPEN.value:
            act2return = TXT.START.value
        if original_action == TXT.CLOSE.value:
            act2return = TXT.STOP.value

        return type2return, act2return

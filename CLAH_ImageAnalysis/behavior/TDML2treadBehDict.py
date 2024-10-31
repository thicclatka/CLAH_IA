from CLAH_ImageAnalysis.behavior import TDML2tBD_enum as TDMLE
from CLAH_ImageAnalysis.behavior import TDML2tBD_utils as TDML_utils
from CLAH_ImageAnalysis.core import BaseClass as BC

#############################################################
#  Process TDML class funcs
#############################################################


class TDML_processor(BC):
    """
    Class for processing TDML data and creating treadBehDict.

    Args:
        folder_path (list, optional): The folder_path to the folder containing TDML files. Defaults to [].

    Attributes:
        folder_folder_path (str): The selected folder folder_path.

    Methods:
        read_TDML: Reads the TDML file and returns the lapDict.
        importTDML2tBD: Creates and processes the treadBehDict.
    """

    def __init__(self, folder_path: list | str = []) -> None:
        self.program_name = "QT"
        self.class_type = "utils"
        BC.__init__(self, self.program_name, mode=self.class_type)

        self.folder_path = self.utils.path_selector(folder_path)

        self.TDMLkey = self.enum2dict(TDMLE.TXT)
        self.TDMLsettings = TDMLE.SETTINGS()
        self.Extractor = TDML_utils.TDML_extractor(
            TDMLkey=self.TDMLkey, TDMLsettings=self.TDMLsettings
        )

    def read_TDML(self) -> dict:
        """
        Reads the TDML file and returns the lapDict.

        Returns:
            dict: The lapDict containing lap data.
        """
        self.utils.folder_tools.chdir_check_folder(self.folder_path, create=False)
        file_to_read = self.findLatest(self.file_tag["TDML"])

        tdml_data = []
        tdml_data = self.saveNloadUtils.load_file(
            file_to_read, multi4json=True, list2append4json=tdml_data
        )

        lapDict, self.total_cue, self.cue_arr, self.cue_valve_ind = (
            self.Extractor.lapDict(tdml_data)
        )
        # keep tdml_data in class for later use
        self.tdml_data = tdml_data
        # return lapDict & self.total_cue
        return lapDict

    def importTDML2tBD(self) -> tuple:
        """
        Creates and processes the treadBehDict.

        Returns:
            tuple: A tuple containing the treadBehDict and cue_arr.
        """
        self.rprint("Creating/processing treadBehDict", end="", flush=True)

        # init classes after procuring self.total_cue & cue_valve_ind from TDML_reader
        self.EventProcessor = TDML_utils.event_processor(
            TDMLkey=self.TDMLkey,
            TDMLsettings=self.TDMLsettings,
            total_cue=self.total_cue,
            cue_arr=self.cue_arr,
            cue_valve_ind=self.cue_valve_ind,
        )

        # initiate class to create empty treadBehDict to fill
        SetupVar = TDML_utils.setup_vars(self.TDMLkey)
        # this func w/in SetupVar requires cue_arr, which sits in EventProcessor
        self.treadBehDict = SetupVar.empty_treadBehDict(self.cue_arr)
        self.currLap = 0
        self.currYPos = 0
        self.currTime = 0

        TIME = self.TDMLkey["TIME"]
        for count, data_dict in enumerate(self.tdml_data):
            self.treadBehDict = self.Extractor.settings(data_dict, self.treadBehDict)
            if TIME in data_dict:
                self.currTime = data_dict[TIME]
                if (
                    self.treadBehDict[TIME]
                    and self.currTime < self.treadBehDict[TIME][-1]
                ):
                    #  Skip to next iteration if currTime is less
                    #  then last entry in time array
                    continue

                #  append time to treadbehDict
                self.treadBehDict[TIME].append(self.currTime)

                # if position in dict entry, will process position data accordingly
                self.treadBehDict, self.currYPos = self.EventProcessor.position(
                    data_dict=data_dict,
                    treadBehDict=self.treadBehDict,
                    currTime=self.currTime,
                    currYPos=self.currYPos,
                )

                #  if lap in dict entry, will process lap data accordingly & append
                self.treadBehDict, self.currLap = self.EventProcessor.lap(
                    data_dict, self.treadBehDict, self.currTime, self.currLap
                )

                # fill in events in treadBehDict
                self.process_events(data_dict)

        # create TimesAdj key in tBD
        self.treadBehDict = self.EventProcessor.time_adjuster(self.treadBehDict)
        self.treadBehDict = self.saveNloadUtils.convert_lists_to_arrays(
            self.treadBehDict
        )
        self.rprint(self.text_lib["completion"]["small_proc"])
        return self.treadBehDict, self.EventProcessor.cue_arr

    def process_events(self, data_dict: dict) -> None:
        """
        Process events from the data dictionary.

        Args:
            data_dict (dict): A dictionary containing event data.

        Returns:
            None
        """
        tdml_id_event = self.TDMLkey["TDML_ID_EVENT"]
        # id num of interest based on .tdml file
        RFID_KEY = self.TDMLkey["RFID_KEY"]
        RFID_VNAME = self.TDMLkey["RFID_VNAME"]
        CONTEXT_KEY = self.TDMLkey["CONTEXT_KEY"]
        CONTEXT_VNAME = self.TDMLkey["CONTEXT_VNAME"]
        LICK = self.TDMLkey["LICK"]

        RFID_CTXT_KEY = [RFID_KEY, CONTEXT_KEY]
        RFID_CTXT_VNAME = [RFID_VNAME, CONTEXT_VNAME]
        SPECIAL_EVENT_KEY = [LICK]
        if tdml_id_event in data_dict:  # when event code is hit
            eventDictEntries = data_dict[tdml_id_event]
            #  for rfid & ctxt (w/in ctxt, rewZone is also covered)
            for RC_COUNT, RC_KEY in enumerate(RFID_CTXT_KEY):
                if RC_KEY in eventDictEntries:
                    self.treadBehDict = self.EventProcessor.rfid_ctxt(
                        eventDictEntries,
                        self.treadBehDict,
                        RFID_CTXT_VNAME[RC_COUNT],
                        self.currTime,
                        self.currYPos,
                        self.currLap,
                    )
            #  to process lick
            for SP_EV_KEY in SPECIAL_EVENT_KEY:
                if SP_EV_KEY in eventDictEntries:
                    self.treadBehDict = self.EventProcessor.special_ev(
                        SP_EV_KEY,
                        self.treadBehDict,
                        self.currTime,
                        self.currYPos,
                        self.currLap,
                    )
            #  to process rew, syncPin, ledPin, cueEvents (for multiple cues)
            self.EventProcessor.valve(
                eventDictEntries,
                self.treadBehDict,
                self.currTime,
                self.currYPos,
                self.currLap,
            )


#############################################################
#  Main function: TDML -> treadBehDict
#############################################################
def TDML2treadBehDict(folder_path: list | str = []) -> tuple:
    """
    Converts TDML data to treadBehDict.

    Args:
        folder_path (list, optional): The folder_path to the folder containing TDML files. Defaults to [].

    Returns:
        tuple: A tuple containing the treadBehDict, lapDict, and cue_arr.
    """
    TDML_proc = TDML_processor(folder_path)
    # reads tdml file & returns lapDict & total_cue
    # stores tdml_data for TDML_proc to use for treadBehDict creation
    lapDict = TDML_proc.read_TDML()
    # using stored tdml_data creates treadBehDict
    treadBehDict, cue_arr = TDML_proc.importTDML2tBD()
    return treadBehDict, lapDict, cue_arr

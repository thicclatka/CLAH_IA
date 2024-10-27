from enum import Enum


class TXT(Enum):
    """
    Enum class `TXT` representing string constants used as keys in PlaceFieldLappedAnalysis.

    Attributes:
        SHUFF: String for a key related to shuffled data.
        SHUFF_MNRATE: String for a key representing the mean rate in shuffled data.
        VAL: String for a key indicating a value.
        ZSCORE: String for a key related to Z-score.
        PVAL: String for a key denoting the p-value.
        INFOSPK: String for a key for information per spike.
        INFOSEC: String for a key for information per second.
        BYLAP: String for a key indicating data by lap.
        RAW_OCCU: String for a key related to raw occupancy data.
        OCCU: String for a key representing occupancy.
        ISPC: String for a key indicating whether a cell is a place cell.
        BESTFIELD: String for a key denoting the best field of a place cell.
        PKRATE: String for a key related to the peak rate.
        POSRATE: String for a key indicating position rates.
        POSSUM: String for a key representing the sum of positions.
        POSRATERAW: String for a key for raw position rates.
        B_BINS, G_BINS: Strings for keys related to bad and good bins, respectively.
        NA_POS: String for a key representing positions with NaN values.
        PFS_MN: String for a key indicating the mean of place field statistics.
        PFS_ST: String for a key for place field statistics.
        RATEPERC: String for a key related to rate percentile.
        EDGE_R_MULT: String for a key representing the multiple of edge rate.
        TRIMSTART, TRIMEND: Strings for keys indicating the start and end of trimmed runs.
        MINRUN: String for a key for minimum run time.
        MINPFBINS: String for a key for the minimum number of place field bins.
        MINVEL: String for a key for minimum velocity.
        SHUFFN: String for a key representing the number of shuffles.
        CIRC_SHUFFN: String for a key for circular shuffles per lap number.
        RATEALLSHUFF: String for a key related to rates in all shuffles.
        POS_PFIN: String for a key representing position in the place field.
        POS_PFPK: String for a key for the peak position in the place field.
        POS_PFIN_ALL: String for a key for all positions in the place field.
        WHICHLAP: String for a key indicating which lap.
        RUNTIME: String for a key related to run times.
        MINLAPSN: String for a key for the minimum number of laps.
        LAPREL: String for a key for lap-related rates.
        THRESHRATE: String for a key for threshold rate.
        SIGRATE: String for a key for significant rate.
        PERCRATE: String for a key for percentile rate.
    """

    SHUFF = "Shuff"
    SHUFF_MNRATE = "shuffMeanRate"
    VAL = "value"
    ZSCORE = "Zscore"
    PVAL = "pVal"
    INFOSPK = "InfoPerSpk"
    INFOSEC = "InfoPerSec"
    BYLAP = "ByLap"
    RAW_OCCU = "rawOccupancy"
    OCCU = "Occupancy"
    ISPC = "isPC"
    BESTFIELD = "BestField"
    PKRATE = "PeakRate"
    POSRATE = "posRates"
    POSSUM = "posSums"
    POSRATERAW = "posRateRaw"
    B_BINS = "badBins"
    G_BINS = "goodBins"
    NA_POS = "nanedPos"
    PFS_MN = "PFSMean"
    PFS_ST = "PFSStats"
    RATEPERC = "RatePerc"
    EDGE_R_MULT = "edgeRateMultiple"
    TRIMSTART = "trimRunStarts"
    TRIMEND = "trimRunEnds"
    MINRUN = "minRunTime"
    MINPFBINS = "minPFBins"
    MINVEL = "minVel"
    SHUFFN = "shuffN"
    CIRC_SHUFFN = "circShuffLapN"
    RATEALLSHUFF = "ratesAllShuff"
    POS_PFIN = "PFInPos"
    POS_PFPK = "PFPeakPos"
    POS_PFIN_ALL = "PFInAllPos"
    WHICHLAP = "whichLap"
    RUNTIME = "runTimes"
    MINLAPSN = "minLapsN"
    LAPREL = "LapRelRate"
    THRESHRATE = "ThreshRate"
    SIGRATE = "sigRate"
    PERCRATE = "percRate"
    TREADPOS = "treadPos"
    TIMESEG = "timeSeg"

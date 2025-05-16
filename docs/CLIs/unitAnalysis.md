# Unit Analysis

## [Quick Tuning](https://github.com/thicclatka/CLAH_IA/blob/main/CLAH_ImageAnalysis/unitAnalysis/quickTuning.py)

### QT Input

- segmented dictionary (segDict)
- treadmill/behavior log/json file (.tdml)

### QT Output

- treadmill behavior dictionary (treadBehDict)
- lap info dictionary (lapDict)
- Cue Cell info & arrays dictionary (CueCellFinderDict)
- cueShiftStruc
- Figures (stored in /path/to/segDict/Figures)

### QT CLI

```console
usage: quickTuning.py   [-h] [-p PATH] [-s2p SESS2PROCESS] [-f FPS] [-sdt SDTHRESH]
                        [-to TIMEOUT] [-ow OVERWRITE] [-pp TOPLOTPKS] [-4p FORPRES]
                        [-cat CONCATCHECK]

Run quickTuning with command line arguments.

options:
  -h, --help            show this help message and exit
  -p PATH, --path PATH  Path to the data directory (e.g. /path/to/data).
                        Default will prompt user to choose path.
                        NOTE: Pick directory which holds session folders.
                              Do not pick/set a session folder.
  -s2p SESS2PROCESS, --sess2process SESS2PROCESS
                        List of sessions to process. Write in format '1,2,3', '1-3',
                        or '1,2-5' to select by specific session number.
                        Input all or ALL to process all eligible sessions that are
                        available within the set path.
                        Default will prompt user to choose.
  -fw FRAMEWINDOW, --frameWindow FRAMEWINDOW
                        Window size (in frames) used for smoothing for the event/peak
                        detection. Default is 15.
  -sdt SDTHRESH, --sdThresh SDTHRESH
                        Threshold multiplier for event/peak detection based on the standard
                        deviation of the signal's derivative.
                        Default is 3.
  -to TIMEOUT, --timeout TIMEOUT
                        Minimum distance between detected peaks/events in seconds.
                        Default is 3.
  -ow OVERWRITE, --overwrite OVERWRITE
                        Overwrite existing files
                        (e.g. pkl & mat for treadBehDict, lapDict, cueShiftStruc)
  -pp TOPLOTPKS, --toPlotPks TOPLOTPKS
                        Whether to plot results from pks_utils. Default is False.
  -4p FORPRES, --forPres FORPRES
                        Whether to export .svg for figures in addition to the usual
                        png output. Default is False.
```

## [WrapMultSessStruc](https://github.com/thicclatka/CLAH_IA/blob/main/CLAH_ImageAnalysis/unitAnalysis/wrapMultSessStruc.py)

### WMSS Input

- cueShiftStruc

### WMSS Output

- multSessSegStruc (location is based on -output_folder parameter)

### WMSS CLI

```console
usage: wrapMultSessStruc.py [-h] [-p PATH] [-s2p SESS2PROCESS] [-out OUTPUT_FOLDER]

Run wrapMultSessStruc with command line arguments.

options:
  -h, --help            show this help message and exit
  -p PATH, --path PATH  Path to the data directory (e.g. /path/to/data).
                        Default will prompt user to choose path.
                        NOTE: Pick directory which holds session folders.
                              Do not pick/set a session folder.
  -s2p SESS2PROCESS, --sess2process SESS2PROCESS
                        List of sessions to process. Write in format '1,2,3', '1-3',
                        or '1,2-5' to select by specific session number.
                        Input all or ALL to process all eligible sessions that are
                        available within the set path.
                        Default will prompt user to choose.
  -out OUTPUT_FOLDER, --output_folder OUTPUT_FOLDER
                        Path for the output of pkl & mat files for multSessSegStruct.
                        Default is None, which prompts user to input experiment keywords
                        & brain region to create output folder name.
                        All output paths will be prepended with '_MS_'.
```

## [Post Cell Registrar Cue Cell Finder (PostCR_CueCellFinder)](https://github.com/thicclatka/CLAH_IA/blob/main/CLAH_ImageAnalysis/unitAnalysis/PostCR_CueCellFinder.py)

### PCRCCF Input

- multSessSegStruc

### PCRCCF Output

- PCRTrigSigDict
    - Figures per subject (/path/to/multSessSegStruc/Figures)
    - Figures across subjects (/path/to/MultSess/dir/~GroupData)

### PCRCFF CLI

```console
usage: PostCR_CueCellFinder.py  [-h] [-p PATH] [-s2p SESS2PROCESS] [-ots OUTLIER_TS]
                                [-sf SESSFOCUS] [-4p FORPRES]

Run Post CellRegistrar Cue Cell Finder with command line arguments.

options:
  -h, --help            show this help message and exit
  -p PATH, --path PATH  Path to the data directory (e.g. /path/to/data).
                        Default will prompt user to choose path.
                        NOTE: Pick directory which holds session folders.
                              Do not pick/set a session folder.
  -s2p SESS2PROCESS, --sess2process SESS2PROCESS
                        List of sessions to process. Write in format '1,2,3', '1-3',
                        or '1,2-5' to select by specific session number.
                        Input all or ALL to process all eligible sessions that are
                        available within the set path.
                        Default will prompt user to choose.
  -ots OUTLIER_TS, --outlier_ts OUTLIER_TS
                        Outlier threshold to filter out meanTrigSig by group where
                        mean value exceeds threshold set here.
                        Default is 10^2.
  -sf SESSFOCUS, --sessFocus SESSFOCUS
                        Select number of sessions to plot. Default is None, which plots
                        all sessions
  -4p FORPRES, --forPres FORPRES
                        Whether to export .svg for figures in addition to the usual
                        png output. Default is False.
  -pit PLOTINDTRIGSIG, --plotIndTrigSig PLOTINDTRIGSIG
                        Whether to plot individual TrigSig by subjected ID. Default is True.
```

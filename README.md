# Calcium Imaging Analysis Pipeline for 2-Photon Microscopy (1-Photon in progress)

## Requirements

- [Anaconda](https://docs.anaconda.com/)
- [CaImAn](https://github.com/thicclatka/CaImAn)
- [ROICaT](https://github.com/RichieHakim/ROICaT)

## Setup

Following the [dev-mode option installation instructions on CaImAn](https://github.com/flatironinstitute/CaImAn/blob/main/docs/source/Installation.rst#section-1b-development-mode-install), start first by installing [Anaconda](https://docs.anaconda.com/anaconda/install/). With anaconda installed, clone CaImAn's repo, create an environment for CaImAn via conda, activate the environment, and install CaImAn's required packages. Once CaImAn is installed, you can clone this repo and install via pip. See below for the exact commands used to install CaImAn and this repo on a UNIX system (MacOS or Linux).

```bash
# Installing CaImAn
git clone git@github.com:thicclatka/CaImAn.git # use my fork of CaImAn
cd /path/to/CaImAn/
conda env create -f environment.yml -n caiman
conda activate caiman # activate environment
pip install -e . # remove -e if you don't want to install in editable mode

# Installing CLAH Image Analysis
git clone git@github.com:thicclatka/CLAH_IA.git
pip install -e /path/to/CLAH_IA # remove -e if you don't want to install in editable mode
```

To set up [ROICaT](https://github.com/RichieHakim/ROICaT), activate caiman environment, and run this:

```bash
conda activate caiman # activate environment
pip install roicat[all]==1.3.3
pip install git+https://github.com/RichieHakim/roiextractors
```

## Order of Operations

For single sessions:

1. [Moco2segDict (M2SD)](#motion-correction-to-segmented-dictionary-moco2segdict)
2. [quickTuning (QT)](#quick-tuning) (need .tdml to run)

For multiple sessions:

1. [M2SD](#motion-correction-to-segmented-dictionary-moco2segdict) -> [QT](#quick-tuning) (Using data directory with single session folders)
2. [wrapMultSessStruc](#wrapmultsessstruc)
3. [cellRegistrar_wROICaT](#cell-registrar-wroicat) (Using data directory with multisession folders from here)
4. [PostCR_CueCellFinder](#post-cell-registrar-cue-cell-finder-postcr_cuecellfinder)
5. [CR_CI_collater](#cell-registrar-cluster-info-collater-cr_ci_collater)

### Structure for folder/path names

For directories where each subdirectory is holding a single session of data:

/path/to/dir/[DIR_NAME_FT_EXPERIMENT_KEYWORDS]/[DATE]\_[SUBJECT_ID]\_[EXPERIMENT]

- [DIR_NAME_FT_EXPERIMENT_KEYWORDS] is the name of the directory holding the experiment data
  - e.g. eOPN3_CA3_2408 or Alzheimers_20240912-30_CA3
  - there is no requirement for a specific format, but try to be consistent with a set of experiments
- [DATE] is the date of the experiment
  - format: YYMMDD (e.g. 240912 for September 12, 2024)
- [SUBJECT_ID] is the subject ID
- [EXPERIMENT] is the experiment name
  - e.g. cueShiftOmitIAA-001
  - no requirement for a specific format, but try to be consistent with a set of experiments
- **NOTE:** Underscore are required between the subject ID, and experiment name

For directories where each subdirectory holds multiple sessions or holding multSessSegStruc per subject:

/path/to/dir/\_MS\_[DIR_NAME_FT_EXPERIMENT_KEYWORDS]\_[BRAIN_REGION]/[SUBJECT_ID]\_[NUM_SESSIONS]

- \_MS\_
  - prefix for multisession directories
  - notes each folder within directory contains multSessSegStruc (see [wrapMultSessStruc](#wrapmultsessstruc))
  - automatically prepended by [wrapMultSessStruc](#wrapmultsessstruc)
- [DIR_NAME_FT_EXPERIMENT_KEYWORDS]
  - see above
  - however, in this case, recommended to use words like OPTO, eOPN3, AD (Alzheimers), Ag (Aged)
  - if output_folder is not specified (by default, it is not), [wrapMultSessStruc](#wrapmultsessstruc) will prompt user to input experiment keywords for output folder name
  - these keywords are required in order to trigger certain conditions to run in [PostCR_CueCellFinder](#post-cell-registrar-cue-cell-finder-postcr_cuecellfinder) and [CR_CI_collater](#cell-registrar-cluster-info-collater-cr_ci_collater)
- [BRAIN_REGION]
  - if output_folder is not specified (by default, it is not), [wrapMultSessStruc](#wrapmultsessstruc) will prompt user to input brain region
    - between CA3 & DG
  - this is required since it is used to set certain parameters for [CellRegistrar_wROICaT](#cell-registrar-wroicat)
- [SUBJECT_ID]\_[NUM_SESSIONS]
  - created automatically by [wrapMultSessStruc](#wrapmultsessstruc)
- **NOTE:** Underscores are required between each element of the folder name

## Functions

### [TifStack Functions](./CLAH_ImageAnalysis/tifStackFunc/__init__.py)

#### [Motion Correction to Segmented Dictionary (Moco2segDict)](./CLAH_ImageAnalysis/tifStackFunc/MoCo2segDict.py)

- I:
  - raw movie stored in H5 file
- O:
  - tifs of average activity
  - evalution of segmented components (CountourPlot_CompEval.png)
  - mmaps of motion corrected movie
  - motioned corrected movie stored in H5 (H5 file with \_eMC tag)
  - segmented dictionary (segDict)

How to run:

```console
usage: MoCo2segDict.py  [-h] [-p PATH] [-s2p SESS2PROCESS] [-mc MOTION_CORRECT]
                        [-sg SEGMENT] [-n4mc N_PROC4MOCO] [-n4cnmf N_PROC4CNMF]
                        [-cat CONCATENATE]

Run Moco2segDict with command line arguments.

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
  -mc MOTION_CORRECT, --motion_correct MOTION_CORRECT
                        Whether to perform motion correction.
                        Default is false. (e.g. -mc y, -mc yes, -mc true to enable)
  -sg SEGMENT, --segment SEGMENT
                        Whether to perform segmentation.
                        Default is false. (e.g. -seg y, -seg yes, -seg true to enable)
  -n4mc N_PROC4MOCO, --n_proc4MOCO N_PROC4MOCO
                        How many processors to use for motion correction.
                        Default is 26 processes
  -n4cnmf N_PROC4CNMF, --n_proc4CNMF N_PROC4CNMF
                        How many processors to use for CNMF segmentation.
                        Default is using all available processes.
  -cat CONCATENATE, --concatenate CONCATENATE
                        Concatenate H5s into a single H5 before motion correction,
                        but create 2 segDicts. ONLY USE THIS TO COMBINE THE RESULTS
                        FOR THE SAME SUBJECT ID ACROSS 2 SESSIONS.
  -psv PREV_SD_VARNAMES, --prev_sd_varnames PREV_SD_VARNAMES
                        Use the old variable names for the segDict (i.e. A, C, S, etc).
                        Default is False, in which names will be A_Spatial, C_Temporal, etc.
```

### [Unit Analysis](./CLAH_ImageAnalysis/unitAnalysis/__init__.py)

#### [Quick Tuning](./CLAH_ImageAnalysis/unitAnalysis/quickTuning.py)

- I:
  - segmented dictionary (segDict)
  - treadmill/behavior log/json file (.tdml)
- O:
  - treadmill behavior dictionary (treadBehDict)
  - lap info dictionary (lapDict)
  - Cue Cell info & arrays dictionary (CueCellFinderDict)
  - cueShiftStruc
  - Figures (stored in /path/to/segDict/Figures)

How to run:

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
  -f FPS, --fps FPS     Window size (in frames) used for smoothing for the event/peak
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

#### [WrapMultSessStruc](./CLAH_ImageAnalysis/unitAnalysis/wrapMultSessStruc.py)

- I:
  - cueShiftStruc
- O:
  - multSessSegStruc (location is based on -output_folder parameter)

How to run:

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

#### [Post Cell Registrar Cue Cell Finder (PostCR_CueCellFinder)](./CLAH_ImageAnalysis/unitAnalysis/PostCR_CueCellFinder.py)

- I:
  - multSessSegStruc
- O:
  - PCRTrigSigDict
  - Figures per subject (/path/to/multSessSegStruc/Figures)
  - Figures across subjects (/path/to/MultSess/dir/~GroupData)

How to run:

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

### [Cell Registration](./CLAH_ImageAnalysis/registration/__init__.py)

#### [Cell Registrar w/ROICaT](./CLAH_ImageAnalysis/registration/cellRegistrar_wROICaT.py)

- I:
  - multSessSegStruc
- O:
  - json with cluster info (\_cluster_info_ROICaT.json)
  - results dictionary (\_results_ROICaT)
  - ROICaT rundata dictionary (\_rundata_ROICaT)
  - json with CR_wROI parameters (CRwROIparams.json)
  - Figures (stored in /path/to/multSessSegStruc/\_Figures_ROICaT)

How to run:

```console
usage: cellRegistrar_wROICaT.py   [-h] [-p PATH] [-s2p SESS2PROCESS]
                                  [-sf SESSFOCUS] [-G USEGPU] [-v VERBOSE]

Run CellRegistar with ROICaT with command line arguments.

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
  -sf SESSFOCUS, --sessFocus SESSFOCUS
                        Set number of sessions to analyze.
                        Default is None, which will analyze all sessions found within
                        multSessSegStruc.
  -G USEGPU, --useGPU USEGPU
                        Whether to use GPU for ROICaT functions. Default is True
  -v VERBOSE, --verbose VERBOSE
                        Whether to print verbose output for ROICaT functions.
                        Default is True
```

#### [Cell Registrar Cluster Info Collater (CR_CI_collater)](./CLAH_ImageAnalysis/registration/CR_CI_collater.py)

- I:
  - json with cluster info (\_cluster_info_ROICaT.json)
- O:
  - csv with cluster info for all subjects (/path/to/\_MS_dir/~GroupData/ClusterInfo_all.csv)
  - csv with cluster info averages by group (/path/to/\_MS_dir/~GroupData/ClusterInfo_means.csv)

How to run:

```console
usage: CR_CI_collater.py [-h] [-p PATH] [-s2p SESS2PROCESS] [-4p FORPRES]

Run CellRegistrar Cluster Info Collater with command line arguments.

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
  -4p FORPRES, --forPres FORPRES
                        Whether to export .svg for figures in addition to the usual
                        png output. Default is False.
```

## [GUI](./CLAH_ImageAnalysis/GUI)

TODO:

- [ ] Test/clean up already existing GUI functions
- [ ] add web-based GUI for running functions over remote connections

## [Decoder](./CLAH_ImageAnalysis/decoder)

TODO:

- [ ] Add usage/explanation for decoder funcs

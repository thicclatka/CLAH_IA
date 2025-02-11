# Structure for folder/path names

## Single session

/ path / to / dir / [DIR_NAME_FT_EXPERIMENT_KEYWORDS] / [DATE]\_[SUBJECT_ID]\_[EXPERIMENT]

### [DIR_NAME_FT_EXPERIMENT_KEYWORDS]

- is the name of the directory holding the experiment data
- e.g. eOPN3_CA3_2408 or Alzheimers_20240912-30_CA3
- there is no requirement for a specific format, but try to be consistent with a set of experiments

### [DATE]

- format: YYMMDD (e.g. 240912 for September 12, 2024)

### [SUBJECT_ID]

### [EXPERIMENT]

- is the experiment name
- e.g. cueShiftOmitIAA-001
- no requirement for a specific format, but try to be consistent with a set of experiments

**_NOTE: Underscores are required between the subject ID, and experiment name_**

## Multisession

/ path / to / dir / \_MS\_[DIR_NAME_FT_EXPERIMENT_KEYWORDS]\_[BRAIN_REGION] / [SUBJECT_ID]\_[NUM_SESSIONS]

### \_MS\_

- prefix for multisession directories
- notes each folder within directory contains multSessSegStruc (see wrapMultSessStruc)
- automatically prepended by [wrapMultSessStruc](CLIs/unitAnalysis.md#wrapmultsessstruc)

### [DIR_NAME_FT_EXPERIMENT_KEYWORDS]

- recommended to use words like OPTO, eOPN3, AD (Alzheimers), Ag (Aged)
- if output_folder is not specified (by default, it is not), [wrapMultSessStruc](CLIs/unitAnalysis.md#wrapmultsessstruc) will prompt user to input experiment keywords for output folder name
- these keywords are required in order to trigger certain conditions to run in [PostCR_CueCellFinder](CLIs/unitAnalysis.md#post-cell-registrar-cue-cell-finder-postcr_cuecellfinder) and CR_CI_collater

### [BRAIN_REGION]

- if output_folder is not specified (by default, it is not), [wrapMultSessStruc](CLIs/unitAnalysis.md#wrapmultsessstruc) will prompt user to input brain region
- between CA3 & DG
- this is required since it is used to set certain parameters for CellRegistrar_wROICaT

### [SUBJECT_ID]\_[NUM_SESSIONS]

- created automatically by [wrapMultSessStruc](CLIs/unitAnalysis.md#wrapmultsessstruc)

**_NOTE: Underscores are required between each element of the folder name_**

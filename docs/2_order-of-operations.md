# Order of Operations

## Terms

- Single session: directories where each subdirectory holds a single session of data
- Multisession: directories where each subdirectory holds multiple sessions of data

## Single session

1. Moco2segDict (M2SD)
2. quickTuning (QT) (need .tdml to run)

## Multisession

1. M2SD -> QT (Using data directory with single session subdirectories)
2. wrapMultSessStruc
3. cellRegistrar_wROICaT (Using data directory with multisession subdirectories from here)
4. PostCR_CueCellFinder
5. CR_CI_collater

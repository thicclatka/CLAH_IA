# Cell Registration

## [Cell Registrar w/ROICaT](https://github.com/thicclatka/CLAH_IA/blob/main/CLAH_ImageAnalysis/registration/cellRegistrar_wROICaT.py)

### Input

- multSessSegStruc

### Output

- json with cluster info (\_cluster_info_ROICaT.json)
- results dictionary (\_results_ROICaT)
- ROICaT rundata dictionary (\_rundata_ROICaT)
- json with CR_wROI parameters (CRwROIparams.json)
- Figures (stored in /path/to/multSessSegStruc/\_Figures_ROICaT)

### CLI

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

## [Cell Registrar Cluster Info Collater (CR_CI_collater)](https://github.com/thicclatka/CLAH_IA/blob/main/CLAH_ImageAnalysis/registration/CR_CI_collater.py)

### Input

- json with cluster info (\_cluster_info_ROICaT.json)

### Output

- csv with cluster info for all subjects (/path/to/\_MS_dir/~GroupData/ClusterInfo_all.csv)
  - csv with cluster info averages by group (/path/to/\_MS_dir/~GroupData/ClusterInfo_means.csv)

### CLI

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

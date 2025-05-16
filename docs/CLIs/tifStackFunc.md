# Motion Correction & Segmentation (CLI)

## [Motion Correction to Segmented Dictionary (Moco2segDict)](https://github.com/thicclatka/CLAH_IA/blob/main/CLAH_ImageAnalysis/tifStackFunc/MoCo2segDict.py)

### Input

- raw movie stored in H5 file (2p) or ISXD (1p)

### Output

- tifs of average activity
- evalution of segmented components (CountourPlot_CompEval.png)
- mmaps of motion corrected movie
- motioned corrected movie stored in H5 (H5 file with \_eMC tag)
- segmented dictionary (segDict) as .pkl & .h5
- see [section below](#cli) for non-default output options

### CLI

```console
usage: MoCo2segDict.py  [-h] [-p PATH] [-s2p SESS2PROCESS] [-mc MOTION_CORRECT]
                        [-sg SEGMENT] [-n4mc N_PROC4MOCO] [-n4cnmf N_PROC4CNMF]
                        [-cat CONCATENATE]

Run Moco2segDict with command line arguments.

options:
  -h, --help            show this help message and exit
  -p PATH, --path PATH  Path to the data directory (e.g. /path/to/data). Default will prompt user to choose path. NOTE: Pick directory which holds session folders; do not pick/set a session folder.
  -s2p SESS2PROCESS, --sess2process SESS2PROCESS
                        List of sessions to process. Write in format '1,2,3', '1-3', or '1,2-5' to select by specific session number. Input all or ALL to process all eligible sessions that are available within
                        the set path. Default will prompt user to choose.
  -fs FROM_SQL, --from_sql FROM_SQL
                        Whether script is being run from SQL scheduler. Default is False.
  -mc MOTION_CORRECT, --motion_correct MOTION_CORRECT
                        Whether to perform motion correction. Default is false. (e.g. -mc y, -mc yes, -mc true to enable)
  -sg SEGMENT, --segment SEGMENT
                        Whether to perform segmentation. Default is false. (e.g. -seg y, -seg yes, -seg true to enable)
  -n4mc N_PROC4MOCO, --n_proc4MOCO N_PROC4MOCO
                        How many processors to use for motion correction. Default is 26 processes
  -n4cnmf N_PROC4CNMF, --n_proc4CNMF N_PROC4CNMF
                        How many processors to use for CNMF segmentation. Default is using all available processes.
  -cat CONCATENATE, --concatenate CONCATENATE
                        Concatenate H5s into a single H5 before motion correction, but create 2 segDicts. ONLY USE THIS TO COMBINE THE RESULTS FOR THE SAME SUBJECT ID ACROSS 2 SESSIONS.
  -mci MC_ITER, --mc_iter MC_ITER
                        Number of iterations for motion correction. Default is 1. WARNING: this is not the same as the number of iterations for rigid motion correction (niter_rig) within caiman and it can add
                        to the total processing time.
  -ow OVERWRITE, --overwrite OVERWRITE
                        Overwrite existing files (segDicts, sqz_H5s, tifs, mmaps, etc). Default is False.
  -cm COMPUTE_METRICS, --compute_metrics COMPUTE_METRICS
                        Calculate motion correction metrics. Default is False.
  -crp USE_CROPPER, --use_cropper USE_CROPPER
                        Use the cropping utility for 1photon data (.isxd files). Default is False.
  -sc SEPARATE_CHANNELS, --separate_channels SEPARATE_CHANNELS
                        Whether to motion correct channels separately. Only applicable for 2photon data with 2 channels. Default is False.
  -ers EXPORT_POSTSEG_RESIDUALS, --export_postseg_residuals EXPORT_POSTSEG_RESIDUALS
                        Whether to export the post-segmentation residuals as a video file. Default is False.
```

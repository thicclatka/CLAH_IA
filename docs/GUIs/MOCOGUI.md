# Motion Correction & Segmentation (GUI)

## Web App

The M2SD Web App provides a streamlined interface for running motion correction and segmentation analysis on imaging data. The app is built using Streamlit and can be run as a system service for continuous availability.

### Requirements

- configured [SQLJobScheduler](https://github.com/thicclatka/SQLJobScheduler)
- System service setup (via systemd)
- a port available for web access

### Key Features

- **Path Search & Selection**: Easily search and select data paths from a SQLite database cache
- **Session Processing**: Choose to process all sessions or select specific ones
- **Main Options**:
    - Motion Correction (MC)
    - Segmentation (SG)
    - Overwrite existing files
- **Advanced Settings**:
    - Concatenate files
    - Compute metrics
    - Export post-segmentation residuals
    - Handle separate channels
    - Motion correction iterations

### Job Scheduling System

The app uses SQL Job Scheduler to manage analysis tasks:

1. **Job Queue**: When analysis is initiated, jobs are added to a SQL-based queue
2. **Email Notifications**: Users must provide an email address to receive notifications about:
   - Job start
   - Analysis progress
   - Completion status
   - Any errors encountered

### Access

The web app can be accessed at: `http://[server-address]:[port]/m2sd`

### Usage Flow

1. Create or refresh database of eligible paths for analysis
2. Search and select sessions to analyze
3. Choose sessions to process
4. Configure analysis options
5. Click "Run Analysis" to queue jobs
6. Monitor progress via email notifications

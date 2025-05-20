# SegDict Utility (GUI)

## Web App

The SegDict Utility web app provides a way to view and evaluate segments saved in the segDict that result after running [Motion Correction and Segmentation](../CLIs/unitAnalysis.md).

### Requirements

- System service setup (via systemd)
- A port available for web access

### Key Features

#### Component Evaluation

- View and evaluate individual components from the segDict
- Accept/reject/undecided classification for each component
- Multiple evaluation methods:
    - CNMF Component Evaluation
    - Accepted Labels from segDict (ISX)
    - Neural Network Component Evaluation
    - Previously Done Accepted/Rejected Components

#### Visualization Tools

- Spatial Profile View:
    - Display component spatial footprints
    - Toggle cell number overlay
    - Show all components or only accepted ones
    - Multiple colormap options
    - Integration with downsampled images

- Temporal Profile View:
    - Display calcium signals for each component
    - Peak detection with two algorithms:
        - Scipy-based peak detection
        - Iterative differences method
    - Signal processing options:
        - Baseline adjustment
        - Smoothing via Savitzky-Golay filter
        - Spike deconvolution visualization
    - Peak transient profile analysis

#### Neural Network Integration

- Model Training:
    - Add sessions to training set
    - Train new models with optimized hyperparameters
    - Cross-validation performance metrics
- Model Management:
    - Save trained models
    - Load existing models
    - Evaluate components using trained models

#### Export Capabilities

- Component Selection Export:
    - Save accepted/rejected/undecided components as JSON
    - Preview export data
- Peaks Dictionary Export:
    - Save peak detection parameters and results
    - Include baseline and smoothing settings
    - Preview peak data

### Access

The web app can be accessed at `http://[server-address]:[port]/segdict`. Based on what is in [Application settings](https://github.com/thicclatka/CLAH_IA/blob/main/SystemdServices/app_settings.json), the default port is `8504`.

### Usage

1. **Session Selection**
   - Search and select a session containing segDict files
   - View session details and component counts

2. **Component Evaluation**
   - Select components using the ROI selector
   - Classify components as accepted/rejected/undecided
   - Use existing evaluations or create new ones

3. **Visualization**
   - Toggle between spatial and temporal views
   - Adjust visualization parameters
   - Analyze peak characteristics

4. **Export**
   - Export component selections
   - Export peak detection results
   - Preview export data before saving

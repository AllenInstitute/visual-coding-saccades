# Allen Institute Visual Coding Dataset Saccade Analysis
*Primary Author: [Chase King](https://chaseking.me)*

## Development To-Do:
1. [ ] Integrate eye tracking into the AllenSDK.
2. [ ] Delete `eye_tracking/local_eye_data_repository.py` (part of legacy eye tracking).
3. [ ] Test `identify_responding_cells_across_sessions.py`.

## Installation
This project was developed in Python 3.7.10 using [AllenSDK](https://allensdk.readthedocs.io/en/latest/) version 2.12.3. [Conda](https://conda.io/) will set up a virtual environment with the exact version of Python used for development along with all the dependencies needed to run the code.

1.  [Download and install Conda](https://conda.io/docs/download.html).

2.  Change your directory to your clone of this repo.
    ```bash
    cd visual-coding-saccades
    ```

3.  Create a Conda environment with Python 3.7.10.
    ```bash
    conda create -n visual-coding-saccades python=3.7.10
    ```

4.  Now activate the Conda environment. You will need to activate the Conda environment in each terminal in which you want to run code from this repo.
    ```bash
    conda activate visual-coding-saccades
    ```

5.  Install the required dependencies.
    ```bash
    pip install -r requirements.txt
    ```

Congratulations! You should now have the right Python version; you can confirm with:

```bash
(visual-coding-saccades) % python -V
Python 3.7.10
```

## Getting Started

The first thing to note is various parameters that can be edited in [`saccade_config.py`](saccade_config.py). Open up this file and change the relevant file paths.

### Data Loading

**Loading session data.** Sessions are loaded from the Allen Brain Observatory Visual Coding datasets, and then cached in internal files to make subsequent analyses faster. This is done in [`/data_loading/parallel_data_loading.py`](/data_loading/parallel_data_loading.py) (this is run in parallel to make the compute time much faster). On a 10-CPU machine, this process took about 15 minutes to load 837 sessions. Since this data is not uploaded to this repository, you will need to run

```bash
python data_loading/parallel_data_loading.py
```

(Note this is done from the main directory, i.e., `visual-coding-saccades`.)

**Loading neural activity data (optional).** Now that this session data has been cached, you can optionally re-run a Python script to classify cells as saccade-responsive. This is a very computationally-expensive operation; only run if you change relevant parameters; it takes ~2 hours on my 10-core laptop.

```bash
cd data_loading
python identify_responding_cells_across_sessions.py
```

### Data Analysis
in the [`analysis`](/analysis) folder, you will find various notebooks that can be used to perform the analyses and generate the figures in the paper. Navigate to this folder for more information.
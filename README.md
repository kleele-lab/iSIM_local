# Deconvolution

Code to deconvolve iSIM images.

## Getting started

1. Copy `script.py` to a file called `script_YOUR_NAME.py`.
1. Open the script file and change the `folder` variable to point to the folder containing your image stacks.
1. Run the script `python script_YOUR_NAME.py`

## Installation

### Assumptions

- We are installing onto [TeslaDesk](https://www.epfl.ch/labs/leb/research/tesladesk-server/)
- Python 3.10.8 or greater is installed
- CUDA is installed

## Steps

1. Copy the contents of this folder to `C:\Internal\deconvolution`.
1. Create a new Python virtual environment inside `C:\Internal\.envs` called `decon_310` with the command: `python -m venv C:\Internal\.envs\decon_310`.
1. Activate the new virtual environment: `C:\Internal\.envs\decon_310\Scripts\activate`.
1. Install the requirements: `pip install -r requirements.txt`

## CUDA and Tensorflow

Use the following chart to determine which version of Tensorflow to install: https://www.tensorflow.org/install/source#gpu

The version that you wish to install should be updated in requirements.txt.

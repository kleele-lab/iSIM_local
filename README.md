Code to deconvolve vi and tif images on the Euler cluster.

# Installation
Copy this GitHub repo in your $HOME on the cluster and cd into it:
```
$git clone https://github.com/kleele-lab/iSIM_local.git
$cd iSIM_local
```
Load the appropriate modules on the cluster :
```
$module load stack/2024-06
$module load python_cuda/3.11.6
$module load jdk
```
Then, create an empty python environment and name it decon-env:

```commandline
$python -m venv decon-env
$source ./decon-env/bin/activate
$pip install -r ./deconvolution/requirements.txt
$pip install python-bioformats
```
You will get an error related to the numpy version. Please ignore it : the installation will be complete and the deconvolution process will happen without issue.
Test the installation with an interactive job on Euler:

```commandline
nmarounina@eu-login-39:~/iSIM_local$ srun -G 1 --mem-per-cpu=16G --pty bash
srun: job 38627224 queued and waiting for resources
srun: job 38627224 has been allocated resources
nmarounina@eu-lo-g3-043:~/iSIM_local$ 
nmarounina@eu-lo-g3-043:~/iSIM_local$ module load stack/2024-06 python_cuda/3.11.6 jdk
nmarounina@eu-lo-g3-043:~/iSIM_local$ source ./decon-env/bin/activate
(decon-env) nmarounina@eu-lo-g3-043:~/iSIM_local$ python ./deconvolution/script_folder.py <path-to-folder>
```
# Usage

To submit a job on Euler, please make sure that you named your python environment decon-env.
Then, the job is submitted using one of the following bash scripts:

```commandline
sbatch run_image_decon.sh </path/to/a/specific/file.vsi>
```
This script will deconvolve only the image provided as input. The following script will take in an entire folder and will deconvolve all the tif or vsi images that it will find inside :

```commandline
sbatch run_folder_decon.sh </path/to/a/folder/with/files/to/deconvolve>
```

Upon a successful job submission, you will be provided with a job ID. The logs of the deconvolution process (including errors) will be available in a flie names:
```commandline
slurm-<jobID>.out
```
The file will be located in the directory from where you ran the sbatch command.

[//]: # (================================================== PREVIOUS VERSION )

[//]: # (# Deconvolution)

[//]: # ()
[//]: # (Code to deconvolve iSIM images.)

[//]: # ()
[//]: # (## Getting started)

[//]: # ()
[//]: # (1. Copy `script.py` to a file called `script_YOUR_NAME.py`.)

[//]: # (1. Open the script file and change the `folder` variable to point to the folder containing your image stacks.)

[//]: # (1. Run the script `python script_YOUR_NAME.py`)

[//]: # ()
[//]: # (## Installation)

[//]: # ()
[//]: # (### Assumptions)

[//]: # ()
[//]: # (- We are installing onto [TeslaDesk]&#40;https://www.epfl.ch/labs/leb/research/tesladesk-server/&#41;)

[//]: # (- Python 3.10.8 or greater is installed)

[//]: # (- CUDA is installed)

[//]: # ()
[//]: # (## Steps)

[//]: # ()
[//]: # (1. Copy the contents of this folder to `C:\Internal\deconvolution`.)

[//]: # (1. Create a new Python virtual environment inside `C:\Internal\.envs` called `decon_310` with the command: `python -m venv C:\Internal\.envs\decon_310`.)

[//]: # (1. Activate the new virtual environment: `C:\Internal\.envs\decon_310\Scripts\activate`.)

[//]: # (1. Install the requirements: `pip install -r requirements.txt`)

[//]: # ()
[//]: # (## CUDA and Tensorflow)

[//]: # ()
[//]: # (Use the following chart to determine which version of Tensorflow to install: https://www.tensorflow.org/install/source#gpu)

[//]: # ()
[//]: # (The version that you wish to install should be updated in requirements.txt.)

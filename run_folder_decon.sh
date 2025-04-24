#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=32G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:16G

module purge

module load stack/2024-06
module load python_cuda/3.11.6
module load jdk

source ./decon-env/bin/activate

# Avoiding the JIT error
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_EULER_ROOT
export CUDA_DIR=$CUDA_EULER_ROOT

# Checking the inputs
if [ $# -eq 0 ];
then
  echo "$0: Missing arguments"
  exit 1
elif [ $# -gt 1 ];
then
  echo "$0: Too many arguments: $@"
  exit 1
else
  abspath=$(realpath $1)
  echo "Reading folder: $1"
fi

# Enter folder with python script
cd deconvolution

# Check for images in mother dir
python script_folder.py $abspath

# Return to initial location
cd ..

exit 0

#use via: sbatch run_folder_decon.sh /nfs/nas22/fs2202/biol_bc_kleele_2/path_to_image


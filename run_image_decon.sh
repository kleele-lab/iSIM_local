#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:16G

module purge

module load stack/2024-06
module load python_cuda/3.9.18
module load jdk

source ./decon-env/bin/activate

# Avoiding the JIT error
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_EULER_ROOT
export CUDA_DIR=$CUDA_EULER_ROOT

# Make sure there is only one argument, i.e., folder path
if [ $# -eq 0 ];
then
  echo "$0: Missing arguments"
  exit 1
elif [ $# -gt 1 ];
then
  echo "$0: Too many arguments: $@"
  exit 1
fi

if [[ "$1" == *.ome.tif ]]
then
	echo "reading image $1"
elif [[ "$1" == *.vsi ]]
then
	echo "reading image $1"
else
	echo"$0: Wrong arguments"
fi

# Enter folder with python script
cd deconvolution

# Run decon
python script_image.py $1

# Return to initial location
cd ..

exit 0

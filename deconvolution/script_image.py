import os
import sys
import itertools
import javabridge as jb
import bioformats as bf

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow

gpus = tensorflow.config.list_physical_devices('GPU')
for gpu in gpus:
    tensorflow.config.experimental.set_memory_growth(gpu, True)

from pathlib import Path
import cuda_decon

# Import
file = sys.argv[1]
file=Path(file)

background = None #change to "median" (or option below) to trigger the median substraction before the decon
# background      0-3: otsu with this scaling factor
# background      > 3: fixed value
# background 'median': median of each z-stack as bg
jb.start_vm(class_path=bf.JARS)
cuda_decon.decon_ome_stack(file.as_posix(), background=background)
jb.kill_vm()

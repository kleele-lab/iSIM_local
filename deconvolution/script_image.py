import os
import sys

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow

gpus = tensorflow.config.list_physical_devices('GPU')
for gpu in gpus:
    tensorflow.config.experimental.set_memory_growth(gpu, True)

from pathlib import Path
import cuda_decon

# Import
folder = sys.argv[1]
files = Path(folder).rglob('*.ome.tif')

# parameters = {
#     'background': "median",
# }

background = "median"
# background      0-3: otsu with this scaling factor
# background      > 3: fixed value
# background 'median': median of each z-stack as bg

for file in files:
    if not file.name.startswith('._'):
        if not 'decon' in file.name:
            print(file.name)
            print(file.as_posix())
            cuda_decon.decon_ome_stack(file.as_posix(), background=background)

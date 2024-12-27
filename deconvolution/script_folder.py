import os
import sys

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow

gpus = tensorflow.config.list_physical_devices('GPU')
for gpu in gpus:
    tensorflow.config.experimental.set_memory_growth(gpu, True)

import cuda_decon

# Import
folder = sys.argv[1]


def list_files(dir):
    r = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            if name.endswith('.ome.tif'):
                if not name.startswith('._'):
                    if not 'decon' in name:
                        r.append(os.path.join(root, name))
    return r


# parameters = {
#     'background': "median",
# }

background = "median"

# background      0-3: otsu with this scaling factor
# background      > 3: fixed value
# background 'median': median of each z-stack as bg

img_list = list_files(folder)
print("Found images:")
print(*img_list, sep="\n")
print("STARTING DECON!")

for file in img_list:
    print(file)
    cuda_decon.decon_ome_stack(file, background=background)

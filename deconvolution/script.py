#!‪C:\Internal\.envs\decon_310\Scripts\python.exe

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

os.system("C:/Internal/.envs/decon_310/Scripts/activate")
os.system("cd C:/Internal/deconvolution")
import tensorflow

gpus = tensorflow.config.list_physical_devices('GPU')
for gpu in gpus:
    tensorflow.config.experimental.set_memory_growth(gpu, True)


from pathlib import Path
from prepare import get_filter_zone_ver_stripes, prepare_one_slice
import cuda_decon

# Import
folder = r"W:\iSIMstorage\Users\Juan\230503_MEF_Opa1\Data"

files = Path(folder).rglob('*0.ome.tif')

parameters = {
    'background': "median",
}
# background      0-3: otsu with this scaling factor
# background      > 3: fixed value
# background 'median': median of each z-stack as bg

for file in files:

    if not 'decon' in file.name:

        print(file.name)
        print(file.as_posix())
        cuda_decon.decon_ome_stack(file.as_posix(), params=parameters)

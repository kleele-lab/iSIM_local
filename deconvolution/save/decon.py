from skimage import io
from file_handling import deconvoleStack
import cuda_decon

# stack = io.imread(r"Z:\iSIMstorage\Users\Dora\20200821_yeast\FOV_1\cell_500mW_70pc_3_MMStack_Pos0_zstep0.25_t000001_c1.tif")
# print(stack.shape)

cuda_params = cuda_decon.CudaParams(background=50)
deconvoleStack(r"//lebnas1.epfl.ch/microsc125/iSIMstorage/Users/Tatjana/211107_U2OS_KdelRfpxMitoGfp_CtrlS1/_1/_1_MMStack_1024_crop.tif", mode='cuda', cuda_params=cuda_params)


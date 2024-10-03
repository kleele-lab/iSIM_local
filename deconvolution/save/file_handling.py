
import json
import os
from multiprocessing import Pool
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import xmltodict
import deconvolve
from tqdm import tqdm





def deconvoleStack(file: str, mode: str = 'cpu', cuda_params=None):
    """ Deconvolve the struct part of a stack of foci/struct data using the deconvolve function
    in the toolbox."""
    stack_struct =tifffile.imread(file, is_ome=False)
    print(stack_struct.shape)
    stackDecon = np.zeros(stack_struct.shape, dtype=np.uint16)
    if mode in ['gpu', 'cuda']:
        import cuda_decon
    if mode in ['gpu', 'cuda'] and cuda_params is None:
        cuda_params = cuda_decon.CudaParams()

    for frame in tqdm(range(stack_struct.shape[0])):
        if mode == 'cpu':
            stackDecon[frame, :, :] = deconvolve.full_richardson_lucy(stack_struct[frame, :, :])
        else:
            stackDecon[frame, :, :] = cuda_decon.richardson_lucy(stack_struct[frame, :, :],
                                                                 params=cuda_params)

    out_file = file[:-8] + '_decon.tiff'
    tifffile.imwrite(out_file, stackDecon)


def deconvolveFolder(folder, n_threads=10, mode='cpu', cuda_params=None, subfolder=None):
    """ Wrapper function for deconvolveOneFolder to allow for parallel computation """

    if isinstance(folder, list) and mode == 'cpu':
        with Pool(n_threads) as p:
            p.map(deconvolveOneFolder, folder, subfolder)
    elif isinstance(folder, list):
        for single_folder in folder:
            deconvolveOneFolder(single_folder, mode, cuda_params, subfolder)
    else:
        deconvolveOneFolder(folder, mode, cuda_params, subfolder)


def deconvolveOneFolder(folder, mode='cpu', cuda_params=None, subfolder=None, channel='network'):
    """ Deconvolve the struct frames in a folder of foci/struct data using the deconvolve function
    in the toolbox. """
    if mode == 'cuda':
        import cuda_decon

    if cuda_params is None and mode == 'cuda':
        print('Using default coda_params')
        cuda_params = cuda_decon.CudaParams(background=1.05, sigma=3.9/2.335)  # 0.92 mito 1 for caulo highlight
    print('sigma: ', cuda_params.kernel['sigma'], '\nbackground: ', cuda_params.background)

    print(folder)
    files, _ = get_files(folder)
    files = files[channel]
    for idx, file in enumerate(tqdm(files)):
        struct_img = io.imread(file)
        if mode == 'cpu':
            decon_img = deconvolve.full_richardson_lucy(struct_img)
        elif mode == 'cuda':
            decon_img = cuda_decon.richardson_lucy(struct_img, params=cuda_params)

        if cuda_params.after_gaussian:
            decon_img = cv2.GaussianBlur(decon_img, (0, 0), cuda_params.after_gaussian)

        if subfolder is None:
            out_file = file[:-8] + 'decon.tiff'
        else:
            os.makedirs(os.path.join(folder, subfolder), exist_ok=True)
            filename = os.path.basename(file)[:-12] + str(idx).zfill(4) + '.decon.tiff'
            out_file = os.path.join(folder, subfolder, filename)
            with open(os.path.join(folder, subfolder, 'params.txt'), 'w') as outp:
                outp.write(json.dumps(cuda_params.to_dict()))

        tifffile.imwrite(out_file, decon_img)
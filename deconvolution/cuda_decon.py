import posixpath
from flowdec import data as fd_data
from flowdec import restoration as fd_restoration
from scipy import ndimage
import numpy as np
from dataclasses import dataclass
import tifffile
import xmltodict
from tqdm import tqdm
from typing import Union
import os
import javabridge as jb
import bioformats as bf

from prepare import prepare_decon, get_filter_zone

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def _init_logger():
    """This is so that Javabridge doesn't spill out a lot of DEBUG messages
    during runtime.
    From CellProfiler/python-bioformats.
    """
    rootLoggerName = jb.get_static_field("org/slf4j/Logger",
                                         "ROOT_LOGGER_NAME",
                                         "Ljava/lang/String;")

    rootLogger = jb.static_call("org/slf4j/LoggerFactory",
                                "getLogger",
                                "(Ljava/lang/String;)Lorg/slf4j/Logger;",
                                rootLoggerName)

    logLevel = jb.get_static_field("ch/qos/logback/classic/Level",
                                   "WARN",
                                   "Lch/qos/logback/classic/Level;")

    jb.call(rootLogger,
            "setLevel",
            "(Lch/qos/logback/classic/Level;)V",
            logLevel)


@dataclass
class Params():
    """ Class for storing parameters to be used in deconvolution"""
    sigma: float = 3.9 / 2.335
    z_step: float = 0.2
    background: Union[int, str] = None #default value
    kernel = None  # to be initialized
    algo = None  # to be initialized



def richardson_lucy(image, params=None):
    original_data_type = image.dtype
    if params.algo is None:
        params.algo = fd_restoration.RichardsonLucyDeconvolver(image.ndim).initialize()

    if params.kernel is None:
        params.kernel = make_kernel(image, sigma=params.sigma)
        print(params.kernel)

    res = params.algo.run(fd_data.Acquisition(data=image, kernel=params.kernel['kernel_array']), niter=10).data
    return res.astype(original_data_type)


class Image():
    """ Generic class to store the data for the deconvolution 
    Image can be read from tif or vsi"""

    def __init__(self):
        data: np.ndarray = np.empty(1)
        metadata: dict = {}
        z_step: float = 0.2
        size_t: int = 0
        size_c: int = 0
        size_z: int = 0
        out_file_end: str = ''

    def read_tiff(self, file_dir: posixpath) -> None:
        tif = tifffile.TiffFile(file_dir)
        Image.metadata = tif.imagej_metadata
        
        if tif.ome_metadata is not None :
            my_dict = xmltodict.parse(tif.ome_metadata, force_list={'Plane'})
            
            size_t = int(my_dict['OME']['Image']["Pixels"]["@SizeT"])
            size_z = int(my_dict['OME']['Image']["Pixels"]["@SizeZ"])
            size_c = int(my_dict['OME']['Image']["Pixels"]["@SizeC"])
            try:
                Image.z_step = float(my_dict['OME']['Image']["Pixels"]['@PhysicalSizeZ'])
            except KeyError:
                print("Could not get z step size. Will put default 0.2")
                Image.z_step = 0.2

        # 'XYCZT' or 'XYZCT' ?
            dim_order = my_dict['OME']['Image']["Pixels"]["@DimensionOrder"]
        else :
            metadata_list=Image.metadata['Info'].split('\n')
            metadata_list = filter(None, metadata_list)
            my_dict=dict(i.split(sep='=',maxsplit=1) for i in metadata_list)

            size_t = int(my_dict[" SizeT "])
            size_z = int(my_dict[" SizeZ "])
            size_c = int(my_dict[" SizeC "])

            try:
                Image.z_step = float(my_dict['Z incrementValue '])
            except KeyError:
                print("Could not get z step size. Will put default 0.2")
                Image.z_step = 0.2
            dim_order = my_dict[" DimensionOrder "]


        Image.data = tif.asarray()

        # This is legacy code to handle tif dimension issues


        ndim = 2 if size_z == 1 else 3
        # Make standardized array with all dimensions
        # time [0], z[1], c[2] if not there, just missing depending on order
        # If it was recorded differently, swap dimensions

        if size_t == 1:
            Image.data = np.expand_dims(Image.data, 0)

        if size_z > 1 and size_c > 1 and dim_order == 'XYZCT':
            Image.data = np.moveaxis(Image.data, 1, 2)

        if size_z == 1 and len(Image.data.shape) < 5:
            Image.data = np.expand_dims(Image.data, 1)
        if size_c == 1 and len(Image.data.shape) < 5:
            Image.data = np.expand_dims(Image.data, 2)

        if size_t != Image.data.shape[0]:
            size_t = Image.data.shape[0]

        Image.size_z = size_z
        Image.size_t = size_t
        Image.size_c = size_c

        out_file = os.path.basename(file_dir).rsplit('.', 2)
        Image.out_file_end = out_file[0] + ".".join(["_decon", *out_file[1:]])

        print("\nReading tif:")
        print("Sizes, t, z, and c : ", size_t, size_z, size_c)
        print("Dim_order in the original file : ", dim_order)
        print("New shape of data going into decon", Image.data.shape, "\n")


    def read_vsi(self, file_dir: posixpath) -> None:

        logger = _init_logger()

        vsi = bf.ImageReader(file_dir)
        txt_metadata = bf.get_omexml_metadata(file_dir)
        txt_metadata = txt_metadata.encode('ascii', errors="ignore")

        my_dict = xmltodict.parse(txt_metadata, force_list={'Plane'})
        Image.metadata = my_dict

        size_t = int(my_dict['OME']['Image'][0]['Pixels']['@SizeT'])
        size_z = int(my_dict['OME']['Image'][0]['Pixels']['@SizeZ'])
        size_c = int(my_dict['OME']['Image'][0]['Pixels']['@SizeC'])
        size_x = int(my_dict['OME']['Image'][0]['Pixels']['@SizeX'])
        size_y = int(my_dict['OME']['Image'][0]['Pixels']['@SizeY'])

        dim_order = my_dict['OME']['Image'][0]['Pixels']['@DimensionOrder']

        try:
            Image.z_step = float(my_dict['OME']['Image'][0]["Pixels"]['@PhysicalSizeZ'])
        except KeyError:
            print("Could not get z step size. Will put default 0.2")
            Image.z_step = 0.2

        if dim_order == 'XYCZT':
            # data = np.zeros((size_t, size_z, size_c, size_y, size_x))
            data = np.zeros((size_x, size_y, size_c, size_z, size_t))
            for i_z in range(0, size_z):
                for i_t in range(0, size_t):
                    interm_data=vsi.read(series='0', z=i_z, t=i_t)
                    if len(interm_data.shape) == 2:
                        interm_data=np.expand_dims(interm_data, axis=-1)
                        data[:, :, :, i_z, i_t] = interm_data
                    else :
                        data[:, :, :, i_z, i_t] = interm_data
        else:
            raise "Dim order is NOT XYZCT"

        #flowdec needs a re-scale of the vsi data (vsi comes with values between 0 and 1)
        #here the re-scale is done up to the uint16 values
        maxval_uint16 = 65535
        Image.data = np.moveaxis(data, [0, 1, 2, 3, 4], [4, 3, 2, 1, 0]) * maxval_uint16
        Image.size_z = size_z
        Image.size_t = size_t
        Image.size_c = size_c

        Image.data = np.moveaxis(Image.data, [0, 1, 2, 3, 4], [0, 1, 2, 4, 3])
        out_file = os.path.basename(file_dir).rsplit('.', 2)
        Image.out_file_end = out_file[0] + ".".join(["_decon.tif"])

        print("\nReading vsi:")
        print("Sizes, t, z, and c : ", size_t, size_z, size_c)
        print("Dim_order in the original file : ", dim_order)
        print("New shape of data going into decon", Image.data.shape, "\n")


def make_kernel(image: np.ndarray, sigma=1.67, z_step=0.2):
    """
    Make a gaussian kernel that fits the psf of the microscope.

    The psf is designed to have the same number of dimensions then
    the image. The maximum shape of the psf is 100 in either dimension,
    this limit has been arbitrarily chosen. From tests, higher psf
    dimensions lead to high processing times and RAM consumptions, without
    visually improving on the results.
    """

    if image.ndim == 3:
        z_sigma = 0.48 / z_step
        sigma = [z_sigma, sigma, sigma]

    size = image.shape
    size = [min([100, x]) for x in size]

    # If even size dimensions crop to have a center pixel
    kernel = {'kernel_array': np.zeros(size, dtype=float),
              'sigma': sigma}
    kernel['kernel_array'][tuple(np.array(kernel['kernel_array'].shape) // 2)] = 1
    kernel['kernel_array'] = ndimage.gaussian_filter(kernel['kernel_array'], sigma=sigma)

    kernel['kernel_array'][kernel['kernel_array'] < 1e-6 * np.max(kernel['kernel_array'])] = 0
    kernel['kernel_array'] = np.divide(kernel['kernel_array'], np.sum(kernel['kernel_array'])).astype(np.float32)
    return kernel


def get_data_c(data_t, size_c, size_z, params):
    """
    Ensures a progressive supply of images to the deconvolution routine.

    Will provide z-stack of images, per channel.
    """

    for channel in range(size_c):
        data_c = data_t[:, channel, :, :]

        if params.background is not None :
            data_c = prepare_decon(data_c, background=params.background, destripe_zones=get_filter_zone)

        yield channel, data_c


def decon_ome_stack(file_dir, background):
    """
    Main deconvolution routine that reads in images, processes,
    deconvolves, and saves the data.

    """

    params = Params
    params.background = background

    Img = Image()
    if file_dir.split('.')[-1] == "tif":
        Img.read_tiff(file_dir)

    elif file_dir.split('.')[-1] == "vsi":
        Img.read_vsi(file_dir)

    # start of the deconvolution loop
    decon = np.empty_like(Img.data)

    for timepoint in tqdm(range(Img.size_t)):
        data_t = Img.data[timepoint, :, :, :, :]
        data_c_iterable = get_data_c(data_t, Img.size_c, Img.size_z,params)
        
        for channel, data_c in data_c_iterable:
            params.kernel = make_kernel(image=data_c, sigma=params.sigma, z_step=params.z_step)

            decon[timepoint, :, channel, :, :] = richardson_lucy(data_c, params=params)

    
    decon=decon.astype(np.uint16)

    with tifffile.TiffWriter(os.path.join(os.path.dirname(file_dir), Img.out_file_end), imagej=True) as dst:
        for decon_one in decon:
            frame = decon_one
            dst.write(
                frame,
                contiguous=True,
                metadata=Img.metadata,
                dtype=np.uint16
            )

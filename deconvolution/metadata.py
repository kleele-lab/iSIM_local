import tifffile
import xmltodict
import os
import uuid
import numpy as np


file_dir = r"Z:\iSIMstorage\Users\Willi\20230201_u2os_presets\FOV_20\FOV_20_MMStack.ome.tif"
# file_dir = r"Z:\iSIMstorage\Users\Willi\220624_Alignment\beads_1\beads_1_MMStack_Pos0.ome.tif"
out_file = r"Z:\iSIMstorage\Users\Willi\20230201_u2os_presets\FOV_20\FOV_20_MMStack_decon_debug.ome.tif"

def do_nothing(str):
    return str



with tifffile.TiffFile(file_dir) as tif:
    imagej_metadata = tif.imagej_metadata
    my_dict = xmltodict.parse(tif.ome_metadata, force_list={'Plane'})
    size_t = int(my_dict['OME']['Image']["Pixels"]["@SizeT"])
    size_z = int(my_dict['OME']['Image']["Pixels"]["@SizeZ"])
    size_c = int(my_dict['OME']['Image']["Pixels"]["@SizeC"])
    try:
        z_step = float(my_dict['OME']['Image']["Pixels"]['@PhysicalSizeZ'])
    except KeyError:
        print("Could not get z step size. Will put default 0.2")
        z_step = 0.2
    # 'XYCZT' or 'XYZCT' ?
    dim_order = my_dict['OME']['Image']["Pixels"]["@DimensionOrder"]
    data = tif.asarray()


    if size_t == 1:
        data = np.expand_dims(data, 0)

    if size_z > 1 and size_c > 1 and dim_order == 'XYZCT':
        data = np.moveaxis(data, 1, 2)

    if size_z == 1:
        data = np.expand_dims(data, 1)
    if size_c == 1:
        data = np.expand_dims(data, 2)
    # print("SHAPE AFTER", data.shape)

print(data.shape)

UUID = uuid.uuid1()

#Get metadata to transfer
with tifffile.TiffReader(file_dir) as reader:
    mdInfo = xmltodict.parse(reader.ome_metadata)
    mdInfo['OME']['Image']['@Name'] = os.path.basename(out_file).split('.')[0]
    for frame_tiffdata in mdInfo['OME']['Image']['Pixels']['TiffData']:
        frame_tiffdata['UUID']['@FileName'] = os.path.basename(out_file)
        frame_tiffdata['UUID']['#text'] =  'urn:uuid:' + str(UUID)  # mdInfo['OME']['Image']['Pixels']['TiffData'][0]['UUID']['#text']
    # mdInfo = xmltodict.unparse(mdInfo)

with tifffile.TiffWriter(os.path.join(os.path.dirname(file_dir), out_file), bigtiff=True, ome=True) as tif:
    tif.write(data, photometric='minisblack', compression='None')

mdInfo = xmltodict.unparse(mdInfo).encode(encoding='UTF-8', errors='strict')
tifffile.tiffcomment(os.path.join(os.path.dirname(file_dir), out_file), comment=mdInfo)
print("--> ", os.path.join(os.path.dirname(file_dir), out_file))








# with tifffile.TiffReader(file_dir) as reader:
#     mdInfo = xmltodict.parse(reader.ome_metadata,  attr_prefix='')
#     # # for frame_tiffdata in mdInfo['OME']['Image']['Pixels']['TiffData']:
#     # #     frame_tiffdata['UUID']['@FileName'] = os.path.basename(out_file)
#     mdInfo = xmltodict.unparse(mdInfo).encode(encoding='utf-8', errors='strict')
#     tifffile.tiffcomment(os.path.join(os.path.dirname(file_dir), out_file), comment=mdInfo)
#     print("--> ", os.path.join(os.path.dirname(file_dir), out_file))


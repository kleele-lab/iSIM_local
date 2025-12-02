
""" Mostly for destriping. Might be interesting to do automatic detection of the brightest spots in the future. """

import numpy as np
import cv2



def get_filter_zone(temp_fft, y_range_param=15, x_range_param=100):
    height, width = temp_fft.shape
    x_center = round(1018*width/2048)
    y_center = round(180*height/2048)
    # y_center = round(360*height/2048)

    x_center2 = round(width-x_center)
    y_center2 = round(height-y_center)
    x_range = round(x_range_param*(width/2048))
    y_range = round(y_range_param*(height/2048))
    filter_zone = np.zeros(temp_fft.shape, dtype=bool)
    filter_zone[(y_center-y_range):(y_center+y_range),
                (x_center-x_range):(x_center+x_range)] = True
    filter_zone[(y_center2-y_range):(y_center2+y_range),
                (x_center2-x_range):(x_center2+x_range)] = True
    return filter_zone

def get_filter_zone_ver_stripes(temp_fft, y_range_param=200, x_range_param=200):
    """Adapted destriping for the new horizontal.
    The orientation might be different for different ImageFlipper settings in Micro-Manager"""
    height, width = temp_fft.shape
    x_center = round(200*width/2048)
    y_center = round(1080*height/2048)
    # y_center = round(360*height/2048)

    x_center2 = round(width-x_center)
    y_center2 = round(y_center)
    x_range = round(x_range_param*(width/2048))
    y_range = round(y_range_param*(height/2048))
    filter_zone = np.zeros(temp_fft.shape, dtype=bool)
    filter_zone[(y_center-y_range):(y_center+y_range),
                (x_center-x_range):(x_center+x_range)] = True
    filter_zone[(y_center2-y_range):(y_center2+y_range),
                (x_center2-x_range):(x_center2+x_range)] = True
    return filter_zone


def prepare_decon(images, background=0.85, destripe_zones=get_filter_zone):
    if images.ndim > 2:
        for idx in range(images.shape[0]):
            images[idx, :, :] = prepare_one_slice(images[idx, :, :], background, destripe_zones)
    else:
        images = prepare_one_slice(images, background, destripe_zones)
    return images

def prepare_one_slice(image, background, filter_zone_source = get_filter_zone):
    if background == 'median':
        background = np.median(image)
    temp_fft = np.fft.fftshift(np.fft.fft2(image))
    filter_zone = filter_zone_source(temp_fft)
    # plt.imshow(np.divide(np.abs(temp_fft),np.max(np.abs(temp_fft))), vmax=0.0001)
    # plt.show()
    temp_fft[filter_zone] = 0
    filtered_img = np.abs(np.fft.ifft2(np.fft.fftshift(temp_fft)))
    # plt.imshow(np.divide(np.abs(temp_fft),np.max(np.abs(temp_fft))), vmax=0.0001)
    # plt.show()
    
    if background < 3:
        ret, mask = cv2.threshold(filtered_img.astype(np.uint16), 0, 1,
                                  cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        filtered_img = filtered_img - ret*background  # 80
    else:
        filtered_img = filtered_img - background
    filtered_img[filtered_img < 0] = 0
    return filtered_img

def prepare_image(image, background=0.85, median=3, gaussian=1.5):
    prep_img = prepare_decon(image, background).astype(np.uint16)
    prep_img = cv2.medianBlur(prep_img, median)
    prep_img = cv2.GaussianBlur(prep_img, (0, 0), gaussian)
    return prep_img

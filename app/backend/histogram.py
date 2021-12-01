import math
from tkinter.messagebox import showinfo

import numpy as np
from matplotlib import pyplot as plt
from osgeo import gdal
from skimage import exposure
from skimage.exposure import equalize_hist, rescale_intensity


class Histogram:
    def his(self, img_name, type):
        img = gdal.Open(img_name)
        b1 = img.GetRasterBand(1).ReadAsArray(0, 0, img.RasterXSize, img.RasterYSize)
        b2 = img.GetRasterBand(2).ReadAsArray(0, 0, img.RasterXSize, img.RasterYSize)
        b3 = img.GetRasterBand(3).ReadAsArray(0, 0, img.RasterXSize, img.RasterYSize)
        img = (np.dstack([b3, b2, b1])).astype(np.uint8)

        # Contrast Stretching Historgram
        if type == 'Contrast Stretching':
            # the min/max intensities of the input image are stretched to the limits allowed by the image’s dtype
            plt.imshow(rescale_intensity(img)), plt.title("Contrast stretching Historgram")
            plt.show()
        # Gamma Corrected Historgram
        elif type == 'Gamma Corrected':
            plt.imshow(exposure.adjust_gamma(img, 2)), plt.title("Gamma Corrected Historgram")
            plt.show()
        # Histogram Equalization
        elif type == 'Histogram Equalization':
            plt.imshow(equalize_hist(img, 2)), plt.title("Histogram Equalization")
            plt.show()
        # Adaptive Equalization Historgram
        elif type == 'Adaptive Equalization':
            plt.imshow(exposure.equalize_adapthist(img, clip_limit=0.03)), plt.title("Adaptive Equalization Histogram")
            plt.show()
        # Standard Deviation Historgram
        elif type == 'Standard Deviation':
            alpha = np.where(b1 + b2 + b3 == 0, 0, 1).astype(np.byte)
            red_stretched = self.std_stretch_data(b1, 2)
            green_stretched = self.std_stretch_data(b2, 2)
            blue_stretched = self.std_stretch_data(b3, 2)
            data_stretched = np.dstack((red_stretched, green_stretched, blue_stretched, alpha))
            plt.imshow(data_stretched), plt.title("Standard Deviation Historgram")
            plt.show()
        else:
            showinfo(
                title='ERROR',
                message=f'Please Select Valid Stretch Type!'
            )

    def std_stretch_data(self, band, n):
        # Get the mean and n standard deviations.
        mean, d = band.mean(), band.std() * n
        # the real max value.
        new_min = math.floor(max(mean - d, band.min()))
        new_max = math.ceil(min(mean + d, band.max()))
        data = np.clip(band, new_min, new_max)

        # Scale the data.
        data = (data - data.min()) / (new_max - new_min)
        return data

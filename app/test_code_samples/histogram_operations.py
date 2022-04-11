from osgeo import gdal_array
import math
from osgeo import gdal
from skimage import exposure
from skimage.exposure import equalize_hist, rescale_intensity
from matplotlib import pyplot
from rasterio.plot import show_hist
import numpy as np
import matplotlib.image as mpimg
import cv2
import matplotlib.pyplot as plt

red = gdal.Open(r'D:\project2022\LC08_L1TP_176039_20210521_20210529_02_T1\LC08_L1TP_176039_20210521_20210529_02_T1_B4.TIF')
green = gdal.Open(r'D:\project2022\LC08_L1TP_176039_20210521_20210529_02_T1\LC08_L1TP_176039_20210521_20210529_02_T1_B3.TIF')
blue = gdal.Open(r'D:\project2022\LC08_L1TP_176039_20210521_20210529_02_T1\LC08_L1TP_176039_20210521_20210529_02_T1_B2.TIF')
destination = r'D:\project2022\isa\rgb.tif'

red_band = red.ReadAsArray()
green_band = green.ReadAsArray()
blue_band = blue.ReadAsArray()

rgb = np.array([red_band, green_band, blue_band])
scaled = (rgb * (255 / 65535)).astype(np.uint8)
gdal_array.SaveArray(scaled, destination, 'GTiff', red)

img = mpimg.imread(r'D:\project2022\isa\rgb.tif').astype(np.uint8)
imgplot = plt.imshow(img)
plt.axis('off')
plt.show()

# Contrast stretching
p2, p98 = np.percentile(img, (2, 98))
img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
plt.imshow ( img_rescale ), plt.title ( "Contrast stretching Historgram" )
plt.show ()
show_hist(img_rescale,density=True,  bins=50, histtype='stepfilled', lw=0.0, stacked=False, alpha=0.3, title="Contrast Stretching Historgram")
pyplot.show()


# Gamma Corrected Historgram
aa=exposure.adjust_gamma ( img, 2 )
plt.imshow ( aa ), plt.title ( "Gamma Corrected Historgram" )
plt.show ()

show_hist(aa,density=True,  bins=50, histtype='stepfilled', lw=0.0, stacked=False, alpha=0.3,title="Gamma Corrected Historgram")
pyplot.show()

# Histogram Equalization
ww=equalize_hist ( img, 2 )
plt.imshow ( ww ), plt.title ( "Histogram Equalization" )
plt.show ()

show_hist(ww,density=True,  bins=50, histtype='stepfilled', lw=0.0, stacked=False, alpha=0.3,title="Histogram Equalization")
pyplot.show()

# Adaptive Equalization Historgram
ee=exposure.equalize_adapthist ( img, clip_limit=0.03 )
plt.imshow ( ee ), plt.title ( "Adaptive Equalization Histogram" )
plt.show ()

show_hist(ee,density=True,  bins=50, histtype='stepfilled', lw=0.0, stacked=False, alpha=0.3,title="Adaptive Equalization Historgram")
pyplot.show()

def std_stretch_data(self, band, n):
    # Get the mean and n standard deviations.
    mean, d = band.mean (), band.std () * n
    # the real max value.
    new_min = math.floor ( max ( mean - d, band.min () ) )
    new_max = math.ceil ( min ( mean + d, band.max () ) )
    data = np.clip ( band, new_min, new_max )

    # Scale the data.
    data = (data - data.min ()) / (new_max - new_min)
    return data

# Standard Deviation Historgram

alpha = np.where ( red + green + blue == 0, 0, 1 ).astype ( np.byte )
red_stretched = std_stretch_data ( red, 2 )
green_stretched = std_stretch_data ( green, 2 )
blue_stretched = std_stretch_data ( blue, 2 )
data_stretched = np.dstack ( (red_stretched, green_stretched, blue_stretched, alpha) )
plt.imshow ( data_stretched ), plt.title ( "Standard Deviation Historgram" )
plt.show ()

show_hist(data_stretched,density=True, bins=50, histtype='stepfilled', lw=0.0, stacked=False, alpha=0.3,title="Standard Deviation Historgram")
pyplot.show()
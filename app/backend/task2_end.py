import matplotlib
import numpy as np
from osgeo import gdal
from skimage import img_as_float
from skimage import exposure
READ_LIBTIFF = False
WRITE_LIBTIFF = False
IFD_LEGACY_API = True
np.seterr(over='ignore')
np.seterr(divide='ignore', invalid='ignore')
matplotlib.rcParams['font.size'] = 8
from skimage.exposure import equalize_hist, rescale_intensity, is_low_contrast

from matplotlib import pyplot as plt
np.seterr(over='ignore')
np.seterr(divide='ignore', invalid='ignore')
import math

img1 = gdal.Open(r'D:\project2021-2022\tiles500\tiles500_1.tif')
b1=img1.GetRasterBand(1).ReadAsArray(0,0,img1.RasterXSize, img1.RasterYSize)
b2=img1.GetRasterBand(2 ).ReadAsArray(0,0,img1.RasterXSize, img1.RasterYSize)
b3=img1.GetRasterBand ( 3 ).ReadAsArray(0,0,img1.RasterXSize, img1.RasterYSize)
img = (np.dstack ( [b3,b2 ,b1 ] )).astype(np.uint8)

def his(img,type):
    print(type)
    # Contrast stretching,
    if type == 'Contrast stretching':
        # the min/max intensities of the input image are stretched to the limits allowed by the image’s dtype
        plt.imshow ( rescale_intensity ( img ) ), plt.title ( "Contrast stretching Historgram" )
        plt.show ()
    elif type =='exposure.adjust_gamma':
        plt.imshow ( exposure.adjust_gamma ( img, 2 ) ), plt.title ( "Gamma Corrected Historgram" )
        plt.show ()
    elif type == 'Histogram equalization':
        plt.imshow ( equalize_hist ( img, 2 ) ), plt.title ( "Histogram equalization" )
        plt.show ()
# Adaptive Equalization
    elif type=='Adaptive Equalization':
        plt.imshow ( exposure.equalize_adapthist ( img, clip_limit=0.03 )), plt.title ( "Adaptive Equalization Histogram" )
        plt.show ()


def std_stretch_data(img1, n):
    # Get the mean and n standard deviations.
    mean, d = img1.mean (), img1.std () * n
    # the real max value.
    new_min = math.floor ( max ( mean - d, img1.min () ) )
    new_max = math.ceil ( min ( mean + d, img1.max () ) )
    data = np.clip ( img1, new_min, new_max )

    # Scale the data.
    data = (data - data.min ()) / (new_max - new_min)
    return data

alpha = np.where ( b1 + b2 + b3 == 0, 0, 1 ).astype ( np.byte )
red_stretched = std_stretch_data ( b1, 2 )
green_stretched = std_stretch_data ( b2, 2 )
blue_stretched = std_stretch_data ( b3, 2 )

data_stretched = np.dstack ( (red_stretched, green_stretched, blue_stretched, alpha) )
plt.imshow ( data_stretched ),plt.title ( "Standard Deviation Historgram" )
plt.show ()



##https://scikit-image.org/docs/0.14.x/api/skimage.exposure.html
##https://zia207.github.io/geospatial-python.io/lesson_06_working-with-raster-data.html
##https://rasterio.readthedocs.io/en/latest/topics/plotting.html
#https://rasterio.readthedocs.io/en/latest/topics/plotting.html

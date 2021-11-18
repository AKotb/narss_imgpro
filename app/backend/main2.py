from task2_end import std_stretch_data,his
from osgeo import gdal
import pandas as pd
from scipy import stats
from skimage.exposure import equalize_hist, rescale_intensity, is_low_contrast
import numpy as np

READ_LIBTIFF = False
WRITE_LIBTIFF = False
IFD_LEGACY_API = True
np.seterr(over='ignore')
np.seterr(divide='ignore', invalid='ignore')
global img1

img1 = gdal.Open(r'D:\project2021-2022\tiles500\tiles500_1.tif')
b1=img1.GetRasterBand(1).ReadAsArray(0,0,img1.RasterXSize, img1.RasterYSize)
b2=img1.GetRasterBand(2 ).ReadAsArray(0,0,img1.RasterXSize, img1.RasterYSize)
b3=img1.GetRasterBand ( 3 ).ReadAsArray(0,0,img1.RasterXSize, img1.RasterYSize)
img = (np.dstack ( [b3,b2 ,b1 ] )).astype(np.uint8)

# class hist:
#     def init(self):
#     self.b1=img1.GetRasterBand(1).ReadAsArray(0,0,img1.RasterXSize, img1.RasterYSize)
#     self.b2=img1.GetRasterBand(2 ).ReadAsArray(0,0,img1.RasterXSize, img1.RasterYSize)
#     self.b3=img1.GetRasterBand ( 3 ).ReadAsArray(0,0,img1.RasterXSize, img1.RasterYSize)
#     img = (np.dstack ( [b3,b2 ,b1 ] )).astype(np.uint8)
#     return b1,b2,b3

n=2
type = 'Contrast stretching'     ##(Contrast stretching ,Gamma Corrected Historgram,Histogram equalization,Adaptive Equalization)

if __name__ == "__main__":
    his( img, type )
    std_stretch_data ( img1, n )


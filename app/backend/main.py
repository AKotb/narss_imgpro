import extent_checker
from osgeo import ogr
from osgeo import gdal

raster = gdal.Open ( r'D:\Work\NARSS\Research Project\2020-2022\Extent_Checker_Data\tiles500_1.tif' )
vector = ogr.Open ( r'D:\Work\NARSS\Research Project\2020-2022\Extent_Checker_Data\poly2.shp' )

if __name__ == "__main__":
    ans = extent_checker.func(raster, vector)
from osgeo import ogr
from osgeo import osr


class ExtentChecker:

    def func(self, raster, vector):
        # Get raster geometry
        transform = raster.GetGeoTransform()
        pixelWidth = transform[1]
        pixelHeight = transform[5]
        cols = raster.RasterXSize
        rows = raster.RasterYSize

        xLeft = transform[0]
        yTop = transform[3]
        xRight = xLeft + cols * pixelWidth
        yBottom = yTop + rows * pixelHeight

        ring = ogr.Geometry(ogr.wkbLinearRing)
        ring.AddPoint(xLeft, yTop)
        ring.AddPoint(xLeft, yBottom)
        ring.AddPoint(xRight, yTop)
        ring.AddPoint(xRight, yBottom)
        ring.AddPoint(xLeft, yTop)
        rasterGeometry = ogr.Geometry(ogr.wkbPolygon)
        rasterGeometry.AddGeometry(ring)

        # Get vector geometry
        layer = vector.GetLayer()
        feature = layer.GetFeature(0)
        vectorGeometry = feature.GetGeometryRef()
        ext = self.get_extent(raster)
        src_srs = osr.SpatialReference()
        src_srs.ImportFromWkt(raster.GetProjection())
        tgt_srs = src_srs.CloneGeogCS()
        geo_ext = self.reproject_coords(ext, src_srs, tgt_srs)
        print(geo_ext)
        print(rasterGeometry.Intersect(vectorGeometry))
        print("xLeft =", xLeft)
        print("yTop =", yTop)
        print("xRight =", xRight)
        print("yBottom =", yBottom)

        layer = vector.GetLayer(0)
        for i in range(layer.GetFeatureCount()):
            feature = layer.GetFeature(i)
            geometry = feature.GetGeometryRef()
            extent = geometry.GetEnvelope()
            vxmin = extent[0]
            vxmax = extent[1]
            vymin = extent[2]
            vymax = extent[3]
        print("vxmin =", vxmin)
        print("vxmax =", vxmax)
        print("vymin =", vymin)
        print("vymax =", vymax)

        if vxmin >= xLeft and vxmax <= xRight and vymin >= yBottom and vymax <= yTop:
            print("inside")
        elif (vxmin - xLeft) < 0.05 and (vxmax - xRight) < 0.05 and (vymin - yBottom) < 0.05 and (vymax - yTop) < 0.05:
            print("identical")
        elif vxmin > xRight or vxmax < xLeft or vymin > yTop or vymax < yBottom:
            print("outside")
        else:
            print("partially inside")

    def get_extent(self, raster):
        """ Return list of corner coordinates from a gdal Dataset """
        xmin, xpixel, _, ymax, _, ypixel = raster.GetGeoTransform()
        width, height = raster.RasterXSize, raster.RasterYSize
        xmax = xmin + width * xpixel
        ymin = ymax + height * ypixel
        return (xmin, ymax), (xmax, ymax), (xmax, ymin), (xmin, ymin)

    def reproject_coords(self, coords, src_srs, tgt_srs):
        """ Reproject a list of x,y coordinates. """
        trans_coords = []
        transform = osr.CoordinateTransformation(src_srs, tgt_srs)
        for x, y in coords:
            x, y, z = transform.TransformPoint(x, y)
            trans_coords.append([x, y])
        return trans_coords

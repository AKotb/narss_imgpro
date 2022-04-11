"""
Classify a multi-band, satellite image.
Usage:
    classify.py <input_fname> <train_data_path> <output_fname> [--method=<classification_method>]
                                                               [--validation=<validation_data_path>]
                                                               [--verbose]
    classify.py -h | --help
The <input_fname> argument must be the path to a GeoTIFF image.
The <train_data_path> argument must be a path to a directory with vector data files
(in shapefile format). These vectors must specify the target class of the training pixels. One file
per class. The base filename (without extension) is taken as class name.
If a <validation_data_path> is given, then the validation vector files must correspond by name with
the training data. That is, if there is a training file train_data_path/A.shp then the corresponding
validation_data_path/A.shp is expected.
The <output_fname> argument must be a filename where the classification will be saved (GeoTIFF format).
No geographic transformation is performed on the data. The raster and vector data geographic
parameters must match.
Options:
  -h --help  Show this screen.
  --method=<classification_method>      Classification method to use: random-forest (for random
                                        forest) or svm (for support vector machines)
                                        [default: random-forest]
  --validation=<validation_data_path>   If given, it must be a path to a directory with vector data
                                        files (in shapefile format). These vectors must specify the
                                        target class of the validation pixels. A classification
                                        accuracy report is writen to stdout.
  --verbose                             If given, debug output is writen to stdout.
"""
import logging

import numpy as np
from osgeo import gdal

logger = logging.getLogger(__name__)

# A list of "random" colors
COLORS = [
    "#000000", "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
    "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87",
    "#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80",
    "#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100",
    "#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F",
    "#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09",
    "#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66",
    "#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C",
    "#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81",
    "#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00",
    "#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700",
    "#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329",
    "#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C",
    "#83AB58", "#001C1E", "#D1F7CE", "#004B28", "#C8D0F6", "#A3A489", "#806C66", "#222800",
    "#BF5650", "#E83000", "#66796D", "#DA007C", "#FF1A59", "#8ADBB4", "#1E0200", "#5B4E51",
    "#C895C5", "#320033", "#FF6832", "#66E1D3", "#CFCDAC", "#D0AC94", "#7ED379", "#012C58"
]


def write_geotiff(fname, data, geo_transform, projection, classes, data_type=gdal.GDT_Byte):
    """
    Create a GeoTIFF file with the given data.
    :param fname: Path to a directory with shapefiles
    :param data: Number of rows of the result
    :param geo_transform: Returned value of gdal.Dataset.GetGeoTransform (coefficients for
                          transforming between pixel/line (P,L) raster space, and projection
                          coordinates (Xp,Yp) space.
    :param projection: Projection definition string (Returned by gdal.Dataset.GetProjectionRef)
    """
    driver = gdal.GetDriverByName('GTiff')
    rows, cols = data.shape
    dataset = driver.Create(fname, cols, rows, 1, data_type)
    dataset.SetGeoTransform(geo_transform)
    dataset.SetProjection(projection)
    band = dataset.GetRasterBand(1)
    band.WriteArray(data)
    blankcalss = [''] + classes
    ct = gdal.ColorTable()
    for pixel_value in range(len(classes) + 1):
        color_hex = COLORS[pixel_value]
        r = int(color_hex[1:3], 16)
        g = int(color_hex[3:5], 16)
        b = int(color_hex[5:7], 16)
        ct.SetColorEntry(pixel_value, (r, g, b, 255))
    band.SetColorTable(ct)
    band.SetCategoryNames(blankcalss)

    metadata = {
        'TIFFTAG_COPYRIGHT': 'CC BY 4.0',
        'TIFFTAG_DOCUMENTNAME': 'classification',
        'TIFFTAG_IMAGEDESCRIPTION': 'Supervised classification.',
        'TIFFTAG_MAXSAMPLEVALUE': str(len(classes)),
        'TIFFTAG_MINSAMPLEVALUE': '0',
        'TIFFTAG_SOFTWARE': 'Python, GDAL, scikit-learn'
    }
    dataset.SetMetadata(metadata)

    dataset = None  # Close the file
    return


def report_and_exit(txt, *args, **kwargs):
    logger.error(txt, *args, **kwargs)
    exit(1)


def predict_patch(single_sub_flat, classifier, i):
    sub_result = classifier.predict(single_sub_flat)
    return sub_result


#     print(i)
#     queue1.put(sub_result)


def Parallel_manager(flat_pixels, classifier):
    sub_flat_pixels = np.array_split(flat_pixels, 1000)
    result = []
    i = 1
    # queue1 = multiprocessing.Queue()
    for single_sub_flat in sub_flat_pixels:
        sub_result = predict_patch(single_sub_flat, classifier, i)
        result = np.append(result, sub_result)
        print(i)

        #         p = multiprocessing.Process(target=predict_patch, args=(single_sub_flat, classifier,queue1,i))
        #         p.start()
        i = i + 1
    #     for single_sub_flat in sub_flat_pixels:
    #         result = np.append(result, queue1.get())
    return result


def writegeoresults(raster_data_path, classifier, classes, classifier_path_file):
    # logging.info("Begin")
    # raster_name          =  "sentinel_mask.img"
    # raster_data_path     =  "data/raster/" + raster_name

    # output_path          = "data/results"
    # method               = "mlp"
    # classifier_file      = "926_mlp_classifier.pickle"  
    # classifier_path_file =  output_path +"/"+ method + "/" + classifier_file

    raster_dataset = gdal.Open(raster_data_path, gdal.GA_ReadOnly)

    geo_transform = raster_dataset.GetGeoTransform()
    proj = raster_dataset.GetProjectionRef()
    bands_data = []

    for b in range(1, raster_dataset.RasterCount + 1):
        band = raster_dataset.GetRasterBand(b)
        bands_data.append(band.ReadAsArray())

    bands_data = np.dstack(bands_data) / np.max(bands_data)
    rows, cols, n_bands = bands_data.shape
    # A sample is a vector with all the bands data. Each pixel (independent of its position) is a
    # sample.
    n_samples = rows * cols

    logger.debug("Classifing...")
    flat_pixels = bands_data.reshape((n_samples, n_bands))

    # with open(classifier_path_file,'rb') as p:  # Python 3: open(..., 'wb')
    #        classifier,classes = pickle.load(p)

    result = Parallel_manager(flat_pixels, classifier)

    # Reshape the result: split the labeled pixels into rows to create an image
    classification = result.reshape((rows, cols))
    write_geotiff(classifier_path_file, classification, geo_transform, proj, classes)

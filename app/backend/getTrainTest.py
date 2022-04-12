import os

import numpy as np
from osgeo import gdal
from backend.writetif import report_and_exit


def create_mask_from_vector(vector_data_path, cols, rows, geo_transform, projection, target_value=1,
                            output_fname='', dataset_format='MEM'):
    data_source = gdal.OpenEx(vector_data_path, gdal.OF_VECTOR)
    if data_source is None:
        report_and_exit("File read failed: %s", vector_data_path)
    layer = data_source.GetLayer(0)
    driver = gdal.GetDriverByName(dataset_format)
    target_ds = driver.Create(output_fname, cols, rows, 1, gdal.GDT_UInt16)
    target_ds.SetGeoTransform(geo_transform)
    target_ds.SetProjection(projection)
    gdal.RasterizeLayer(target_ds, [1], layer, burn_values=[target_value])
    return target_ds


def vectors_to_raster(file_paths, rows, cols, geo_transform, projection):
    labeled_pixels = np.zeros((rows, cols))
    for i, path in enumerate(file_paths):
        label = i + 1
        ds = create_mask_from_vector(path, cols, rows, geo_transform, projection,
                                     target_value=label)
        band = ds.GetRasterBand(1)
        a = band.ReadAsArray()
        labeled_pixels += a
        ds = None
    return labeled_pixels


def gettraingtest(raster_data_path, train_data_path, validation_data_path):
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

    try:
        files = [f for f in sorted(os.listdir(train_data_path)) if f.endswith('.shp')]
        classes = [f.split('.')[0] for f in files]
        shapefiles = [os.path.join(train_data_path, f) for f in files if f.endswith('.shp')]
    except OSError.FileNotFoundError as e:
        report_and_exit(str(e))

    labeled_pixels = vectors_to_raster(shapefiles, rows, cols, geo_transform, proj)
    is_train = np.nonzero(labeled_pixels)

    training_labels = labeled_pixels[is_train]

    training_samples = bands_data[is_train]
    training_samples = training_samples.astype(float)  #####

    try:
        files = [f for f in sorted(os.listdir(validation_data_path)) if f.endswith('.shp')]
        classes = [f.split('.')[0] for f in files]
        shapefiles = [os.path.join(validation_data_path, f) for f in files if f.endswith('.shp')]
    except OSError.FileNotFoundError as e:
        report_and_exit(str(e))

    labeled_pixels = vectors_to_raster(shapefiles, rows, cols, geo_transform, proj)
    is_train = np.nonzero(labeled_pixels)

    validition_labels = labeled_pixels[is_train]

    validition_samples = bands_data[is_train]
    validition_samples = validition_samples.astype(float)  #####

    flat_pixels = bands_data.reshape((n_samples, n_bands))

    return training_samples, training_labels, classes, validition_samples, validition_labels, flat_pixels, rows, cols, n_bands, geo_transform, proj

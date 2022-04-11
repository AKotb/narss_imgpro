import ast
import configparser
import os
import random
from shutil import copyfile

from osgeo import ogr


def deffout(inLayer, path):
    # import pdb; pdb.set_trace()
    outShapefile = os.path.join(path, crops_names[crop] + ".shp")

    outDriver = ogr.GetDriverByName("ESRI Shapefile")

    # Remove output shapefile if it already exists
    if os.path.exists(outShapefile):
        outDriver.DeleteDataSource(outShapefile)

    # Create the output shapefile
    outDataSource = outDriver.CreateDataSource(outShapefile)
    outLayer = outDataSource.CreateLayer(fieldname, geom_type=ogr.wkbPoint)

    # Add input Layer Fields to the output Layer
    inLayerDefn = inLayer.GetLayerDefn()
    for i in range(0, inLayerDefn.GetFieldCount()):
        fieldDefn = inLayerDefn.GetFieldDefn(i)
        outLayer.CreateField(fieldDefn)

    return outDataSource, outLayer


def savecrop(inLayer, selectedFeature, crop, tragetIndexes, path):
    outDataSource, outLayer = deffout(inLayer, path)
    # Get the output Layer's Feature Definition
    outLayerDefn = outLayer.GetLayerDefn()
    # import pdb; pdb.set_trace()
    print(crop)
    print(inLayer.GetFeatureCount())
    # Add features to the ouput Layer
    for i in tragetIndexes:
        # Get the input Feature
        inFeature = selectedFeature[i]
        # Create output Feature
        outFeature = ogr.Feature(outLayerDefn)
        # Add field values from input Layer
        for i in range(0, outLayerDefn.GetFieldCount()):
            outFeature.SetField(outLayerDefn.GetFieldDefn(i).GetNameRef(), inFeature.GetField(i))
        # Set geometry as centroid
        geom = inFeature.GetGeometryRef()
        # inFeature = None

        outFeature.SetGeometry(geom)
        # Add new feature to output Layer

        outLayer.CreateFeature(outFeature)
        outFeature = None
    outDataSource = None

    copyfile(prjfilePath, os.path.join(path, crops_names[crop] + ".prj"))


# Get the input Layer
# trainPercent  = 50
# inputFolder   = "/content/drive/MyDrive/000/03project/data/input/oneshpfile"
# shapefilename = "chip8_2017_trainingPoints_2.shp"
# prjfilename   = "chip8_2017_trainingPoints_2.prj" 
# outFolder     = "/content/drive/MyDrive/000/03project/data/input/isolatedshpfiles"


# fieldname     = "code"
# crops_names   = {2: "building", 3:"trees",4:"street",5:"vehilce"} 

# trainName     = "train"
# testName      = "test"


config = configparser.ConfigParser()
config.read('/content/drive/MyDrive/000/03project/data/HighResData/calcification.config')

trainPercent = int(config['SplitShapefile']['trainPercent'])

inputFolder = config['SplitShapefile']['inputFolder']
shapefilename = config['SplitShapefile']['shapefilename']
prjfilename = config['SplitShapefile']['prjfilename']

outFolder = config['general']['inputShapefilesPath']

fieldname = config['SplitShapefile']['fieldname']
crops_indexes_names = config['SplitShapefile']['crops_indexes_names']
# import pdb; pdb.set_trace()
crops_names = ast.literal_eval(crops_indexes_names)

trainName = config['general']['trainName']
testName = config['general']['testName']

trainOutFolder = os.path.join(outFolder, trainName)
testOutFolder = os.path.join(outFolder, testName)

inShapefile = os.path.join(inputFolder, shapefilename)
prjfilePath = os.path.join(inputFolder, prjfilename)

inDriver = ogr.GetDriverByName("ESRI Shapefile")
inDataSource = inDriver.Open(inShapefile, 0)
inLayer = inDataSource.GetLayer()

crops = []
corpsNames = []
for feature in inLayer:
    crop = feature.GetField(fieldname)

    if not (crop in crops):
        crops.append(crop)

for crop in crops:
    inLayer.SetAttributeFilter(fieldname + "='" + str(crop) + "'")

    if inLayer.GetFeatureCount() >= 5:

        print(crop)
        print(inLayer.GetFeatureCount())
        # print crop , inLayer.GetFeatureCount()
        selectedFeature = []
        for i in range(0, inLayer.GetFeatureCount()):
            # Get the input Feature
            inFeature = inLayer.GetNextFeature()
            selectedFeature.append(inFeature)

        selectedFeatureIndexes = list(range(len(selectedFeature)))
        # import pdb; pdb.set_trace()
        random.shuffle(selectedFeatureIndexes)
        trainIndexes = selectedFeatureIndexes[:trainPercent]
        testIndexes = selectedFeatureIndexes[trainPercent:]

        if not os.path.exists(trainOutFolder):
            os.makedirs(trainOutFolder)

        if not os.path.exists(testOutFolder):
            os.makedirs(testOutFolder)

        savecrop(inLayer, selectedFeature, crop, trainIndexes, trainOutFolder)
        savecrop(inLayer, selectedFeature, crop, testIndexes, testOutFolder)

    # Save and close DataSources
inDataSource = None

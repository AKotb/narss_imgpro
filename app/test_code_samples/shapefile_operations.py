import tkinter
from tkintermapview import TkinterMapView
from PyQt5 import QtWidgets
import sys
from pathlib import Path
import geojson
import subprocess
import urllib.request as ur
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtWebEngineWidgets import *

root_tk = tkinter.Tk()

try:
    import PyQt5.QtWebEngineWidgets as QtWeb
except ImportError:
    import PyQt5.QtWebKitWidgets as QtWeb

sys.argv.append("--disable-web-security")

# Create application
app = QtWidgets.QApplication(sys.argv)

# Add window
win = QtWidgets.QWidget()
win.setWindowTitle('Editing shapefile')

# Add layout
layout = QtWidgets.QVBoxLayout()
win.setLayout(layout)

# Create QWebView
view = QtWeb.QWebEngineView()

# include code from map.html and map.js


view.setHtml('''
<!DOCTYPE html>
<html>
<head>
    <title>Leaflet polygon with area</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.5.1/dist/leaflet.css">
    <link rel="stylesheet" href="https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.css">
    <script src="https://unpkg.com/leaflet@1.5.1/dist/leaflet.js"></script>
    <script src="https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.js"></script>
    <style type="text/css">
        #map {
            height: 1200px;
            width: 2200px;
        }
        .area-tooltip {
            background: #363636;
            background: rgba(0,0,0,0.5);
            border: none;
            color: #f8d5e4;
        }


    </style>

</head>
<body margin:0px ;
      padding:0px>
<div id="map" margin:0></div>
<script>
        function createAreaTooltip(layer) {
            if(layer.areaTooltip) {
                return;
            }

            layer.areaTooltip = L.tooltip({
                permanent: true,
                direction: 'center',
                className: 'area-tooltip'
            });

            layer.on('remove', function(event) {
                layer.areaTooltip.remove();
            });

            layer.on('add', function(event) {
                updateAreaTooltip(layer);
                layer.areaTooltip.addTo(map);
            });

            if(map.hasLayer(layer)) {
                updateAreaTooltip(layer);
                layer.areaTooltip.addTo(map);
            }
        }



        function updateAreaTooltip(layer) {
            var area = L.GeometryUtil.geodesicArea(layer.getLatLngs()[0]);
            var readableArea = L.GeometryUtil.readableArea(area, true);
            var latlng = layer.getCenter();

            layer.areaTooltip
                .setContent(readableArea)
                .setLatLng(latlng);
        }

        /**
         * SIMPLE EXAMPLE
         */
         var map = L.map('map').setView([30, 31], 8);

        // Creating a Layer object

        L.tileLayer('http://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: 'Map data &copy; <a href="http://openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        }).addTo(map);

        var polygon = L.polygon([
            [51.509, -0.08],
            [51.503, -0.06],
            [51.51, -0.047]
        ]).addTo(map);



        createAreaTooltip(polygon);

        /**
         * EXAMPLE WITH LEAFLET DRAW CONTROL
         */
        var drawnItems = L.featureGroup().addTo(map);

        map.addControl(new L.Control.Draw({

            edit: {
                featureGroup: drawnItems,
                poly: {
                    allowIntersection: false
                }
            },
            draw: {
                marker: true,
                circle: true,
                circlemarker: true,
                rectangle: true,
                polyline: true,
                polygon: {
                    allowIntersection: false,
                    showArea: true
                }
            }
        }));

        map.on(L.Draw.Event.CREATED, function(event) {
            var layer = event.layer;

            if(layer instanceof L.Polygon) {
                createAreaTooltip(layer);
            }

            drawnItems.addLayer(layer);
        });

        map.on(L.Draw.Event.EDITED, function(event) {
            event.layers.getLayers().forEach(function(layer) {
                if(layer instanceof L.Polygon) {
                    updateAreaTooltip(layer);
                }
            })
        });

    map.on('draw:created', function (e) {
        var type = e.layerType,
            layer = e.layer;

        if (type === 'polygon') {
            let dataStr = JSON.stringify(layer.toGeoJSON());
         let dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);

         let exportFileDefaultName = ' D:/project2022/js/amany.json';


         let linkElement = document.createElement('a');
         linkElement.setAttribute('href', dataUri);
         linkElement.setAttribute('download', exportFileDefaultName);
         linkElement.click();
        }
        // here you add it to a layer to display it in the map
        drawnItems.addLayer(layer);
        function download(filename, text) {
              var element = document.createElement('a');
              element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(text));
              element.setAttribute('download', filename);

              element.style.display = 'none';
              document.body.appendChild(element);

              element.click();
              document.body.removeChild(element);
            }
        function exportToCsvFile(jsonData) {
                let csvStr = parseJSONToCSVStr(dataStr);
                let dataUri = 'data:text/csv;charset=utf-8,'+ csvStr;

                let exportFileDefaultName = 'data.csv';

                let linkElement = document.createElement('a');
                linkElement.setAttribute('href', dataUri);
                linkElement.setAttribute('download', exportFileDefaultName);
                linkElement.click();
            }


                });

</script>
</body>
</html>
''')

# Add QWebView to the layout
layout.addWidget(view)

# Show window, run app
win.show()
app.exec_()

### from osgeo import gdal
##srcDS = gdal.OpenEx('test.json')
##ds = gdal.VectorTranslate('test.shp', srcDS, format='ESRI Shapefile')

###https://stackoverflow.com/questions/35164123/using-basemap-as-a-figure-in-a-python-gui

#### You can convert graphics you draw on your map into shapefiles or geodatabase feature classes. The Convert Graphics To Features command, which is available from the Drawing menu on the Draw toolbar or by right-clicking a data frame in the table of contents, supports all the graphic types you can draw with the tools in the graphics palette on the Draw toolbar, including circles, curved lines, and freehand lines. You can also convert graphic text into annotation feature classes.   ####

from tkinter import Tk
from tkinter import filedialog as fd
from osgeo import ogr
from osgeo import gdal
from app.backend import extent_checker


class HistogramGeneratorInterface:
    def __init__(self, master=None):
        Tk.frame.__init__(self, master)
        self.master = master
        self.image_file_name = None
        self.shpfile_file_name = None
        self.init_window()

    def init_window(self):
        # window title in the title bar
        self.master.title("Extent Checker")

        # window title label
        window_lbl = Tk.Label(self.master, text="Extent Checker")
        window_lbl.grid(padx=10, pady=10, row=0, column=0, columnspan=3)

        # Open raster image
        input_image_lbl = Tk.Label(self.master, text="Image")
        input_image_lbl.grid(sticky='W', padx=10, pady=10, row=1, column=0)
        input_image_txt_field = Tk.Text(self.master, height=1, width=50)
        input_image_txt_field.grid(sticky='W', padx=10, pady=10, row=1, column=1, columnspan=2)
        input_image_btn = Tk.Button(self.master, text="Browse", command=self.select_image)
        input_image_btn.grid(sticky='W', padx=10, pady=10, row=1, column=3)

        # Open ShapeFile
        input_shpfile_lbl = Tk.Label(self.master, text="ShapeFile")
        input_shpfile_lbl.grid(sticky='W', padx=10, pady=10, row=2, column=0)
        input_shpfile_txt_field = Tk.Text(self.master, height=1, width=50)
        input_shpfile_txt_field.grid(sticky='W', padx=10, pady=10, row=2, column=1, columnspan=2)
        input_shpfile_btn = Tk.Button(self.master, text="Browse", command=self.select_shapefile)
        input_shpfile_btn.grid(sticky='W', padx=10, pady=10, row=2, column=3)

        # Control Buttons
        self.check_extent_btn = Tk.Button(self.master, text="Check", command=self.check_extent)
        self.check_extent_btn.grid(sticky='E', padx=10, pady=10, row=3, column=1)
        self.cancel_btn = Tk.Button(self.master, text="Cancel", command=self.exit)
        self.cancel_btn.grid(sticky='W', padx=10, pady=10, row=3, column=2)

    def exit(self):
        self.master.destroy()

    def select_image(self):
        filetypes = (
            ('TIF Files', '*.tif'),
            ('All Files', '*.*')
        )
        self.image_file_name = fd.askopenfilename(
            title='Open File',
            initialdir='/',
            filetypes=filetypes)
        self.input_image_txt_field.delete(1.0, Tk.END)
        self.input_image_txt_field.insert(Tk.END, self.image_file_name)

    def select_shapefile(self):
        filetypes = (
            ('TIF Files', '*.shp'),
            ('All Files', '*.*')
        )
        self.shpfile_file_name = fd.askopenfilename(
            title='Open File',
            initialdir='/',
            filetypes=filetypes)
        self.input_shpfile_txt_field.delete(1.0, Tk.END)
        self.input_shpfile_txt_field.insert(Tk.END, self.shpfile_file_name)

    def check_extent(self):
        raster_image = gdal.Open(self.image_file_name)
        shapefile = ogr.Open(self.shpfile_name)
        ans = extent_checker.func(raster_image, shapefile)
from tkinter import *
from tkinter import filedialog as fd
from osgeo import gdal
from osgeo import ogr
from tkinter import ttk


class MLClassificationInterface(Frame):

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.selected_ml_classification_method = None
        self.ml_classification_method = None
        self.cancel_btn = Button(self.master, text="Cancel", command=self.exit)
        self.classify_btn = Button(self.master, text="Classify", command=self.classify)
        self.master = master
        self.image_file_name = None
        self.train_shpfile_file_name = None
        self.input_image_txt_field = None
        self.input_train_shpfile_txt_field = None
        self.init_window()

    def init_window(self):
        # window title in the title bar
        self.master.title("ML Image Classification")

        # window title label
        window_lbl = Label(self.master, text="ML Image Classification")
        window_lbl.grid(padx=10, pady=10, row=0, column=0, columnspan=3)

        # Open raster image
        input_image_lbl = Label(self.master, text="Image")
        input_image_lbl.grid(sticky='W', padx=10, pady=10, row=1, column=0)
        self.input_image_txt_field = Text(self.master, height=1, width=50)
        self.input_image_txt_field.grid(sticky='W', padx=10, pady=10, row=1, column=1, columnspan=2)
        input_image_btn = Button(self.master, text="Browse", command=self.select_image)
        input_image_btn.grid(sticky='W', padx=10, pady=10, row=1, column=3)

        # Open ShapeFile
        input_train_shpfile_lbl = Label(self.master, text="Training Set ShapeFile")
        input_train_shpfile_lbl.grid(sticky='W', padx=10, pady=10, row=2, column=0)
        self.input_train_shpfile_txt_field = Text(self.master, height=1, width=50)
        self.input_train_shpfile_txt_field.grid(sticky='W', padx=10, pady=10, row=2, column=1, columnspan=2)
        input_train_shpfile_btn = Button(self.master, text="Browse", command=self.select_train_shapefile)
        input_train_shpfile_btn.grid(sticky='W', padx=10, pady=10, row=2, column=3)

        # choose ml classification method
        ml_classification_method_lbl = Label(self.master, text="Classification Method")
        ml_classification_method_lbl.grid(sticky='W', padx=10, pady=10, row=3, column=0)
        ml_classification_method_value = StringVar()
        self.ml_classification_method = ttk.Combobox(self.master, textvariable=ml_classification_method_value)
        self.ml_classification_method.bind("<<ComboboxSelected>>", self.ml_classification_method_value)
        self.ml_classification_method['values'] = ['KNN', 'SVM', 'RF', 'MLP']
        self.ml_classification_method.current(0)
        self.ml_classification_method.grid(sticky='W', padx=10, pady=10, row=3, column=1, columnspan=2)

        # Control Buttons
        self.classify_btn.grid(sticky='E', padx=10, pady=10, row=4, column=1)
        self.cancel_btn.grid(sticky='W', padx=10, pady=10, row=4, column=2)

    def exit(self):
        self.master.destroy()

    def select_image(self):
        filetypes = (
            ('TIF Files', '*.tif'),
            ('All Files', '*.*')
        )
        self.image_file_name = fd.askopenfilename(
            title='Open File',
            initialdir='D:/NARSS/Research_Project/2020-2022/Data/Extent_Checker_Data',
            filetypes=filetypes)
        self.input_image_txt_field.delete(1.0, END)
        self.input_image_txt_field.insert(END, self.image_file_name)

    def select_train_shapefile(self):
        filetypes = (
            ('TIF Files', '*.shp'),
            ('All Files', '*.*')
        )
        self.train_shpfile_file_name = fd.askopenfilename(
            title='Open File',
            initialdir='D:/NARSS/Research_Project/2020-2022/Data/Extent_Checker_Data',
            filetypes=filetypes)
        self.input_train_shpfile_txt_field.delete(1.0, END)
        self.input_train_shpfile_txt_field.insert(END, self.train_shpfile_file_name)

    def ml_classification_method_value(self, event):
        self.selected_ml_classification_method = self.ml_classification_method.get()

    def classify(self):
        raster_image = gdal.Open(self.image_file_name)
        print(self.image_file_name)
        train_shapefile = ogr.Open(self.train_shpfile_file_name)
        print(self.train_shpfile_file_name)
        print(self.selected_ml_classification_method)
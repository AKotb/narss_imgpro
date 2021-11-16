from tkinter import Tk, Menu, messagebox, Frame, Label, StringVar, IntVar, Radiobutton, Button
from tkinter import ttk
from tkinter import filedialog as fd
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import geopandas
import geoplot
from app.imgpro import extent_checker_interface
from app.imgpro import histogram_generator_interface


class GUIHandler:

    def __init__(self, master=None):
        Tk.frame.__init__(self, master)
        self.master = master
        self.selected_red_bnd = 1
        self.selected_green_bnd = 1
        self.selected_blue_bnd = 1
        self.img_src = None
        self.file_name = 'Input Image'
        self.init_window()

    def init_window(self):
        # window title in the title bar
        self.master.title("NARSS Image Processing Application")
        menubar = Menu(self.master)
        self.master.config(menu=menubar)

        # File Menu
        file_menu = Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Image", command=self.open_image)
        file_menu.add_separator()
        file_menu.add_command(label="Open Shapefile", command=self.open_shapefile)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.exit)
        menubar.add_cascade(label="File", menu=file_menu)

        # Process Menu
        process_menu = Menu(menubar, tearoff=0)
        process_menu.add_command(label="Process 1", command=self.process_1)
        process_menu.add_separator()
        process_menu.add_command(label="Process 2", command=self.process_1)
        process_menu.add_separator()
        process_menu.add_command(label="Process 3", command=self.process_1)
        menubar.add_cascade(label="Process", menu=process_menu)

        # Process Menu
        utility_menu = Menu(menubar, tearoff=0)
        utility_menu.add_command(label="Extent Checker", command=self.extent_checker)
        utility_menu.add_separator()
        utility_menu.add_command(label="Histogram Generator", command=self.histogram_generator)
        menubar.add_cascade(label="Utilities", menu=utility_menu)

        # Help Menu
        help_menu = Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self.about)
        menubar.add_cascade(label="Help", menu=help_menu)

    def exit(self):
        exit()

    def open_shapefile(self):
        filetypes = (
            ('TIF Files', '*.shp'),
            ('All Files', '*.*')
        )
        file_name = fd.askopenfilename(
            title='Open File',
            #initialdir='/',
            initialdir='D:/NARSS/Research_Project/2020-2022/Data',
            filetypes=filetypes)
        shapefile = geopandas.read_file(file_name)
        fig, ax = plt.subplots(1, figsize=(10, 20))
        shapefile.plot(column='Gov_Eng', cmap=None, ax=ax, legend=True, legend_kwds={'frameon': True, 'loc': 'lower left', 'title': 'Districts', 'fontsize': 8})
        plt.get_current_fig_manager().set_window_title(file_name)
        plt.show()

    def open_image(self):
        filetypes = (
            ('TIF Files', '*.tif'),
            ('All Files', '*.*')
        )
        self.file_name = fd.askopenfilename(
            title='Open File',
            #initialdir='/',
            initialdir='D:/NARSS/Research_Project/2020-2022/Data',
            filetypes=filetypes)
        self.img_src = rasterio.open(self.file_name)
        self.open_multi_band_image(self.img_src)

    def open_multi_band_image(self, img_src):
        self.root = Tk()
        self.root.title("RGB Band Selection")
        self.root.geometry("500x300")
        frame = Frame(self.root, width=500, height=300)
        label_header = Label(frame, text="RGB Band Selection", font=("SansSerif", 16))
        label_red = Label(frame, text="Red", font=("SansSerif", 14))
        label_green = Label(frame, text="Green", font=("SansSerif", 14))
        label_blue = Label(frame, text="Blue", font=("SansSerif", 14))
        frame.pack()
        label_header.pack()
        label_header.place(x=150, y=20)
        label_red.pack()
        label_red.place(x=0, y=100)
        label_green.pack()
        label_green.place(x=0, y=150)
        label_blue.pack()
        label_blue.place(x=0, y=200)

        bnd_lst = []
        for band in range(img_src.count):
            band += 1
            bnd_lst.append(band)

        band1_value = StringVar()
        self.band_1 = ttk.Combobox(frame, textvariable=band1_value)
        self.band_1.bind("<<ComboboxSelected>>", self.band1_value)
        self.band_1['values'] = bnd_lst
        self.band_1.current(0)
        self.band_1.place(x=130, y=100)

        band2_value = StringVar()
        self.band_2 = ttk.Combobox(frame, textvariable=band2_value)
        self.band_2.bind("<<ComboboxSelected>>", self.band2_value)
        self.band_2['values'] = bnd_lst
        self.band_2.current(0)
        self.band_2.place(x=130, y=150)

        band3_value = StringVar()
        self.band_3 = ttk.Combobox(frame, textvariable=band3_value)
        self.band_3.bind("<<ComboboxSelected>>", self.band3_value)
        self.band_3['values'] = bnd_lst
        self.band_3.current(0)
        self.band_3.place(x=130, y=200)

        view_btn = Button(frame, text="View", command=self.submit)
        view_btn.pack()
        view_btn.place(x=200, y=250)
        cancel_btn = Button(frame, text="Cancel", command=self.cancel)
        cancel_btn.pack()
        cancel_btn.place(x=250, y=250)
        self.root.mainloop()

    def band1_value(self, event):
        self.selected_red_bnd = self.band_1.get()

    def band2_value(self, event):
        self.selected_green_bnd = self.band_2.get()

    def band3_value(self, event):
        self.selected_blue_bnd = self.band_3.get()

    def cancel(self):
        self.root.destroy()

    def submit(self):
        # Read the grid values into numpy arrays
        red = self.img_src.read(int(self.selected_red_bnd))
        green = self.img_src.read(int(self.selected_green_bnd))
        blue = self.img_src.read(int(self.selected_blue_bnd))

        # Normalize the bands
        redn = self.normalize(red)
        greenn = self.normalize(green)
        bluen = self.normalize(blue)

        # Stack bands
        rgb = np.dstack((redn, greenn, bluen))

        # View the color composite
        plt.get_current_fig_manager().set_window_title(self.file_name)
        plt.imshow(rgb)
        plt.axis('off')
        plt.show()

    # Normalize bands into 0.0 - 1.0 scale
    def normalize(self, array):
        array_min, array_max = array.min(), array.max()
        return (array - array_min) / (array_max - array_min)

    def about(self):
        messagebox.showinfo("NARSS_IMGPro",
                            "NARSS Image Processing Application [version 0.1]")

    def process_1(self):
        root = Tk()
        root.geometry("650x250")
        root.mainloop()

    def process_2(self):
        root = Tk()
        root.geometry("650x250")
        root.mainloop()

    def process_3(self):
        root = Tk()
        root.geometry("650x250")
        root.mainloop()

    def extent_checker(self):
        root = Tk()
        root.geometry("650x250")
        extent_checker_interface.ExtentCheckerInterface(root)
        root.mainloop()

    def histogram_generator(self):
        root = Tk()
        root.geometry("650x250")
        histogram_generator_interface.HistogramGeneratorInterface(root)
        root.mainloop()

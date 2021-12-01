from tkinter import *
from tkinter import filedialog as fd
from tkinter import ttk

from app.backend import histogram


class HistogramGeneratorInterface(Frame):

    def __init__(self, master=None):
        Tk.frame.__init__(self, master)
        self.master = master
        self.image_file_name = None
        self.stretch_type_value = None
        self.input_image_txt_field = None
        self.init_window()

    def init_window(self):
        # window title in the title bar
        self.master.title("Histogram Generator")

        # window title label
        window_lbl = Label(self.master, text="Histogram Generator")
        window_lbl.grid(padx=10, pady=10, row=0, column=0, columnspan=3)

        # Open raster image
        input_image_lbl = Label(self.master, text="Image")
        input_image_lbl.grid(sticky='W', padx=10, pady=10, row=1, column=0)
        self.input_image_txt_field = Text(self.master, height=1, width=50)
        self.input_image_txt_field.grid(sticky='W', padx=10, pady=10, row=1, column=1, columnspan=2)
        input_image_btn = Button(self.master, text="Browse", command=self.select_image)
        input_image_btn.grid(sticky='W', padx=10, pady=10, row=1, column=3)

        # Select Stretch Type
        input_stretchtype_lbl = Label(self.master, text="Stretch Type")
        input_stretchtype_lbl.grid(sticky='W', padx=10, pady=10, row=2, column=0)
        stretch_type_value = StringVar()
        self.stretch_type = ttk.Combobox(self.master, width=30, textvariable=stretch_type_value)
        self.stretch_type['values'] = ['Select', 'Contrast Stretching', 'Gamma Corrected', 'Standard Deviation',
                                       'Adaptive Equalization', 'Histogram Equalization']
        self.stretch_type.current(0)
        self.stretch_type.bind("<<ComboboxSelected>>", self.stretch_type_changed)
        self.stretch_type.grid(sticky='W', padx=10, pady=10, row=2, column=1, columnspan=2)

        # Control Buttons
        self.display_histogram_btn = Button(self.master, text="Show", command=self.display_histogram)
        self.display_histogram_btn.grid(sticky='E', padx=10, pady=10, row=4, column=1)
        self.cancel_btn = Button(self.master, text="Cancel", command=self.exit)
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
            # initialdir='/',
            initialdir='D:/Work/NARSS/Research Project/2020-2022/Histogram_Data',
            filetypes=filetypes)
        self.input_image_txt_field.delete(1.0, END)
        self.input_image_txt_field.insert(END, self.image_file_name)

    def display_histogram(self):
        print(self.image_file_name)
        print(self.stretch_type_value)
        histogram_obj = histogram.Histogram()
        histogram_obj.his(self.image_file_name, self.stretch_type_value)

    def stretch_type_changed(self, event):
        """ handle the stretch changed event """
        self.stretch_type_value = self.stretch_type.get()

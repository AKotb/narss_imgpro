import threading
import time
from tkinter import *
from tkinter import filedialog as fd
from tkinter import ttk
from tkinter.ttk import *

from backend import ml_classifiers


class MLClassificationInterface(Frame):

    def __init__(self, master=None, **kw):
        Frame.__init__(self, master)
        super().__init__(master, **kw)

        self.master = master
        self.image_file_name = None
        self.input_image_txt_field = None
        self.train_shpfiles_location = None
        self.test_shpfiles_location = None
        self.output_data_location = None
        self.ml_classification_method = None
        self.selected_ml_classification_method = "RF"
        self.svm_c_value = 1000
        self.svm_gamma_value = 0.1
        self.rf_njobs_value = 5
        self.rf_nestimators_value = 45
        self.mlp_fst_hdn_lyr_value = 100
        self.mlp_snd_hdn_lyr_value = 100
        self.mlp_trd_hdn_lyr_value = 100
        self.mlp_frt_hdn_lyr_value = 100
        self.knn_knnparam_value = 3

        self.input_train_shpfile_txt_field = Text(self.master, height=1, width=50)
        self.input_test_shpfile_txt_field = Text(self.master, height=1, width=50)
        self.output_data_location_txt_field = Text(self.master, height=1, width=50)

        ## SVM
        self.input_svm_c_lbl = Label(self.master, text="C")
        self.input_svm_c_txt_field = Text(self.master, height=1, width=10)
        self.input_svm_gamma_lbl = Label(self.master, text="Gamma")
        self.input_svm_gamma_txt_field = Text(self.master, height=1, width=10)

        ## RF
        self.input_rf_njobs_lbl = Label(self.master, text="N Jobs")
        self.input_rf_njobs_txt_field = Text(self.master, height=1, width=10)
        self.input_rf_nestimators_lbl = Label(self.master, text="N Estimators")
        self.input_rf_nestimators_txt_field = Text(self.master, height=1, width=10)

        ## MLP
        self.input_mlp_fst_hdn_lyr_lbl = Label(self.master, text="First Hidden Layer")
        self.input_mlp_fst_hdn_lyr_txt_field = Text(self.master, height=1, width=10)
        self.input_mlp_snd_hdn_lyr_lbl = Label(self.master, text="Second Hidden Layer")
        self.input_mlp_snd_hdn_lyr_txt_field = Text(self.master, height=1, width=10)
        self.input_mlp_trd_hdn_lyr_lbl = Label(self.master, text="Third Hidden Layer")
        self.input_mlp_trd_hdn_lyr_txt_field = Text(self.master, height=1, width=10)
        self.input_mlp_frt_hdn_lyr_lbl = Label(self.master, text="Fourth Hidden Layer")
        self.input_mlp_frt_hdn_lyr_txt_field = Text(self.master, height=1, width=10)

        ## KNN
        self.input_knn_knnparam_lbl = Label(self.master, text="KNN Parameter")
        self.input_knn_knnparam_txt_field = Text(self.master, height=1, width=10)

        self.process_execution_pb = Progressbar(self.master, orient=HORIZONTAL, length=400, mode='determinate')
        self.classify_btn = Button(self.master, text="Classify", command=self.classify)
        self.cancel_btn = Button(self.master, text="Cancel", command=self.exit)

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
        self.input_image_txt_field.grid(
            sticky='W', padx=10, pady=10, row=1, column=1, columnspan=2)
        input_image_btn = Button(
            self.master, text="Browse", command=self.select_image)
        input_image_btn.grid(sticky='W', padx=10, pady=10, row=1, column=3)

        # select taining shapefiles location
        input_train_shpfile_lbl = Label(
            self.master, text="Training Set ShapeFiles")
        input_train_shpfile_lbl.grid(
            sticky='W', padx=10, pady=10, row=2, column=0)
        self.input_train_shpfile_txt_field.grid(
            sticky='W', padx=10, pady=10, row=2, column=1, columnspan=2)
        input_train_shpfile_btn = Button(
            self.master, text="Browse", command=self.select_train_shapefiles)
        input_train_shpfile_btn.grid(
            sticky='W', padx=10, pady=10, row=2, column=3)

        # select test shapefiles location
        input_test_shpfile_lbl = Label(self.master, text="Test Set ShapeFiles")
        input_test_shpfile_lbl.grid(
            sticky='W', padx=10, pady=10, row=3, column=0)
        self.input_test_shpfile_txt_field.grid(
            sticky='W', padx=10, pady=10, row=3, column=1, columnspan=2)
        input_test_shpfile_btn = Button(
            self.master, text="Browse", command=self.select_test_shapefiles)
        input_test_shpfile_btn.grid(
            sticky='W', padx=10, pady=10, row=3, column=3)

        # Select output data directory
        output_data_location_lbl = Label(
            self.master, text="Output Data Directory")
        output_data_location_lbl.grid(
            sticky='W', padx=10, pady=10, row=4, column=0)
        self.output_data_location_txt_field.grid(
            sticky='W', padx=10, pady=10, row=4, column=1, columnspan=2)
        output_data_location_btn = Button(
            self.master, text="Browse", command=self.select_output_data_directory)
        output_data_location_btn.grid(
            sticky='W', padx=10, pady=10, row=4, column=3)

        # choose ml classification method
        ml_classification_method_lbl = Label(
            self.master, text="Classification Method")
        ml_classification_method_lbl.grid(
            sticky='W', padx=10, pady=10, row=5, column=0)
        ml_classification_method_value = StringVar()
        self.ml_classification_method = ttk.Combobox(
            self.master, textvariable=ml_classification_method_value)
        self.ml_classification_method.bind(
            "<<ComboboxSelected>>", self.ml_classification_method_value)
        self.ml_classification_method['values'] = ['Select Method', 'RF', 'KNN', 'SVM', 'MLP']
        self.ml_classification_method.current(0)
        self.ml_classification_method.grid(
            sticky='W', padx=10, pady=10, row=5, column=1, columnspan=2)

        # Progress Bar
        self.process_execution_pb.grid(
            sticky='W', padx=10, pady=10, row=8, column=1, columnspan=2)

        # Control Buttons
        self.classify_btn.grid(sticky='E', padx=10, pady=10, row=9, column=1)
        self.cancel_btn.grid(sticky='W', padx=10, pady=10, row=9, column=2)

    def exit(self):
        self.master.destroy()

    def select_image(self):
        filetypes = (
            ('TIF Files', '*.tif'),
            ('IMG Files', '*.img'),
            ('All Files', '*.*')
        )
        self.image_file_name = fd.askopenfilename(
            title='Open File',
            initialdir='D:/NARSS/Research_Project/2020-2022/Data',
            filetypes=filetypes)
        self.input_image_txt_field.delete(1.0, END)
        self.input_image_txt_field.insert(END, self.image_file_name)

    def select_train_shapefiles(self):
        self.train_shpfiles_location = fd.askdirectory(
            title='Training Shapefiles Location',
            initialdir='D:/NARSS/Research_Project/2020-2022/Data')
        self.input_train_shpfile_txt_field.delete(1.0, END)
        self.input_train_shpfile_txt_field.insert(
            END, self.train_shpfiles_location)

    def select_test_shapefiles(self):
        self.test_shpfiles_location = fd.askdirectory(
            title='Test Shapefiles Location',
            initialdir='D:/NARSS/Research_Project/2020-2022/Data')
        self.input_test_shpfile_txt_field.delete(1.0, END)
        self.input_test_shpfile_txt_field.insert(
            END, self.test_shpfiles_location)

    def select_output_data_directory(self):
        self.output_data_location = fd.askdirectory(
            title='Output Data Location',
            initialdir='D:/NARSS/Research_Project/2020-2022/Data')
        self.output_data_location_txt_field.delete(1.0, END)
        self.output_data_location_txt_field.insert(
            END, self.output_data_location)

    def ml_classification_method_value(self, event):
        self.selected_ml_classification_method = self.ml_classification_method.get()
        if self.selected_ml_classification_method == "SVM":
            self.input_rf_njobs_lbl.grid_remove()
            self.input_rf_njobs_txt_field.grid_remove()
            self.input_rf_nestimators_lbl.grid_remove()
            self.input_rf_nestimators_txt_field.grid_remove()

            self.input_mlp_fst_hdn_lyr_lbl.grid_remove()
            self.input_mlp_fst_hdn_lyr_txt_field.grid_remove()
            self.input_mlp_snd_hdn_lyr_lbl.grid_remove()
            self.input_mlp_snd_hdn_lyr_txt_field.grid_remove()
            self.input_mlp_trd_hdn_lyr_lbl.grid_remove()
            self.input_mlp_trd_hdn_lyr_txt_field.grid_remove()
            self.input_mlp_frt_hdn_lyr_lbl.grid_remove()
            self.input_mlp_frt_hdn_lyr_txt_field.grid_remove()

            self.input_knn_knnparam_lbl.grid_remove()
            self.input_knn_knnparam_txt_field.grid_remove()

            self.input_svm_c_lbl.grid(sticky='W', padx=10, pady=10, row=6, column=0)
            self.input_svm_c_txt_field.grid(sticky='W', padx=10, pady=10, row=6, column=1)
            self.input_svm_c_txt_field.delete(1.0, END)
            self.input_svm_c_txt_field.insert(END, '1000')
            self.input_svm_gamma_lbl.grid(sticky='W', padx=10, pady=10, row=7, column=0)
            self.input_svm_gamma_txt_field.grid(sticky='W', padx=10, pady=10, row=7, column=1)
            self.input_svm_gamma_txt_field.delete(1.0, END)
            self.input_svm_gamma_txt_field.insert(END, '0.1')

        elif self.selected_ml_classification_method == "RF":
            self.input_svm_c_lbl.grid_remove()
            self.input_svm_c_txt_field.grid_remove()
            self.input_svm_gamma_lbl.grid_remove()
            self.input_svm_gamma_txt_field.grid_remove()

            self.input_mlp_fst_hdn_lyr_lbl.grid_remove()
            self.input_mlp_fst_hdn_lyr_txt_field.grid_remove()
            self.input_mlp_snd_hdn_lyr_lbl.grid_remove()
            self.input_mlp_snd_hdn_lyr_txt_field.grid_remove()
            self.input_mlp_trd_hdn_lyr_lbl.grid_remove()
            self.input_mlp_trd_hdn_lyr_txt_field.grid_remove()
            self.input_mlp_frt_hdn_lyr_lbl.grid_remove()
            self.input_mlp_frt_hdn_lyr_txt_field.grid_remove()

            self.input_knn_knnparam_lbl.grid_remove()
            self.input_knn_knnparam_txt_field.grid_remove()

            self.input_rf_njobs_lbl.grid(sticky='W', padx=10, pady=10, row=6, column=0)
            self.input_rf_njobs_txt_field.grid(sticky='W', padx=10, pady=10, row=6, column=1)
            self.input_rf_njobs_txt_field.delete(1.0, END)
            self.input_rf_njobs_txt_field.insert(END, '5')
            self.input_rf_nestimators_lbl.grid(sticky='W', padx=10, pady=10, row=7, column=0)
            self.input_rf_nestimators_txt_field.grid(sticky='W', padx=10, pady=10, row=7, column=1)
            self.input_rf_nestimators_txt_field.delete(1.0, END)
            self.input_rf_nestimators_txt_field.insert(END, '45')

        elif self.selected_ml_classification_method == "MLP":
            self.input_svm_c_lbl.grid_remove()
            self.input_svm_c_txt_field.grid_remove()
            self.input_svm_gamma_lbl.grid_remove()
            self.input_svm_gamma_txt_field.grid_remove()

            self.input_rf_njobs_lbl.grid_remove()
            self.input_rf_njobs_txt_field.grid_remove()
            self.input_rf_nestimators_lbl.grid_remove()
            self.input_rf_nestimators_txt_field.grid_remove()

            self.input_knn_knnparam_lbl.grid_remove()
            self.input_knn_knnparam_txt_field.grid_remove()

            self.input_mlp_fst_hdn_lyr_lbl.grid(sticky='W', padx=10, pady=10, row=6, column=0)
            self.input_mlp_fst_hdn_lyr_txt_field.grid(sticky='W', padx=10, pady=10, row=6, column=1)
            self.input_mlp_fst_hdn_lyr_txt_field.delete(1.0, END)
            self.input_mlp_fst_hdn_lyr_txt_field.insert(END, '100')
            self.input_mlp_snd_hdn_lyr_lbl.grid(sticky='W', padx=10, pady=10, row=7, column=0)
            self.input_mlp_snd_hdn_lyr_txt_field.grid(sticky='W', padx=10, pady=10, row=7, column=1)
            self.input_mlp_snd_hdn_lyr_txt_field.delete(1.0, END)
            self.input_mlp_snd_hdn_lyr_txt_field.insert(END, '100')
            self.input_mlp_trd_hdn_lyr_lbl.grid(sticky='W', padx=10, pady=10, row=6, column=2)
            self.input_mlp_trd_hdn_lyr_txt_field.grid(sticky='W', padx=10, pady=10, row=6, column=3)
            self.input_mlp_trd_hdn_lyr_txt_field.delete(1.0, END)
            self.input_mlp_trd_hdn_lyr_txt_field.insert(END, '100')
            self.input_mlp_frt_hdn_lyr_lbl.grid(sticky='W', padx=10, pady=10, row=7, column=2)
            self.input_mlp_frt_hdn_lyr_txt_field.grid(sticky='W', padx=10, pady=10, row=7, column=3)
            self.input_mlp_frt_hdn_lyr_txt_field.delete(1.0, END)
            self.input_mlp_frt_hdn_lyr_txt_field.insert(END, '100')
        
        elif self.selected_ml_classification_method == "KNN":
            self.input_svm_c_lbl.grid_remove()
            self.input_svm_c_txt_field.grid_remove()
            self.input_svm_gamma_lbl.grid_remove()
            self.input_svm_gamma_txt_field.grid_remove()

            self.input_rf_njobs_lbl.grid_remove()
            self.input_rf_njobs_txt_field.grid_remove()
            self.input_rf_nestimators_lbl.grid_remove()
            self.input_rf_nestimators_txt_field.grid_remove()

            self.input_mlp_fst_hdn_lyr_lbl.grid_remove()
            self.input_mlp_fst_hdn_lyr_txt_field.grid_remove()
            self.input_mlp_snd_hdn_lyr_lbl.grid_remove()
            self.input_mlp_snd_hdn_lyr_txt_field.grid_remove()
            self.input_mlp_trd_hdn_lyr_lbl.grid_remove()
            self.input_mlp_trd_hdn_lyr_txt_field.grid_remove()
            self.input_mlp_frt_hdn_lyr_lbl.grid_remove()
            self.input_mlp_frt_hdn_lyr_txt_field.grid_remove()

            self.input_knn_knnparam_lbl.grid(sticky='W', padx=10, pady=10, row=6, column=0)
            self.input_knn_knnparam_txt_field.grid(sticky='W', padx=10, pady=10, row=6, column=1)
            self.input_knn_knnparam_txt_field.delete(1.0, END)
            self.input_knn_knnparam_txt_field.insert(END, '3')

        else:
            self.input_svm_c_lbl.grid_remove()
            self.input_svm_c_txt_field.grid_remove()
            self.input_svm_gamma_lbl.grid_remove()
            self.input_svm_gamma_txt_field.grid_remove()

            self.input_rf_njobs_lbl.grid_remove()
            self.input_rf_njobs_txt_field.grid_remove()
            self.input_rf_nestimators_lbl.grid_remove()
            self.input_rf_nestimators_txt_field.grid_remove()

            self.input_mlp_fst_hdn_lyr_lbl.grid_remove()
            self.input_mlp_fst_hdn_lyr_txt_field.grid_remove()
            self.input_mlp_snd_hdn_lyr_lbl.grid_remove()
            self.input_mlp_snd_hdn_lyr_txt_field.grid_remove()
            self.input_mlp_trd_hdn_lyr_lbl.grid_remove()
            self.input_mlp_trd_hdn_lyr_txt_field.grid_remove()
            self.input_mlp_frt_hdn_lyr_lbl.grid_remove()
            self.input_mlp_frt_hdn_lyr_txt_field.grid_remove()

            self.input_knn_knnparam_lbl.grid_remove()
            self.input_knn_knnparam_txt_field.grid_remove()

    def show_progress_bar(self, num):
        for i in range(num):
            self.master.update_idletasks()
            self.process_execution_pb['value'] += 1
            if self.selected_ml_classification_method == "SVM":
                time.sleep(7.5)
            elif self.selected_ml_classification_method == "RF":
                time.sleep(1)
            elif self.selected_ml_classification_method == "KNN":
                time.sleep(2)
            elif self.selected_ml_classification_method == "MLP":
                time.sleep(1.6)
            elif self.selected_ml_classification_method == "KNN":
                time.sleep(7.5)
            else:
                time.sleep(2)

    def execute_classification(self):
        self.classify_btn["state"] = DISABLED
        self.cancel_btn["state"] = DISABLED
        ml_classifiers_obj = ml_classifiers.MLClassifier()

        if self.selected_ml_classification_method == "SVM":
            print("SVM Method Selected!")
            self.svm_c_value = self.input_svm_c_txt_field.get(1.0, END)
            self.svm_gamma_value = self.input_svm_gamma_txt_field.get(1.0, END)
            ml_classifiers_obj.set_svm_parameters(self.selected_ml_classification_method, self.image_file_name,
                                                  self.train_shpfiles_location, self.test_shpfiles_location,
                                                  self.output_data_location, self.svm_c_value, self.svm_gamma_value)

        elif self.selected_ml_classification_method == "RF":
            print("RF Method Selected!")
            self.rf_njobs_value = self.input_rf_njobs_txt_field.get(1.0, END)
            self.rf_nestimators_value = self.input_rf_nestimators_txt_field.get(1.0, END)
            ml_classifiers_obj.set_rf_parameters(self.selected_ml_classification_method, self.image_file_name,
                                                 self.train_shpfiles_location, self.test_shpfiles_location,
                                                 self.output_data_location, self.rf_njobs_value,
                                                 self.rf_nestimators_value)

        elif self.selected_ml_classification_method == "MLP":
            print("MLP Method Selected!")
            self.mlp_fst_hdn_lyr_value = self.input_mlp_fst_hdn_lyr_txt_field.get(1.0, END)
            self.mlp_snd_hdn_lyr_value = self.input_mlp_snd_hdn_lyr_txt_field.get(1.0, END)
            self.mlp_trd_hdn_lyr_value = self.input_mlp_trd_hdn_lyr_txt_field.get(1.0, END)
            self.mlp_frt_hdn_lyr_value = self.input_mlp_frt_hdn_lyr_txt_field.get(1.0, END)
            ml_classifiers_obj.set_mlp_parameters(self.selected_ml_classification_method, self.image_file_name,
                                                 self.train_shpfiles_location, self.test_shpfiles_location,
                                                 self.output_data_location, self.mlp_fst_hdn_lyr_value,
                                                 self.mlp_snd_hdn_lyr_value, self.mlp_trd_hdn_lyr_value, self.mlp_frt_hdn_lyr_value)

        elif self.selected_ml_classification_method == "KNN":
            print("KNN Method Selected!")
            self.knn_knnparam_value = self.input_knn_knnparam_txt_field.get(1.0, END)
            ml_classifiers_obj.set_knn_parameters(self.selected_ml_classification_method, self.image_file_name,
                                                 self.train_shpfiles_location, self.test_shpfiles_location,
                                                 self.output_data_location, self.knn_knnparam_value)


        else:
            print("No Method Selected!")

        self.classify_btn["state"] = NORMAL
        self.cancel_btn["state"] = NORMAL

    def classify(self):
        t1 = threading.Thread(target=self.show_progress_bar, args=(400,))
        t2 = threading.Thread(target=self.execute_classification)
        t1.start()
        t2.start()

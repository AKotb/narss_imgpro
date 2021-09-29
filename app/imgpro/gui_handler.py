from tkinter import Tk, Menu, messagebox


class GUIHandler():

    def __init__(self, master=None):
        Tk.frame.__init__(self, master)
        self.master = master
        self.init_window()

    def init_window(self):
        # window title in the title bar
        self.master.title("NARSS Image Processing Application")

        menubar = Menu(self.master)
        self.master.config(menu=menubar)

        # File Menu
        file_menu = Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open", command=self.open)
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

        # Help Menu
        help_menu = Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self.about)
        menubar.add_cascade(label="Help", menu=help_menu)

    def exit(self):
        exit()

    def open(self):
        print("Open Menu Item Pressed!")

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

from tkinter import Tk
from app.imgpro.gui_handler import GUIHandler


def main():
    root = Tk()
    root.geometry("600x20+50+50")
    app = GUIHandler(root)
    root.mainloop()


if __name__ == '__main__':
    main()

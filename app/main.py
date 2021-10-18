from tkinter import Tk

from app.imgpro.gui_handler import GUIHandler


def print_welcome(name):
    print(f'Welcome to , {name}')


def main():
    root = Tk()
    root.geometry("600x20+50+50")
    app = GUIHandler(root)
    root.mainloop()


if __name__ == '__main__':
    print_welcome('NARSS Image Processing App')
    main()

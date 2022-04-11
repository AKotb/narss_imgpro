import time
from tkinter import *
from tkinter.ttk import *

ws = Tk()
ws.title('PythonGuides')
ws.geometry('400x250+1000+300')


def step():
    for i in range(100):
        ws.update_idletasks()
        pb1['value'] += 1

        time.sleep(0.05)


pb1 = Progressbar(ws, orient=HORIZONTAL, length=300, mode='determinate')
pb1.pack(expand=True)

Button(ws, text='Start', command=step).pack()

ws.mainloop()

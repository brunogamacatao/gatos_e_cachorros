import PIL.Image
from main import testa_imagem, carrega_rede
from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
import PIL.ImageTk

class App(Frame):
    def chg_image(self):
        if self.im.mode == "1": # bitmap image
            self.img = PIL.ImageTk.BitmapImage(self.im, foreground="white")
        else:              # photo image
            self.img = PIL.ImageTk.PhotoImage(self.im)
        self.la.config(image=self.img, bg="#000000",
            width=self.img.width(), height=self.img.height())

    def open(self):
        self.filename = filedialog.askopenfilename()
        if self.filename != "":
            self.im = PIL.Image.open(self.filename)
        self.chg_image()
    
    def classifica(self):
        classe = testa_imagem(self.filename)
        messagebox.showinfo('Classificação', 'A imagem é um {}'.format(classe))

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master.title('Classificador de Imagens')

        self.filename = ''

        fram = Frame(self)
        Button(fram, text="Abrir Arquivo", command=self.open).pack(side=LEFT)
        Button(fram, text="Classifica", command=self.classifica).pack(side=LEFT)
        fram.pack(side=TOP, fill=BOTH)

        self.la = Label(self)
        self.la.pack()

        self.pack()

if __name__ == "__main__":
    carrega_rede()
    app = App(); app.mainloop()
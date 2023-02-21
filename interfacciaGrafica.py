import tkinter as tk
from PIL import Image, ImageTk
from itertools import count
import os

IMAGE_PATH = 'logo2.jpg'
GIFIN_PATH='inserimento.gif'
ID_PATH='hand_scan.png'
ID_PATH='carta-identità-elettronica.jpg'
NEG_PATH='negato.jpg'
WIDTH, HEIGTH = 600, 400

FONT=("TkHeadingFont bold",10)


class ImageLabel(tk.Label):
    """a label that displays images, and plays them if they are gifs"""
    def load(self, im):
        if isinstance(im, str):
            im = Image.open(im)
        self.loc = 0
        self.frames = []

        try:
            for i in count(1):
                self.frames.append(ImageTk.PhotoImage(im.copy()))
                im.seek(i)
        except EOFError:
            pass

        try:
            self.delay = im.info['duration']
        except:
            self.delay = 100

        if len(self.frames) == 1:
            self.config(image=self.frames[0])
        else:
            self.next_frame()

    def unload(self):
        self.config(image="")
        self.frames = None

    def next_frame(self):
        if self.frames:
            self.loc += 1
            self.loc %= len(self.frames)
            self.config(image=self.frames[self.loc])
            self.after(self.delay, self.next_frame)

def center(win):
    win.update_idletasks()
    width = WIDTH
    height = HEIGTH
    x = (win.winfo_screenwidth() // 2) - (width // 2)
    y = (win.winfo_screenheight() // 2) - (height // 2)
    win.geometry('{}x{}+{}+{}'.format(width, height, x, y))

def interfaccia():
    global root
    root = tk.Tk()
    center(root)
    root.resizable(False,False)
    root.iconbitmap('favicon.ico')
    root.title('Pagina iniziale')
    #root.attributes('-transparentcolor', 'red')

    # Display image on a Label widget.
    img = ImageTk.PhotoImage(Image.open(IMAGE_PATH).resize((WIDTH, HEIGTH)), master=root)
    lbl = tk.Label(root, image=img)
    lbl.img = img  # Keep a reference in case this code put is in a function.
    lbl.place(relx=0.5, rely=0.5, anchor='center')  # Place label in center of parent

    fbutt1=tk.Frame(root)
    fbutt1.place(x=WIDTH//4,y=HEIGTH//1.5)

    fbutt2=tk.Frame(root)
    fbutt2.place(x=WIDTH//2,y=HEIGTH//1.5)

    button1 = tk.Button(fbutt1, font=FONT, text="INSERIMENTO DATI", bg="white", fg="red", command=inserimento) #command=cosa fare alla pressione del tasto
    button1.pack(side=tk.LEFT ,ipadx=5)
    button2= tk.Button(fbutt2,font=FONT, text="AUTENTICAZIONE BIOMETRICA", bg="white", fg="red", command=identificazione )
    button2.pack(side=tk.RIGHT,ipadx=5)
    root.mainloop()

    return mod
    
def inserimento():
    global mod
    mod='1'
    root.destroy()
    
def fin_ins():    
    global rt
    rt = tk.Tk()
    rt.iconbitmap('favicon.ico')
    rt.title('Pagina inserimento')
    center(rt)
    rt.resizable(False,False)
    lbl = ImageLabel(rt)
    lbl.pack()
    lbl.load(GIFIN_PATH)
    lbl.place(relx=0.5, rely=0.5, anchor='center')  # Place label in center of parent
    lbl_dati=tk.Label(rt,font=('Helvetica',24),padx=5, pady=5, text='Inserimento dati in corso...')
    lbl_dati.pack()
    #ret_id(str(10))
    #print(mod)
    # newWindow = tk.Toplevel(root)
    rt.protocol("WM_DELETE_WINDOW", lambda: (rt.quit(), rt.destroy()))
    rt.mainloop()


def identificazione():
    global mod
    mod='2'
    root.destroy()
    
def fin_autenticazione():   
    global rt
    def invio(id_entry, lbl_id, button):
        global num
        id_entry.pack_forget()
        button.pack_forget()
        lbl_id.pack_forget()
        num=id_entry.get()
        #print(num)
        lbl_id=tk.Label(rt, font=FONT,text='ID inserito correttamente')
        lbl_id.pack(padx=5, pady=5)
        lbl_w=tk.Label(rt, font=FONT,text='Attendere...processo di autenticazione in corso')
        lbl_w.pack(padx=5, pady=5)
        rt.after(10000, rt.destroy) #dopo 10 secondi si chiude
        
    rt = tk.Tk()
    rt.iconbitmap('favicon.ico')
    rt.title('Pagina identificazione')
    center(rt)
    rt.resizable(False,False)
    img = ImageTk.PhotoImage(Image.open(ID_PATH).resize((WIDTH, HEIGTH)), master=rt)
    lbl = tk.Label(rt, image=img)
    lbl.img = img  # Keep a reference in case this code put is in a function.
    lbl.place(relx=0.5, rely=0.5, anchor='center')  # Place label in center of parent

    lbl_id=tk.Label(rt, text='Inserire identificativo:')
    lbl_id.pack(padx=5, pady=5)
    identità=tk.StringVar()
    id_entry=tk.Entry(rt, textvariable=identità, show="*")
    id_entry.pack()
    button=tk.Button(rt, text='Invio ID', command=lambda:invio(id_entry, lbl_id, button))
    button.pack(padx=5, pady=5)
    rt.mainloop()
    
    return num

def Id_corretta(path, classe, percentuale):
    #rt.quit()
    global rti
    rti = tk.Tk()
    rti.iconbitmap('favicon.ico')
    rti.title('Pagina Id corretta')
    center(rti)
    rti.resizable(False,False)
    img = ImageTk.PhotoImage(Image.open(path).resize((250,200)), master=rti)
    lbl = tk.Label(rti, image=img)
    lbl.img = img  # Keep a reference in case this code put is in a function.
    lbl.pack(side=tk.LEFT)
    lbl_text=tk.Label(rti,font=FONT, text='AUTENTICAZIONE AVVENUTA CORRETTAMENTE')
    lbl_text.place(x=WIDTH//2.25,y=HEIGTH//2.5)
    lbl2_text=tk.Label(rti,font=FONT,fg="green", text='CLASSE--> '+str(classe)+'\n PERCENTUALE--> '+str(percentuale))
    lbl2_text.place(x=WIDTH//1.75,y=HEIGTH//2)
    rti.protocol("WM_DELETE_WINDOW", lambda: (rti.quit(), rti.destroy()))
    rti.mainloop()
    
    
def Id_Non_corretta():
    rt.quit()
    #os.chdir("C:/Users/Alberto/Desktop/TesiMagistrale/filePython")
    global rti
    rti = tk.Tk()
    rti.iconbitmap('favicon.ico')
    rti.title('Pagina Id non corretta')
    center(rti)
    rti.resizable(False,False)
    img = ImageTk.PhotoImage(Image.open(NEG_PATH).resize((WIDTH, HEIGTH)), master=rti)
    lbl = tk.Label(rti, image=img)
    lbl.img = img  # Keep a reference in case this code put is in a function.
    lbl.place(relx=0.5, rely=0.5, anchor='center')  # Place label in center of parent
    rti.protocol("WM_DELETE_WINDOW", lambda: (rti.quit(), rti.destroy()))
    rti.mainloop()


def ret_id(ide): #ritorna l'identità a seguito dell'inserimento dati
    rt.quit()
    global rti
    rti=tk.Tk()
    rti.iconbitmap('favicon.ico')
    rti.title('Pagina numero identificativo')
    center(rti)
    rti.resizable(False,False)
    img = ImageTk.PhotoImage(Image.open(ID_PATH).resize((WIDTH, HEIGTH)), master=rti)
    lbl = tk.Label(rti, image=img)
    lbl.img = img  # Keep a reference in case this code put is in a function.
    lbl.place(relx=0.5, rely=0.5, anchor='center')  # Place label in center of parent
    
    testo=tk.Frame(rti)
    testo.place(x=WIDTH//8,y=HEIGTH//12)
    lbl_id=tk.Label(testo,font=('Helvetica',20), text='Identificativo utente:\n'+ide)
    lbl_id.pack()
    rti.protocol("WM_DELETE_WINDOW", lambda: (rti.quit(), rti.destroy()))
    rti.mainloop()

#Id_corretta("C:/Users/Alberto/Desktop/TesiMagistrale/Dataset/Palmo/test/011_test.jpg", 100, 0.8)
#interfaccia()    
#Id_Non_corretta()
# def fine_inserimento(idx):
    # rti.destroy()
    
# s=fin_ident()
# print(s)
#ret_id(str(10))

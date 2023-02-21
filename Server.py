from base64 import b64decode
import socketio
import eventlet
from termcolor import colored
import os
from aes import *
from interfacciaGrafica import *
import threading
from neural import *
import hashlib

#os.system('color')
acquisition_path="C:/Users/Alberto/Desktop/TesiMagistrale/Dataset/"
file_path="C:/Users/Alberto/Desktop/TesiMagistrale/filePython"
palmo_path=os.path.join(acquisition_path, 'Palmo')
dorso_path=os.path.join(acquisition_path, 'Dorso')
cnt=0

def create_id():
    numero=0
    num=os.listdir(palmo_path+'/train')
    #print (num)
    
    if not num:
        max='%03d' % (numero+1)
        return max
    else:
        max=numero
        for image in num:
            numero=int(image)
            if numero>max:
                max=numero
        max='%03d' %  (max+1)
        return max

class Folder():
    def __init__(self,nome):
        self.path=os.path.join(acquisition_path, nome+'/')
        
    def create_folder(self, i):
            try:
                os.makedirs(self.path+'/'+i, exist_ok = True)
                return self;
            except OSError as error:
                print("Error!")
                return False;

sio = socketio.Server()
app = socketio.WSGIApp(sio)

@sio.event
def connect(sid, environ):
    print(colored(f'connect {sid}]', 'green'))

    
@sio.on('richiesta')
def invio_id(sid): 
    global identità
    global id_hash
    global mod
    #print("Inserire modalità di funzionamento: ")
    #mod=input("1--Inserimento dati, 2--Identificazione: ")
    os.chdir(file_path)
    mod=interfaccia()
    #global s
    if mod=='1':
        s=threading.Thread(target=fin_ins, args=()) #cosi si lascia la finestra fino alla fine dell'inserimento
        s.start()
        identità=create_id()
        # id_hash=hashlib.md5(identità.encode()).hexdigest()
        # with open("identità.txt", 'a') as file:
            # file.write(identità+'_'+id_hash+'\n')

        fpt=Folder("Palmo/train").create_folder(identità)
        fdt=Folder("Dorso/train").create_folder(identità)
        fpv=Folder("Palmo/valid").create_folder(identità)
        fdv=Folder("Dorso/valid").create_folder(identità)

    elif mod=='2':
        #s=threading.Thread(target=fin_ident, args=())
        #s.start()
        identità=fin_autenticazione() #qua si rimane in attesa del valore di identità
        # with open("identità.txt", 'r') as file:
            # while True: 
                # line = file.readline() 
                # code=line.split('_')[1].split('\n')[0]
                # if code==id_hash:
                    # identità=line.split('_')[0]
                    # break
                    
    packet=identità+'_'+mod
    print("Il vostro identificativo è: "+identità)
    sio.emit('identità', encrypt(packet))

@sio.on('invio')
def estrai(sid,invio):
    global cnt
    cnt=cnt+1
    
    if mod=='1':
        ricevuto=decrypt_packet(invio)
        img=ricevuto[0]
        nome=ricevuto[1].decode("utf-8")
        mano=ricevuto[2].decode("utf-8")
        wavelenght=ricevuto[3].decode("utf-8")
        num=ricevuto[4].decode("utf-8")
        if nome=="Palmo":
            path=Folder("Palmo").path
            if ((num=='01' or num=='02') and (wavelenght=='850' or wavelenght=='940')):
                path=os.path.join(path, 'valid/'+identità)
            else:
                path=os.path.join(path, 'train/'+identità)
            data=b64decode(img)
            img_file = open(path+'/'+ identità+'_'+mano+'_'+wavelenght+'_'+num+'.jpg', 'wb')
            img_file.write(data)
            img_file.close()
     
        else:
            path=Folder("Dorso").path
            if ((num=='01' or num=='02') and (wavelenght=='850' or wavelenght=='940')):
                path=os.path.join(path, 'valid/'+identità)
            else:
                path=os.path.join(path, 'train/'+identità)
            data=b64decode(img)
            img_file = open(path+'/'+ identità+'_'+nome+'_'+mano+'_'+wavelenght+'_'+num+'.jpg', 'wb')
            img_file.write(data)
            img_file.close()
        print("Nome: "+nome+", mano: "+mano+ ", Lunghezza donda: " +wavelenght+", Numero: "+ num)
        
        if cnt==120: #riaddestro il modello
            training_model("Palmo")
            #training_model("Dorso")
            cnt==0
        
    elif mod=='2': #autenticazione fatta solo sul palmo
        ricevuto=decrypt_packet(invio)
        img=ricevuto[0]
        data=b64decode(img)
        path=Folder("Palmo").path
        num_classes = len(os.listdir(path+"/train"))  
        t_path=os.path.join(path, 'test')
        global test_path
        test_path=t_path+'/'+ str(identità)+'_test.jpg'
        img_file = open(test_path, 'wb')
        img_file.write(data)
        img_file.close()
        global classe, percentuale
        classe, percentuale= predict(path+'/'+str(num_classes)+'classes_Model.pt',test_path)
        print(classe, percentuale)
        if (identità==classe and classe!='012' and percentuale>0.75):
            print("CORRETTO")
        


@sio.event
def disconnect(sid):
    os.chdir(file_path)
    if mod=='1':
        ret_id(identità)
    elif (identità==classe and classe!='012' and percentuale>0.75):
        #print(test_path)
        Id_corretta(test_path, classe, percentuale)
    else:
        Id_Non_corretta()
	#ritorno l'identificaizone
    #s.terminate()
    print(colored(f'disconnect {sid}', 'green'))



eventlet.wsgi.server(eventlet.listen(('', 5000)), app, log_output=False)

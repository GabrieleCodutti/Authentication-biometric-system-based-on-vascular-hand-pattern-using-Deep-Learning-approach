from picamera import PiCamera
import RPi.GPIO as GPIO
from time import sleep
import os
import socketio
from termcolor import colored
from base64 import b64encode, b64decode
import threading
from aes import *
from io import BytesIO

acquisition_path="/home/pi/Desktop/Capture/Dataset/"
camera_time_warmup=2
img_resolution=(768,576)
img_color=(128,128)
img_rotation_in_degree=90
conversion_factor=1000000
wavelenght=[465,635,700,850,940]
gpio_port=[21, 20, 16, 26, 19]
framerate=[30, 15, 8, 80 ,10]
iso=[50, 400, 1600,10,50]
invio=[]


def initialize_gpio():
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(16, GPIO.OUT)
    GPIO.setup(19, GPIO.OUT)
    GPIO.setup(20, GPIO.OUT)
    GPIO.setup(21, GPIO.OUT)
    GPIO.setup(26, GPIO.OUT)
    GPIO.output(16, GPIO.LOW)
    GPIO.output(19, GPIO.LOW)
    GPIO.output(20, GPIO.LOW)
    GPIO.output(21, GPIO.LOW)
    GPIO.output(26, GPIO.LOW)

    
def send_packet(invio):
        while True:
            for i in range(len(invio)):
                print(colored('packet sent'+str(i), 'yellow'))
                try:
                    sio.call('invio',encrypt_packet(invio[i]) , timeout=10)
                except TimeoutError:
                    print(colored('Error', 'yellow'))
                    break
                else:
                    print(colored('ack received', 'yellow')) 
            break
        sio.disconnect()

def capture_image(camera, nome, mano, wavelenght, gpio_port, framerate,iso):
    global invio
    for i in range (6):
        stream = BytesIO()
        GPIO.output(gpio_port, GPIO.HIGH)
        camera.resolution= img_resolution
        camera.color_effects= img_color
        camera.rotation=img_rotation_in_degree
        camera.framerate=framerate
        camera.shutter_speed=round(conversion_factor/framerate)
        camera.iso=iso
        camera.capture(stream,'jpeg')
        stream.seek(0)
        data = b64encode(stream.read())    
        invio.append([data, nome, mano, str(wavelenght), '%02d'%(i+1)]) 

    GPIO.output(gpio_port, GPIO.LOW)
    
    
def acquisition(nome, mano):
    print("Inizio acquisizione immagini del "+nome+" "+ mano+" della mano:")
    initialize_gpio()
    camera=PiCamera()
    for i in range(len(wavelenght)):
        sleep(camera_time_warmup)
        capture_image(camera,nome,mano, wavelenght[i], gpio_port[i], framerate[i], iso[i])
    camera.close()
    GPIO.cleanup()
    
def inserimento():
    acquisition("Palmo","r")
    sleep(5.0)
    acquisition("Dorso","r")
    sleep(5.0)
    acquisition("Palmo","l")
    sleep(5.0)
    acquisition("Dorso","l")
    send_packet(invio)
    
def autenticazione():
    print("Processo di identificazione in corso... inserire Palmo Destro")
    initialize_gpio()
    camera=PiCamera()
    sleep(camera_time_warmup)
    stream = BytesIO()
    GPIO.output(19, GPIO.HIGH) #porta corrispondente alla lunghezza 940
    camera.resolution= img_resolution
    camera.color_effects= img_color
    camera.rotation=img_rotation_in_degree
    camera.framerate=framerate[4]
    camera.shutter_speed=round(conversion_factor/framerate[4])
    camera.iso=iso[4]
    camera.capture(stream,'jpeg')
    stream.seek(0)
    data = b64encode(stream.read())     # convert to base64 format, Return the encoded string.
    invio.append([data,"Palmo","r",'940', '01']) #invio al server l'immagine, se il palmo o dorso , se dx o sx e la lunghezza d'onda
    GPIO.output(19, GPIO.LOW)
    send_packet(invio)

if __name__=="__main__":
    sio = socketio.Client()

    @sio.event
    def connect():
       print(colored('connection established', 'green'))
       sio.emit('richiesta')
    
    @sio.event
    def identità(packet):
        packet=decrypt(packet)
        packet=packet.decode('utf-8')
        ide=packet.split('_')[0]
        mod=packet.split('_')[1]
        print("ricevo identità e modalità: "+ packet)
        global ident
        ident=ide
        if mod=='1':
            inserimento()
        elif mod=='2':
            autenticazione()


    @sio.event
    def disconnect():
        print(colored('disconnected from server', 'green'))
        
    sio.connect('http://192.168.38.254:5000', wait_timeout = 10)


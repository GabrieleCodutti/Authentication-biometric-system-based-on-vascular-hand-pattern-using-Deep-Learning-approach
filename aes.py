from Crypto.Random import get_random_bytes
from Crypto.PublicKey import RSA
from Crypto.Cipher import AES, PKCS1_OAEP
import math


def encrypt(data):

    if(type(data) is str): #il dato in ingresso deve essere in bytes
        data=data.encode('utf-8')
    elif (type(data) is int):
        data=(data).to_bytes(math.ceil((data).bit_length() / 8), byteorder = 'big', signed=False)
        
    recipient_key = RSA.import_key(open("receiver.pem").read())
    session_key = get_random_bytes(16)

    # Criptare la chiave di sessione con la chiave pubblica RSA
    cipher_rsa = PKCS1_OAEP.new(recipient_key)
    enc_session_key = cipher_rsa.encrypt(session_key)

    # Criptare i dati con la chiave di sessione AES
    cipher_aes = AES.new(session_key, AES.MODE_EAX)
    ciphertext, tag = cipher_aes.encrypt_and_digest(data)

    packet= enc_session_key+cipher_aes.nonce+tag+ciphertext
    
    return packet

def decrypt(packet):
    
    private_key = RSA.import_key(open("private.pem").read())
    enc_session_key=packet[:private_key.size_in_bytes()]
    nonce=packet[private_key.size_in_bytes():private_key.size_in_bytes()+16]
    tag=packet[private_key.size_in_bytes()+16:private_key.size_in_bytes()+32]
    ciphertext=packet[private_key.size_in_bytes()+32:]

    # Decriptare la chiave di sessione con la chiave privata RSA
    cipher_rsa = PKCS1_OAEP.new(private_key)
    session_key = cipher_rsa.decrypt(enc_session_key)

    # Decriptare i dati con la chiave di sessione AES
    cipher_aes = AES.new(session_key, AES.MODE_EAX, nonce)
    data = cipher_aes.decrypt_and_verify(ciphertext, tag)
    
    return data

def encrypt_packet(data):
    invio=[]
    for i in range(len(data)):
        invio.append(encrypt(data[i]))
    return invio
    
def decrypt_packet(data):
    ricevuto=[]
    for i in range(len(data)):
        ricevuto.append(decrypt(data[i]))
    return ricevuto
        
def generate_keys():
    key = RSA.generate(2048)
    private_key = key.export_key()
    file_out = open("private.pem", "wb")
    file_out.write(private_key)
    file_out.close()

    public_key = key.publickey().export_key()
    file_out = open("receiver.pem", "wb")
    file_out.write(public_key)
    file_out.close()


#%%
import torch, torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import torchinfo
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

model_path="C:/Users/Alberto/Desktop/TesiMagistrale/Dataset/Palmo/11classes_Model.pt"
img_path="C:\\Users\\Alberto\\Desktop\\TesiMagistrale\\001_r_940_03.jpg"

def save_plots(history, num, percorso):
    history = np.array(history)
    plt.plot(history[:,0:2])
    plt.legend(['Tr Loss', 'Val Loss'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.ylim(0,5)
    plt.savefig(percorso+'/'+str(num)+'_Loss.png')
    plt.close()
    plt.plot(history[:,2:4])
    plt.legend(['Tr Accuracy', 'Val Accuracy'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.ylim(0,1)
    plt.savefig(percorso+'/'+str(num)+'_Acc.png')
    plt.close()


image_transforms = { 
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=10),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], 
                             [0.5, 0.5, 0.5])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], 
                             [0.5, 0.5, 0.5])
    ]),
    'test': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5])
    ])
    }
    
# Load the Data

# Set train and valid directory paths
#idx_to_class={}
#os.chdir("C:/Users/Alberto/Desktop/TesiMagistrale/Dataset/Palmo")
def load(): #funzione che fissa alcune variabili globali in base a delle scelte
    global idx_to_class, train_data_size, valid_data_size, train_data_loader, valid_data_loader

    train_directory = os.path.join(os.getcwd(), "train")
    #print(train_directory)
    valid_directory = os.path.join(os.getcwd(), "valid")

    # Batch size
    bs = 128
    # Number of classes
    #num_classes = len(os.listdir(valid_directory)) 

    # Load Data from folders
    
    data = {
    'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
    'valid': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid'])
    }

    idx_to_class = {v: k for k, v in data['train'].class_to_idx.items()}

    # Size of Data, to be used for calculating Average Loss and Accuracy
    train_data_size = len(data['train'])
    valid_data_size = len(data['valid'])

    # Create iterators for the Data loaded using DataLoader module
    train_data_loader = DataLoader(data['train'], batch_size=bs, shuffle=True)
    valid_data_loader = DataLoader(data['valid'], batch_size=bs, shuffle=True)

def train_and_validate(device, model, loss_criterion, optimizer, epochs=25):
    '''
    Function to train and validate
    Parameters
        :param model: Model to train and validate
        :param loss_criterion: Loss Criterion to minimize
        :param optimizer: Optimizer for computing gradients
        :param epochs: Number of epochs (default=25)
  
    Returns
        model: Trained Model with best validation accuracy
        history: (dict object): Having training loss, accuracy and validation loss, accuracy
    '''
    
    start = time.time()
    history = []
    best_acc = 0.0

    print("Inizio Training: aggiornamento modello in corso")

    for epoch in range(epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch+1, epochs))
        
        # Set to training mode
        model.train()
        
        # Loss and Accuracy within the epoch
        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0
        
        for i, (inputs, labels) in enumerate(train_data_loader):

            inputs = inputs.to(device)
            labels = labels.to(device)

            # Clean existing gradients
            optimizer.zero_grad()
            
            # Forward pass - compute outputs on input data using the model
            outputs = model(inputs)
            
            # Compute loss
            loss = loss_criterion(outputs, labels)
            
            # Backpropagate the gradients
            loss.backward()
            #loss.requires_grad = True #(se uso il model)
            
            # Update the parameters
            optimizer.step()
            
            # Compute the total loss for the batch and add it to train_loss
            train_loss += loss.item() * inputs.size(0)
            
            # Compute the accuracy
            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            
            # Convert correct_counts to float and then compute the mean
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            
            # Compute total accuracy in the whole batch and add to train_acc
            train_acc += acc.item() * inputs.size(0)
            #print("Batch number: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}".format(i, loss.item(), acc.item()))

            
        # Validation - No gradient tracking needed
        with torch.no_grad():

            # Set to evaluation mode
            model.eval()

            # Validation loop
            for j, (inputs, labels) in enumerate(valid_data_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Forward pass - compute outputs on input data using the model
                outputs = model(inputs)

                # Compute loss
                loss = loss_criterion(outputs, labels)

                # Compute the total loss for the batch and add it to valid_loss
                valid_loss += loss.item() * inputs.size(0)

                # Calculate validation accuracy
                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))

                # Convert correct_counts to float and then compute the mean
                acc = torch.mean(correct_counts.type(torch.FloatTensor))

                # Compute total accuracy in the whole batch and add to valid_acc
                valid_acc += acc.item() * inputs.size(0)
                #print("Validation Batch number: {:03d}, Validation: Loss: {:.4f}, Accuracy: {:.4f}".format(j, loss.item(), acc.item()))
            
        # Find average training loss and training accuracy
        avg_train_loss = train_loss/train_data_size 
        avg_train_acc = train_acc/train_data_size

        # Find average training loss and training accuracy
        avg_valid_loss = valid_loss/valid_data_size 
        avg_valid_acc = valid_acc/valid_data_size

        writer.add_scalars("Loss", {'Train':avg_train_loss, 'Validate':avg_valid_loss}, epoch)
        #writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalars("Accuracy",{'Train':avg_train_acc, 'Validate':avg_valid_acc}, epoch)
        #writer.add_scalar("Loss/validate", avg_valid_loss, epoch)

        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])
        epoch_end = time.time()
    
        print("Epoch : {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation : Loss : {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(epoch+1, avg_train_loss, avg_train_acc*100, avg_valid_loss, avg_valid_acc*100, epoch_end-epoch_start))
        
        # Save if the model has best accuracy till now
        #torch.save(model, dataset+'_model_'+str(epoch)+'.pt')
            
    return model, history

def predict(model_path, img_path):
    os.chdir("C:/Users/Alberto/Desktop/TesiMagistrale/Dataset/Palmo") #ci interessa l'autenticazione tramite palmo
    load()
    #se voglio caricare il modello che ho addestrato in precedenza
    model= torch.load(model_path)
    if torch.cuda.is_available():
        model.cuda()
    transform = image_transforms['test']
    test_image = Image.open(img_path).convert('RGB')
    plt.imshow(test_image)
    
    test_image_tensor = transform(test_image)

    if torch.cuda.is_available():
        test_image_tensor = test_image_tensor.view(1,3, 224, 224).cuda()
    else:
        test_image_tensor = test_image_tensor.view(1,3, 224, 224) 
    
    with torch.no_grad():
        model.eval()
        # Model outputs log probabilities
        out = model(test_image_tensor)
        ps = torch.exp(out)
        topk, topclass = ps.topk(3, dim=1)

    return(idx_to_class[topclass.cpu().numpy()[0][0]],topk.cpu().numpy()[0][0])
    
#%%
#print(predict(model_path, "C:\\Users\\Alberto\\Desktop\\TesiMagistrale\\Dataset\\Palmo\\valid\\011\\011_l_940_01.jpg"))
 #%%     
def training_model(nome):
    if nome=="Palmo":
        percorso='C:/Users/Alberto/Desktop/TesiMagistrale/Dataset/Palmo'
        os.chdir(percorso)
    elif nome=="Dorso":
        percorso='C:/Users/Alberto/Desktop/TesiMagistrale/Dataset/Dorso'
        os.chdir(percorso)
    load()
    global writer
    writer = SummaryWriter()
    num_classes = len(os.listdir(percorso+"/train"))  
    model= torch.load(percorso+"/"+str(num_classes-1)+"classes_Model.pt") #carico modello precedente
    if torch.cuda.is_available():
        model.cuda()

    # Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier[4]= nn.Linear(4096, 4096)
    model.classifier[6] = nn.Linear(4096, num_classes)
    model.classifier.add_module("7", nn.LogSoftmax(dim = 1))

    # Define Optimizer and Loss Function
    loss_func = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters())

    torch.cuda.memory_summary(device=None, abbreviated=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_epochs = 20
    trained_model, history = train_and_validate(device,model.to(device), loss_func, optimizer, num_epochs)
    #os.chdir(percorso)
    torch.save(history,str(num_classes)+'classes_History.pt')
    torch.save(trained_model,str(num_classes)+'classes_Model.pt')
    torch.cuda.empty_cache()
    writer.flush()
    writer.close()
    save_plots(history,num_classes, percorso)
    
#training_model("Palmo")
# %%

#importing necessary libraries
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import torch.utils.data
import pandas as pd
from collections import OrderedDict
from PIL import Image
import argparse
import json

# define Mandatory and Optional Arguments for the script
parser = argparse.ArgumentParser (description = "Parser of training script")

parser.add_argument ('data_dir', help = 'Provide data directory. Mandatory argument', type = str)
parser.add_argument ('--save_dir', help = 'Provide saving directory. Optional argument', type = str)
parser.add_argument ('--arch', help = 'Vgg13 can be used if this argument specified, otherwise Alexnet will be used', type = str)
parser.add_argument ('--lrn', help = 'Learning rate, default value 0.001', type = float)
parser.add_argument ('--hidden_units', help = 'Hidden units in Classifier. Default value is 2048', type = int)
parser.add_argument ('--epochs', help = 'Number of epochs', type = int)
parser.add_argument ('--GPU', help = "Option to use GPU", type = str)

#setting values data loading
args = parser.parse_args ()

data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

#defining device: either cuda or cpu
if args.GPU == 'GPU':
    device = 'cuda'
else:
    device = 'cpu'

#data loading
if data_dir: #making sure we do have value for data_dir
    # Define your transforms for the training, validation, and testing sets
    train_data_transforms = transforms.Compose ([transforms.RandomRotation (30),
                                                transforms.RandomResizedCrop (224),
                                                transforms.RandomHorizontalFlip (),
                                                transforms.ToTensor (),
                                                transforms.Normalize ([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                                ])

    valid_data_transforms = transforms.Compose ([transforms.Resize (255),
                                                transforms.CenterCrop (224),
                                                transforms.ToTensor (),
                                                transforms.Normalize ([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                                ])

    test_data_transforms = transforms.Compose ([transforms.Resize (255),
                                                transforms.CenterCrop (224),
                                                transforms.ToTensor (),
                                                transforms.Normalize ([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                                ])
    # Load the datasets with ImageFolder
    train_image_datasets = datasets.ImageFolder (train_dir, transform = train_data_transforms)
    valid_image_datasets = datasets.ImageFolder (valid_dir, transform = valid_data_transforms)
    test_image_datasets = datasets.ImageFolder (test_dir, transform = test_data_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_image_datasets, batch_size = 64, shuffle = True)
    valid_loader = torch.utils.data.DataLoader(valid_image_datasets, batch_size = 64, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_image_datasets, batch_size = 64, shuffle = True)
    #end of data loading block

#mapping from category label to category name
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

def load_model (arch, hidden_units):
    if arch == 'vgg13': #setting model based on vgg13
        model = models.vgg13 (pretrained = True)
        for param in model.parameters():
            param.requires_grad = False
        if hidden_units: #in case hidden_units were given
            classifier = nn.Sequential  (OrderedDict ([
                            ('fc1', nn.Linear (25088, 4096)),
                            ('relu1', nn.ReLU ()),
                            ('dropout1', nn.Dropout (p = 0.3)),
                            ('fc2', nn.Linear (4096, hidden_units)),
                            ('relu2', nn.ReLU ()),
                            ('dropout2', nn.Dropout (p = 0.3)),
                            ('fc3', nn.Linear (hidden_units, 102)),
                            ('output', nn.LogSoftmax (dim =1))
                            ]))
        else: #if hidden_units not given
            classifier = nn.Sequential  (OrderedDict ([
                        ('fc1', nn.Linear (25088, 4096)),
                        ('relu1', nn.ReLU ()),
                        ('dropout1', nn.Dropout (p = 0.3)),
                        ('fc2', nn.Linear (4096, 2048)),
                        ('relu2', nn.ReLU ()),
                        ('dropout2', nn.Dropout (p = 0.3)),
                        ('fc3', nn.Linear (2048, 102)),
                        ('output', nn.LogSoftmax (dim =1))
                        ]))
    else: #setting model based on default Alexnet ModuleList
        arch = 'alexnet' #will be used for checkpoint saving, so should be explicitly defined
        model = models.alexnet (pretrained = True)
        for param in model.parameters():
            param.requires_grad = False
        if hidden_units: #in case hidden_units were given
            classifier = nn.Sequential  (OrderedDict ([
                            ('fc1', nn.Linear (9216, 4096)),
                            ('relu1', nn.ReLU ()),
                            ('dropout1', nn.Dropout (p = 0.3)),
                            ('fc2', nn.Linear (4096, hidden_units)),
                            ('relu2', nn.ReLU ()),
                            ('dropout2', nn.Dropout (p = 0.3)),
                            ('fc3', nn.Linear (hidden_units, 102)),
                            ('output', nn.LogSoftmax (dim =1))
                            ]))
        else: #if hidden_units not given
            classifier = nn.Sequential  (OrderedDict ([
                        ('fc1', nn.Linear (9216, 4096)),
                        ('relu1', nn.ReLU ()),
                        ('dropout1', nn.Dropout (p = 0.3)),
                        ('fc2', nn.Linear (4096, 2048)),
                        ('relu2', nn.ReLU ()),
                        ('dropout2', nn.Dropout (p = 0.3)),
                        ('fc3', nn.Linear (2048, 102)),
                        ('output', nn.LogSoftmax (dim =1))
                        ]))
    model.classifier = classifier #we can set classifier only once as cluasses self excluding (if/else)
    return model, arch

# Defining validation Function. will be used during training
def validation(model, valid_loader, criterion):
    model.to (device)

    valid_loss = 0
    accuracy = 0
    for inputs, labels in valid_loader:

        inputs, labels = inputs.to(device), labels.to(device)
        output = model.forward(inputs)
        valid_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return valid_loss, accuracy

#loading model using above defined functiion
model, arch = load_model (args.arch, args.hidden_units)

#Actual training of the model
#initializing criterion and optimizer
criterion = nn.NLLLoss ()
if args.lrn: #if learning rate was provided
    optimizer = optim.Adam (model.classifier.parameters (), lr = args.lrn)
else:
    optimizer = optim.Adam (model.classifier.parameters (), lr = 0.001)


model.to (device) #device can be either cuda or cpu
#setting number of epochs to be run
if args.epochs:
    epochs = args.epochs
else:
    epochs = 7

print_every = 40
steps = 0

#runing through epochs
for e in range (epochs):
    running_loss = 0
    for ii, (inputs, labels) in enumerate (train_loader):
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad () #where optimizer is working on classifier paramters only

        # Forward and backward passes
        outputs = model.forward (inputs) #calculating output
        loss = criterion (outputs, labels) #calculating loss (cost function)
        loss.backward ()
        optimizer.step () #performs single optimization step
        running_loss += loss.item () # loss.item () returns scalar value of Loss function

        if steps % print_every == 0:
            model.eval () #switching to evaluation mode so that dropout is turned off
            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                valid_loss, accuracy = validation(model, valid_loader, criterion)

            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                  "Valid Loss: {:.3f}.. ".format(valid_loss/len(valid_loader)),
                  "Valid Accuracy: {:.3f}%".format(accuracy/len(valid_loader)*100))

            running_loss = 0
            # Make sure training is back on
            model.train()

#saving trained Model
model.to ('cpu') #no need to use cuda for saving/loading model.
# Save the checkpoint
model.class_to_idx = train_image_datasets.class_to_idx #saving mapping between predicted class and class name,
#second variable is a class name in numeric

#creating dictionary for model saving
checkpoint = {'classifier': model.classifier,
              'state_dict': model.state_dict (),
              'arch': arch,
              'mapping':    model.class_to_idx
             }
#saving trained model for future use
if args.save_dir:
    torch.save (checkpoint, args.save_dir + '/checkpoint.pth')
else:
    torch.save (checkpoint, 'checkpoint.pth')

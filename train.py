# import module

import numpy as np
import time
import json

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict

import model_utils
from model_utils import build_model, save_checkpoint, validation

import argparse

# arguments
parser = argparse.ArgumentParser(description='Parameters for building model')
parser.add_argument('--hidden_units', action="store",
                    dest="hidden_units", default = '12595')
parser.add_argument('--epochs', action="store",
                    dest="epochs", default = '3')
parser.add_argument('--device', action="store",
                    dest="device", default = 'gpu')
parser.add_argument('--arch', action="store",
                    dest="arch", default = 'vgg')

args = vars(parser.parse_args())

# load the data

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# TODO: Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([
                                        transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
test_transforms = transforms.Compose([transforms.Resize(256),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])


# TODO: Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)


# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

class_to_idx = train_data.class_to_idx

# label mapping
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
## building the model

# hyperparameters
hidden_units = int(args['hidden_units'])
output_size=len(class_to_idx)
dropout = 0.5
learning_rate = 0.002
model_type = args['arch']

# create the model
model = build_model(class_to_idx, model_type, hidden_units, output_size, dropout)

# define criterion and optimizer

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.classifier.parameters(), lr=learning_rate, momentum = 0.9)

# train the network
epochs = int(args['epochs'])
print_every = 40
steps = 0

## change to cuda
device = 'cuda' if args['device'] == 'gpu' else 'cpu'

model.to(device)

for e in range(epochs):
    running_loss = 0
    for ii, (inputs, labels) in enumerate(trainloader):
        steps += 1

        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward and backward passes
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if steps % print_every == 0:
            model.eval() # This moves your model to evaluation mode
            with torch.no_grad(): #This turns off gradients during evaluation which is faster and saves memory 
                validation_loss, accuracy = validation(model, validloader, criterion)
            print("Epoch: {}/{}... ".format(e+1, epochs),
                  "Training Loss: {:.4f}".format(running_loss/print_every),
                  "Validation Loss: {:.4f}".format(validation_loss/print_every),
                  "Validation accuracy: {:.4f}".format(accuracy))

            running_loss = 0

# save checkpoint 
save_checkpoint('checkpoint4.pth', model_type, model, learning_rate, hidden_units)
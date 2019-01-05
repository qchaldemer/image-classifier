#module
from torchvision import models
import torch
from torch import nn
from collections import OrderedDict

def build_model(class_to_idx, model_type='vgg',
                   hidden_units=12595,
                   output_size = 102,
                   dropout = 0.5):
    
    # loading networks
    if model_type == 'resnet':
        model = models.resnet50(pretrained=True) 
    else:
        model = models.vgg19(pretrained=True) 
    
    #get the input size
    input_size = model.classifier[0].in_features
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
        
    #classifier
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_size, hidden_units)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=dropout)),
                          ('fc2', nn.Linear(hidden_units, output_size)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    model.classifier = classifier
    model.class_to_idx = class_to_idx
    
    return model

def save_checkpoint(filepath, model_type, model, learning_rate, hidden_units):
    
    network= {
            'model_type': model_type,
            'learning_rate': learning_rate,
            'hidden': hidden_units,
            'state_dict': model.state_dict(),
            'class_to_idx': model.class_to_idx}

    torch.save(network, filepath)
    
def validation(model, validloader, criterion):
    test_loss = 0
    accuracy = 0
    total = 0
    correct = 0

    for (inputs, labels) in validloader:
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        accuracy = (100 * correct / total)
        
    
    return test_loss, accuracy
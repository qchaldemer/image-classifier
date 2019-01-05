import numpy as np
import torch
from PIL import Image
import model_utils
from model_utils import build_model

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''                       
    
    # TODO: Process a PIL image for use in a PyTorch model
    
    pil_image = Image.open(image_path)
    
    width, height = pil_image.size
    ratio = float(width/height);
        
    if height > width:
        height = int(height * 256 / width)
        width = int(256)
    else:
        width = int(width * 256 / height)
        height = int(256)
        
    resized_image = pil_image.resize((width, height), Image.ANTIALIAS)
    
    # Crop center portion of the image
    x0 = (width - 224) / 2
    y0 = (height - 224) / 2
    x1 = x0 + 224
    y1 = y0 + 224
    crop_image = resized_image.crop((x0,y0,x1, y1))
    
    # Normalize:
    np_image = np.array(crop_image) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    return np_image.transpose(2,0,1)

def load_checkpoint(filepath):
    
    checkpoint = torch.load(filepath)
    
    learning_rate = checkpoint['learning_rate']
    hidden_units = checkpoint['hidden']
    class_to_idx = checkpoint['class_to_idx']
    model_type = checkpoint['model_type']
    
    output_size = len(class_to_idx)
    
    model = build_model(class_to_idx, model_type,
                   hidden_units, output_size)
    model.load_state_dict(checkpoint['state_dict'])
    
    return model, learning_rate, hidden_units, class_to_idx

def predict(image_path, device, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    # Calculate the class probabilities (softmax) for img
    model.eval()

    img = process_image(image_path)
    tensor_in = torch.from_numpy(img)

    tensor_in = tensor_in.float() 
    tensor_in = tensor_in.unsqueeze(0)

    model.to(device)
    tensor_in.to(device)
    
    with torch.no_grad():
        output = model.forward(tensor_in.cuda())
        
    output = torch.exp(output)
        
    topk_prob, topk_index = torch.topk(output, topk) 
    topk_prob = topk_prob.tolist()[0]
    topk_index = topk_index.tolist()[0]
    
    idx_to_cat = {value: key for key, value in model.class_to_idx.items()}
    
    top_cat = [idx_to_cat[ele] for ele in topk_index]

    return topk_prob, top_cat




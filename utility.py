import argparse
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import numpy as np
import pandas as pd
import json

def get_input_args():
    # Creates parse 
    parser = argparse.ArgumentParser(description='Training NN options')

    # Creates arguments
    parser.add_argument(
        'data_directory',
        action='store',
        help='path to directory of data'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        dest='save_dir',
        default='',
        help='path to directory to save the checkpoint'
    )
    parser.add_argument(
        '--arch',
        type=str,
        dest='arch',
        default='densenet169',
        help='chosen model (any model on \'torchvision.models\', please insert as value the exact name of the model => densenet169, vgg13...)'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        dest='learning_rate',
        default=0.001,
        help='chosen learning rate'
    )
    parser.add_argument(
        '--hidden_units',
        type=int,
        dest='hidden_units',
        default=1664,
        help='chosen hidden units, this value is dependent on the chosen model'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        dest='epochs',
        default=10,
        help='chosen epochs'
    )
    parser.add_argument(
        '--gpu',
        dest='gpu',
        action='store_true',
        help='use the GPU for training'
    )

    # returns parsed argument collection
    return parser.parse_args()

def do_deep_learning(model, trainloader, validloader, epochs, print_every, criterion, optimizer, device='cpu'):
    epochs = epochs
    steps = 0
    model = model.to(device)
    model.train()
    print("Training model...")

    e = 0
    while e < epochs:
        running_loss = 0
        ii = 0
        while ii < len(trainloader):
            steps += 1
            inputs, labels = next(iter(trainloader))
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            ii += 1

            if steps % print_every == 0:
                model.eval()
                accuracy = 0
                validation_loss = 0
                jj = 0
                while jj < len(validloader):
                    inputs, labels = next(iter(validloader))
                    inputs, labels = inputs.to(device), labels.to(device)
                    output = model.forward(inputs)
                    validation_loss += criterion(output, labels)
                    probabilities = torch.exp(output).data
                    equality = (labels.data == probabilities.max(1)[1])
                    accuracy += equality.type_as(torch.FloatTensor()).mean()
                    jj += 1

                print("Epoch: {}/{}... ".format(e + 1, epochs),
                      "| Training Loss: {:.4f}".format(running_loss / print_every),
                      "| Validation Loss: {:.3f}.. ".format(validation_loss / len(validloader)),
                      "| Validation Accuracy: {:.3f}%".format(accuracy / len(validloader) * 100))

                running_loss = 0
                model.train()

        e += 1

    print("Done training!")

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    epochs = checkpoint['epochs']
    optimizer = checkpoint['optimizer_dict']
    model = checkpoint['model']
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    for param in model.parameters():
        param.requires_grad = False
    return model

def process_image(image_path):
    ''' 
    Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns a NumPy array
    '''
    # Define target size for resizing
    target_size = 256, 256

    # Open the image using PIL
    image = Image.open(image_path)

    # Resize the image while maintaining aspect ratio
    image = image.resize(target_size)

    # Crop the center of the image
    crop_box = (16, 16, 240, 240)
    image = image.crop(crop_box)

    # Convert PIL image to NumPy array
    np_image = np.array(image)

    # Normalize the image
    np_image_normalized = (np_image / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]

    # Transpose the color channels
    np_image_normalized = np_image_normalized.transpose((2, 0, 1))

    return np_image_normalized

def view_classify(im,probs, classes, cat_to_name):

    print(create_dataframe(classes, probs, cat_to_name))

def create_dataframe(classes, probs, cat_to_name):
    if cat_to_name is not None:
        with open(cat_to_name, 'r') as f:
            cat_to_name_data = json.load(f)
        name_classes = [cat_to_name_data[i] for i in classes]
    else:
        name_classes = classes

    df = pd.DataFrame({'classes': name_classes, 'values': probs})

    return df

def predict(image_path, model, device, topk=5):
    ''' 
    Predict the class (or classes) of an image using a trained deep learning model.

    Parameters:
        image_path (str): Path to the image file.
        model (torch.nn.Module): Trained PyTorch model.
        device (str): Device to use ('cpu' or 'cuda' if available).
        topk (int): Number of top classes to return.

    Returns:
        Tuple: A tuple containing a list of probabilities and a list of class labels.
    '''
    # Load and preprocess the image
    image = torch.from_numpy(process_image(image_path))
    image = image.unsqueeze(0).float()

    # Move model and image to the specified device
    model, image = model.to(device), image.to(device)

    # Set the model to evaluation mode and disable gradient computation
    model.eval()
    with torch.no_grad():
        # Use unpacking to get probabilities and classes directly
        probs, classes = torch.exp(model.forward(image)).topk(topk)

    # Convert results to numpy arrays
    probs, classes = probs.cpu().numpy()[0], classes.cpu().numpy()[0]

    # Map indices to class labels
    idx_to_class = {idx: class_label for class_label, idx in model.class_to_idx.items()}
    class_labels = [idx_to_class[idx] for idx in classes]

    return probs, class_labels

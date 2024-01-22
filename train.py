import argparse
import torch
from torchvision import datasets, transforms, models
from collections import OrderedDict
from utility import get_input_args, do_deep_learning
from torch import nn
from torch import optim

def main():
    # Get parameters from command line.
    in_arg = get_input_args()

    # Extract them to variables.
    arch = in_arg.arch
    data_dir = in_arg.data_directory
    epochs = in_arg.epochs
    gpu = in_arg.gpu
    hidden_units = in_arg.hidden_units
    learn_rate = in_arg.learning_rate
    save_dir = in_arg.save_dir

    # Set hyper variables.
    device = torch.device("cuda:0" if torch.cuda.is_available() and gpu else "cpu")
    train_dir = data_dir + '/train'
    test_dir = data_dir + '/test'
    valid_dir = data_dir + '/valid/'
    batch = 64
    print_every = 20
    output_size = 102

    # Transform images.
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Load the data in different datasets
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=data_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=data_transforms)

    # Create the data loaders.
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=batch)

    # Select the NN depending on the arch variable.
    if arch == 'vgg':
        model = models.vgg16(pretrained=True)
        input_size = model.classifier[0].in_features
    elif arch == 'densenet':
        model = models.densenet121(pretrained=True)
        input_size = model.classifier.in_features
    else:
        print(f"Unsupported architecture: {arch}")
        return

    for param in model.parameters():
        param.requires_grad = False

    # Set classifier structure.
    fc1_in = input_size
    fc1_out = input_size // 2
    fc2_in = fc1_out
    fc2_out = fc2_in // 2
    fc3_in = fc2_out
    fc3_out = output_size

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(fc1_in, fc1_out)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(p=0.5)),
        ('fc2', nn.Linear(fc2_in, fc2_out)),
        ('relu2', nn.ReLU()),
        ('fc3', nn.Linear(fc3_in, fc3_out)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    
    # Replace the model classifier with our model.
    model.classifier = classifier

    # Set the criterion and the optimizer.
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)

    # Do deep learning!
    do_deep_learning(model, trainloader, testloader, epochs, print_every, criterion, optimizer, device)

    # Save the checkpoint!
    checkpoint = {
        'epochs': epochs,
        'input_size': input_size,
        'output_size': output_size,
        'learn_rate': learn_rate,
        'batch_size': batch,
        'data_transforms': data_transforms,
        'model': model,
        'state_dict': model.state_dict(),
        'optimizer_dict': optimizer.state_dict(),
        'class_to_idx': train_data.class_to_idx,
        'classifier': model.classifier,
    }
    torch.save(checkpoint, 'checkpoint.pth')
    # python train.py ./flowers/ --save_dir ./checkpoint.pth --arch densenet --learning_rate 
    # 0.001 --hidden_units 1664 --epochs 1 --gpu

if __name__ == '__main__':
    main()

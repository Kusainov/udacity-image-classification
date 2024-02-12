# Flower Species Image Classifier

This project is part of the Udacity Nanodegree in AI Programming with Python. The goal of this project is to train an image classifier to recognize different species of flowers. The trained model can be used in a variety of applications, such as a smartphone app that identifies flowers using the device's camera.

## Overview

The project is divided into multiple steps:

1. Load and preprocess the image dataset.
2. Train the image classifier on the dataset.
3. Use the trained classifier to predict image content.

## Dependencies

This project requires the following dependencies:
- Python 3.x
- PyTorch
- torchvision
- NumPy
- Matplotlib
- PIL

You can install the required dependencies using pip:

```bash
pip install torch torchvision numpy matplotlib pillow
```

## Usage

### Training the Classifier

1. Clone this repository:

```bash
git clone https://github.com/Viswesh934/udacity-image-classifier.git
cd udacity-image-classifier
```

2. Download the flower dataset. You can use the provided data or download your own dataset.

3. Update the `data_dir` variable in the `image_classifier.py` script to point to the directory containing your dataset.

4. Run the training script:

```bash
python image_classifier.py
```

5. The script will train the classifier and save the trained model as `checkpoint.pth`.

### Making Predictions

1. Load the trained model using the provided checkpoint:

```python
model = load_checkpoint('checkpoint.pth')
```

2. Use the `predict` function to make predictions on new images:

```python
probs, classes = predict(image_path, model, device='cuda' if torch.cuda.is_available() else 'cpu')
print(probs)
print(classes)
```

### Example Usage

For example, to classify an image located at `data/test/image.jpg`:

```python
probs, classes = predict('data/test/image.jpg', model, device='cuda')
print(probs)
print(classes)
```

## Neural Network Architecture

The neural network architecture used for this task is DenseNet169, which is a convolutional neural network architecture known for its efficiency and accuracy in image classification tasks. DenseNet169 is implemented using PyTorch's torchvision library.

## Training the Neural Network

To train the neural network, follow these steps:

1. Define the neural network architecture by creating an instance of the DenseNet169 model provided by torchvision.
2. Set up your data loaders for training and validation.
3. Define your loss function (criterion) and optimizer.
4. Call the `train` function with your model, criterion, optimizer, data loaders, and device (CPU or GPU) as arguments.

Example code for training the neural network:

```python
import torch
import torchvision.models as models
from flower_classifier import train

# Define the DenseNet169 model
model = models.densenet169(pretrained=True)

# Set the number of input features for the classifier layer
num_ftrs = model.classifier.in_features

# Modify the classifier layer to match the number of flower species
model.classifier = torch.nn.Linear(num_ftrs, num_classes)

# Set up your data loaders for training and validation
# Define your loss function and optimizer

# Call the train function
train(model, criterion, optimizer, dataloaders, device, epochs=5)
```
## Sanity checking:

Sanity checking in the context of machine learning involves performing basic tests and checks to ensure that the data, model, and training process are set up correctly before proceeding with more extensive training. This can include verifying data loading pipelines, inspecting model architectures, and validating the behavior of the model with sample inputs.

![image](https://github.com/Viswesh934/udacity-image-classifier/assets/98519767/54a0fd2f-5e56-448c-bfb9-0b2bade6fa7c)

## Classification:

Classification is a task in machine learning where the goal is to assign predefined labels or categories to input data based on its features. In image classification, for example, the input data consists of images, and the task is to assign each image to one or more predefined classes or categories. Classification models are trained on labeled datasets and learn to map input features to the corresponding class labels during training.

![image](https://github.com/Viswesh934/udacity-image-classifier/assets/98519767/bd844ade-6929-4736-8511-ea14a8b03529)

### Command Line Application

The command line application consists of two scripts:

1. **train.py**: This script is used to train a neural network model on image data. It accepts command-line arguments such as the directory containing the image data, the architecture of the neural network, number of training epochs, learning rate, GPU usage, number of hidden units, and save directory. After parsing the arguments, the script loads the image data, selects the appropriate neural network architecture, sets up the classifier, trains the model, and saves the trained model as a checkpoint file.

   Example usage:
   ```bash
   python train.py ./flowers/ --save_dir ./checkpoint.pth --arch densenet --learning_rate  0.001 --hidden_units 1664 --epochs 1 --gpu

  2. **predict.py**: This script is used to predict the class labels of images using a trained neural network model. It accepts command-line arguments such as the path to the image file, the path to the checkpoint file containing the trained model, the number of top classes to return, and whether to use GPU for inference. The script loads the trained model from the checkpoint file, processes the input image, performs inference using the model, and displays the predicted class labels along with their probabilities.

   Example usage:
   ```bash
   python predict.py ./flowers/test/1/image_06743.jpg ./checkpoint.pth --top_k 5 --category_names cat_to_name.json --gpu 
```
## File structure:
- **assets**: Directory possibly containing additional project assets such as images, diagrams, or documentation.
- **flowers**: Directory containing data related to flowers, likely used for training and testing the image classifier.
- **Image Classifier Project.ipynb**: Jupyter Notebook file containing the project code, possibly for development and experimentation.
- **LICENSE**: File containing license information for the project.
- **README.md**: Markdown file providing an overview of the project, possibly including installation instructions, usage guidelines, and other relevant information.
- **cat_to_name.json**: JSON file containing mappings from category labels to category names, potentially used for labeling images in the classifier.
- **checkpoint.pth**: File containing a model checkpoint or saved model state, likely used for checkpointing during training or for model persistence.
- **predict.py**: Python script for making predictions using the trained image classifier model.
- **train.py**: Python script for training the image classifier model.
- **utility.py**: Python script containing utility functions used across the project, possibly including functions for data preprocessing, model evaluation, etc.

## License:
This project is licensed under the MIT License - see the LICENSE file for details.  

- Acknowledgement:
@Kusainov for providing the file structure
@udacity for providing the project review

- Author: Sigireddy Viswesh
- Mail: sigireddyviswesh@gmail.com



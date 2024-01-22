import argparse
import torch
import json
from utility import load_checkpoint, predict, view_classify

def main():
    # Get parameters from command line.
    in_arg = get_input_args()

    # Extract them to variables.
    path_image = in_arg.input_img
    checkpoint = in_arg.checkpoint
    top_k = in_arg.top_k
    cat_to_name = in_arg.category_names
    gpu = in_arg.gpu
    
    # Set hyper variables.
    device = torch.device("cuda:0" if torch.cuda.is_available() and gpu else "cpu")

    # Load the model from directory.
    
    model = load_checkpoint(checkpoint)
    

    # Predict the image's category.
    probs, classes = predict(path_image, model, device, int(top_k))
    
    # Print results.
    view_classify(path_image,probs, classes, cat_to_name)

# Get parameters from command line.
def get_input_args():
    """
    Get parameters from command line.
    Returns:
        parse_args() - data structure
    """
    parser = argparse.ArgumentParser(description='Predicting image with DenseNet')

    parser.add_argument(
        'input_img',
        default='/workspace/home/ImageClassifier/flowers/test/3/image_06641.jpg',
        nargs='*',
        action="store",
        type=str,
        help='path to a single image file'
    )
    parser.add_argument(
        'checkpoint',
        default='checkpoint.pth',
        nargs='*',
        action="store",
        type=str,
        help='path to the checkpoint file'
    )
    parser.add_argument(
        '--top_k',
        default=5,
        dest='top_k',
        action="store",
        type=int,
        help='top k most likely classes'
    )
    parser.add_argument(
        '--category_names',
        dest='category_names',
        action="store",
        default='cat_to_name.json',
        help='path to the JSON file containing category names'
    )
    parser.add_argument(
        '--gpu',
        default="gpu",
        action="store",
        dest="gpu",
        help='use the GPU for training'
    )

    return parser.parse_args()

if __name__ == '__main__':
    main()

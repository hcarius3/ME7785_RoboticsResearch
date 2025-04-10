#!/usr/bin/env python3
import os
import argparse
import csv
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Settings
NUM_CLASSES = 6
IMG_SIZE = 224

# Values for normalization created seperately
mean = [0.66400695, 0.45201, 0.4441439]
std = [0.13950367, 0.15291268, 0.14623028]

# test without data augmentation
eval_transform = transforms.Compose([   
    transforms.Resize(IMG_SIZE), # Resize to models needs
    transforms.CenterCrop(IMG_SIZE), # shouldn't do anything
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
# ------------------------------------------------------------------------------
#                  DO NOT MODIFY FUNCTION NAMES OR ARGUMENTS
# ------------------------------------------------------------------------------

def initialize_model(model_path=None):
    """
    Initialize and return your trained model.
    You MUST modify this function to load and/or construct your model.
    DO NOT change the function name or its input/output.
    
    Args:
        model_path: The path to your pretrained model file (if one is needed).
    Returns:
        model: Your trained model.
    """

    # Load the pre-trained ResNet18 model with default weights
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Modify the final fully connected layer for our task (NUM_CLASSES = 6)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    # If a model path is provided, load the weights
    if model_path:
        print(f"Loading model weights from {model_path}")
        model.load_state_dict(torch.load(model_path))

    # Set the model to evaluation mode (important for inference)
    model.eval()
    
    return model


def predict(model, image):
    """
    Run inference on a single image using your model.
    You MUST modify this function to perform prediction.
    DO NOT change the function signature.
    
    Args:
        model: The model object returned by initialize_model().
        image: The input image (as a NumPy array) to classify.
    
    Returns:
        int: The predicted class label.
    """

    # Convert the image (numpy array) to a PIL image
    image = Image.fromarray(image)

    # Preprocess the image (resize, normalize, etc.)
    image = eval_transform(image)  # Use the transformation defined for evaluation
    image = image.unsqueeze(0)  # Add a batch dimension (since the model expects a batch of images)

    # Check device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Move the model and image to the correct device
    model = model.to(device)
    image = image.to(device)

    # Make prediction
    with torch.no_grad():  # Disable gradient calculation (since we're not training)
        outputs = model(image)  # Forward pass
        _, predicted = torch.max(outputs, 1)  # Get the class with the highest score

    # Return the predicted class label as an integer
    return predicted.item()


# ------------------------------------------------------------------------------
#                      DO NOT MODIFY ANY CODE BELOW THIS LINE
# ------------------------------------------------------------------------------

def load_validation_data(data_path):
    """
    Load validation images and labels from the given directory.
    Expects a 'labels.txt' file in the directory and images in .png format.
    
    Args:
        data_path (str): Path to the validation dataset.
    
    Returns:
        list of tuples: Each tuple contains (image_path, true_label)
    """
    labels_file = os.path.join(data_path, "labels.txt")
    data = []
    with open(labels_file, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            # Assumes row[0] is the image filename (without extension) and row[1] is the label.
            image_file = os.path.join(data_path, row[0] + ".png")  # Modify if images use a different extension.
            data.append((image_file, int(row[1])))
    return data

def evaluate_model(model, validation_data):
    """
    Evaluate the model on the validation dataset.
    Computes and prints the confusion matrix and overall accuracy.
    
    Args:
        model: The model object.
        validation_data (list): List of tuples (image_path, true_label).
    """
    num_classes = 6  # Number of classes (adjust if needed)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    correct = 0
    total = len(validation_data)
    
    for image_path, true_label in validation_data:
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print("Warning: Could not load image:", image_path)
            continue
        # Get the predicted label using the student's implementation.
        predicted_label = predict(model, image)
        
        if predicted_label == true_label:
            correct += 1
        confusion_matrix[true_label][predicted_label] += 1
        print(f"Image: {os.path.basename(image_path)} - True: {true_label}, Predicted: {predicted_label}")
    
    accuracy = correct / total if total > 0 else 0
    print("\nTotal accuracy:", accuracy)
    print("Confusion Matrix:")
    print(confusion_matrix)

def main():
    parser = argparse.ArgumentParser(description="Model Grader for Lab 6")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the validation dataset directory (must contain labels.txt and images)")
    parser.add_argument("--model_path", type=str, required=False,
                        help="Path to the trained model file (if applicable)")
    args = parser.parse_args()
    
    # Path to the validation dataset directory from command line argument.
    VALIDATION_DATASET_PATH = args.data_path

    # Path to the trained model file from command line argument.
    MODEL_PATH = args.model_path
    
    # Load validation data.
    validation_data = load_validation_data(VALIDATION_DATASET_PATH)
    
    # Initialize the model using the student's implementation.
    model = initialize_model(MODEL_PATH) if MODEL_PATH else initialize_model()
    
    # Evaluate the model on the validation dataset.
    evaluate_model(model, validation_data)

if __name__ == "__main__":
    main()

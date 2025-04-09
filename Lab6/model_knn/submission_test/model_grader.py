# ECE 7785 Intro to Robotics Research
# Hendrik Carius and Daniel Terrell

#!/usr/bin/env python3
import cv2
import argparse
import csv
import math
import pickle
import numpy as np
import random
import os
from skimage.feature import hog
from skimage import data, exposure
from rembg import remove
from rembg.session_factory import new_session
import io
from PIL import Image
import albumentations as A
from albumentations.core.composition import OneOf
from albumentations.augmentations.transforms import RandomBrightnessContrast
from albumentations.augmentations.geometric.transforms import Affine

# ------------------------------------------------------------------------------
#                  DO NOT MODIFY FUNCTION NAMES OR ARGUMENTS
# ------------------------------------------------------------------------------

x_image = 100
y_image = 100

lightweight_model_session = new_session("u2netp")

def preprocess_image(image, save_masks=False, mask_prefix="mask"):
    """Applies background removal, crops the image around the object, and resizes it."""
    # Convert the image to a PIL Image for rembg
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Convert the PIL Image to bytes
    image_bytes = io.BytesIO()
    image_pil.save(image_bytes, format="PNG")
    image_bytes = image_bytes.getvalue()

    # Remove the background using rembg with the lightweight model
    output = remove(image_bytes, session=lightweight_model_session)
    image_no_bg = Image.open(io.BytesIO(output)).convert("RGB")

    # Convert the image back to OpenCV format
    image_no_bg = cv2.cvtColor(np.array(image_no_bg), cv2.COLOR_RGB2BGR)

    # Convert the image to grayscale to find contours
    gray = cv2.cvtColor(image_no_bg, cv2.COLOR_BGR2GRAY)

    # Find contours of the non-zero regions (object)
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Find the bounding box of the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Crop the image around the bounding box
        cropped_image = image_no_bg[y:y+h, x:x+w]
    else:
        # If no contours are found, use the entire image
        cropped_image = image_no_bg

    # Resize the cropped image to the standard dimensions
    resized_image = cv2.resize(cropped_image, (x_image, y_image))

    # Save the cropped and resized image for debugging if requested
    if save_masks:
        cv2.imwrite(f"{mask_prefix}_cropped.png", cropped_image)
        cv2.imwrite(f"{mask_prefix}_resized.png", resized_image)

    return resized_image

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

    # TODO: Load your trained model here.
    # For example, if you saved your model using pickle or a deep learning framework,
    # load it and return the model object.

    if model_path is None:
        raise NotImplementedError("initialize_model() is not implemented. Please implement this function.")

    model = cv2.ml.KNearest_load(model_path)

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

    # TODO: Implement your model's prediction logic here.
    # The function should return an integer corresponding to the predicted class.
    
    image = preprocess_image(image)

    image = image.flatten().astype(np.float32).reshape(1, -1)

    # Find nearest with k=3
    retval, results, neigh_resp, dists = model.findNearest(image, k=3)
    return int(results[0][0])

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

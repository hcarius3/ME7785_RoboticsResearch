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

x_image = 100
y_image = 100

def check_split_value_range(val):
    try:
        float_val = float(val)
        if float_val < 0 or float_val > 1:
            raise argparse.ArgumentTypeError("Received data split ratio of %s which is an invalid value. The input ratio must be in range [0, 1]!" % float_val)
        return float_val
    except ValueError:
        raise argparse.ArgumentTypeError(f"Received '{val}' which is not a valid float!")

def check_k_value(val):
    try:
        int_val = int(val)
        if float(val) != int_val:
            raise argparse.ArgumentTypeError(f"Received '{val}' which is a float not an integer. The KNN value input must be an integer!")
        if int_val % 2 == 0 or int_val < 1:
            raise argparse.ArgumentTypeError(f"Received '{val}' which not a positive, odd integer. The KNN value input must be a postive, odd integer!")
        return int_val
    except ValueError:
        raise argparse.ArgumentTypeError(f"Received '{val}' which is not a valid integer!")

def load_and_split_data(data_paths, split_ratio):
    """
    Uses the provided labels.txt file to split the data into training and testing sets.

    Args:
        data_path (str): Path to the dataset.
        split_ratio (float): must be a float between 0 and 1. Split ratio will be used to split the data into training and testing sets. 
                             split_ratio of the data will be used for training and (1-split_ratio) will be used for testing. 
                             For example if split ratio was 0.7, 70% of the data will be used for training and the remaining 30% will be used for testing.

    Returns:
        list of tuples for testing and training (image_path, true_label)
    """

    # Get all data paths
    all_lines = []

    for data_path in data_paths:
        print(data_path)
        with open(data_path + '/labels.txt', 'r') as f:
            reader = csv.reader(f)
            lines = list(reader)
            all_lines.extend(lines)

    # Randomly shuffle and split the combined data
    random.shuffle(all_lines)
    train_lines = all_lines[:math.floor(len(all_lines) * split_ratio)]
    test_lines = all_lines[math.floor(len(all_lines) * split_ratio):]

    return train_lines, test_lines

# Create a session with the lightweight model
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

def train_model(data_path, train_lines, image_type, model_filename, save_model, augment=False):
    """
    Loads the images from the training set and uses them to create a KNN model.
    Optionally augments the dataset using Albumentations.

    Args:
        data_path (str): Path to the dataset.
        train_lines (tuple): Tuple of the training data containing (image_number, true_label)
        image_type (str): Image extension to load (e.g. .png, .jpg, .jpeg)
        model_filename (str): Filename to save the trained model.
        save_model (bool): Whether to save the trained model.
        augment (bool): Whether to augment the dataset using Albumentations.

    Returns:
        knn (knn_model_object): The KNN model.
    """

    # Define Albumentations augmentations
    augmentation_pipeline = A.Compose([
        A.Affine(rotate=(-20, 20), p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
    ])

    # Load and preprocess images
    train_images = []
    train_labels = []

    for i in range(len(train_lines)):
        img_path = data_path + train_lines[i][0] + image_type
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Unable to load image {img_path}")
            continue

        # Preprocess the image (e.g., background removal, resizing)
        img_preprocessed = preprocess_image(img, save_masks=True)

        # Add the original image to the dataset
        train_images.append(img_preprocessed)
        train_labels.append(np.int32(train_lines[i][1]))

        # Apply augmentations if enabled
        if augment:
            for _ in range(5):  # Generate 5 augmented versions of each image
                augmented = augmentation_pipeline(image=img_preprocessed)
                augmented_image = augmented["image"]
                train_images.append(augmented_image)
                train_labels.append(np.int32(train_lines[i][1]))

    # Convert to numpy arrays
    train_images = np.array(train_images)
    train_data = train_images.reshape(len(train_images), x_image * y_image * 3).astype(np.float32)
    train_labels = np.array(train_labels)

    # Train the KNN model
    knn = cv2.ml.KNearest_create()
    knn.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)

    print("KNN model created!")

    if save_model:
        # Save the trained model
        knn.save(model_filename + '.xml')
        print(f"KNN model saved to {model_filename}.xml")

    return knn

def test_model(data_path, test_lines, image_type, knn_model, knn_value, show_img):
    """
    Loads the images and tests the provided KNN model prediction with the dataset label.
    The images and labels must be in the given directory.

    Args:
        data_path (str): Path to the dataset.
        test_lines (tuple): Tuple of the testing data containing (image_number, true_label)
        image_type (str): Image extension to load (e.g. .png, .jpg, .jpeg)
        knn_model (model object): The KNN model
        knn_value (int): The number of KNN neighbors to consider when classifying
        show_img: A boolean whether to show images as they are processed or not

    Returns:
        None
    """

    if show_img:
        Title_images = 'Original Image'
        Title_resized = 'Image Resized'
        cv2.namedWindow(Title_images, cv2.WINDOW_AUTOSIZE)

    correct = 0.0
    num_classes = len(set([int(line[1]) for line in test_lines]))  # Use test_lines here
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

    k = knn_value

    for i in range(len(test_lines)):
        img_path = data_path + test_lines[i][0] + image_type
        original_img = cv2.imread(img_path)
        if original_img is None:
            print(f"Warning: Unable to load image {img_path}")
            continue

        # Preprocess the image (edge detection)
        img_edges = preprocess_image(original_img)

        # Flatten and convert the image to match the training data format
        test_img = img_edges.flatten().reshape(1, x_image * y_image * 3).astype(np.float32)

        if show_img:
            cv2.imshow(Title_images, original_img)
            cv2.imshow(Title_resized, img_edges)
            key = cv2.waitKey()
            if key == 27:  # Esc key to stop
                break

        test_label = np.int32(test_lines[i][1])

        # Perform KNN prediction
        ret, results, neighbours, dist = knn_model.findNearest(test_img, k)

        if test_label == ret:
            print(str(test_lines[i][0]) + " Correct, " + str(ret))
            correct += 1
            confusion_matrix[np.int32(ret)][np.int32(ret)] += 1
        else:
            confusion_matrix[test_label][np.int32(ret)] += 1

            print(str(test_lines[i][0]) + " Wrong, " + str(test_label) + " classified as " + str(ret))
            print("\tneighbours: " + str(neighbours))
            print("\tdistances: " + str(dist))

    print("\n\nTotal accuracy: " + str(correct / len(test_lines)))
    print(confusion_matrix)

def main():
    parser = argparse.ArgumentParser(description="Example Model Trainer and Tester with Basic KNN for 7785 Lab 6!")
    parser.add_argument("-p","--data_path", nargs="+", type=str, required=True, help="Path to the valid dataset directory (must contain labels.txt and images)")
    parser.add_argument("-r","--data_split_ratio", type=check_split_value_range, required=False, default=0.5, help="Ratio of the train, test split. Must be a float between 0 and 1. The number entered is the percentage of data used for training, the remaining is used for testing!")
    parser.add_argument("-k","--knn-value", type=check_k_value, required=False, default=3, help="KNN value. Must be an odd integer greater than zero.")
    parser.add_argument("-i","--image_type", type=str, required=False, default=".png", help="Extension of the image files (e.g. .png, .jpg)")
    parser.add_argument("-s","--save_model_bool", action='store_true', required=False, help="Boolean flag to save the KNN model as an XML file for later use.")
    parser.add_argument("-n","--model_filename", type=str, required=False, default="knn_model", help="Filename of the saved KNN model.")
    parser.add_argument("-t","--dont_test_model_bool", action='store_false', required=False, help="Boolean flag to not test the created KNN model on split testing set (training only).")
    parser.add_argument("-d","--show_img", action='store_true', required=False, help="Boolean flag to show the tested images as they are classified.")
    parser.add_argument("-a","--augment", action='store_true', required=False, help="Boolean flag to augment the dataset by rotating images.")


    args = parser.parse_args()

    #Path to dataset directory from command line argument.
    dataset_path = []
    for path in args.data_path:
        dataset_path.append(os.getcwd() + "/" + path + "/")
    print(dataset_path)
    #dataset_path = args.data_path 

    #Ratio of datasplit from command line argument.
    data_split_ratio = args.data_split_ratio

    #Image type from command line argument.
    image_type = args.image_type

    #Boolean if true will save the KNN model as a XML file from command line argument.
    save_model_bool = args.save_model_bool

    #Filename for the saved KNN model from command line argument.
    model_filename = args.model_filename

    #Boolean if true will test the model on the split testing set based on command line argument.
    test_model_bool = args.dont_test_model_bool

    #Number of neighbors to consider for KNN.
    knn_value = args.knn_value

    #Boolean if true will show the images as they are tested.
    show_img= args.show_img

    #Boolean if true will augment the dataset by rotating images.
    augment = args.augment

    train_lines, test_lines = load_and_split_data(dataset_path, data_split_ratio)
    knn_model = train_model(dataset_path[0], train_lines, image_type, model_filename, save_model_bool, augment)
    if(test_model_bool):
        test_model(dataset_path[0], test_lines, image_type, knn_model, knn_value, show_img)

if __name__ == "__main__":
    main()
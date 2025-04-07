#!/usr/bin/env python3
import cv2
import argparse
import csv
import math
import pickle
import numpy as np
import random
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

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
        # data_path = os.getcwd() + "/" + data_path
        # print(data_path)
        with open(data_path + 'labels.txt', 'r') as f:
            reader = csv.reader(f)
            lines = list(reader)
            all_lines.extend(lines)

    # Randomly shuffle and split the combined data
    random.shuffle(all_lines)
    train_lines = all_lines[:math.floor(len(all_lines) * split_ratio)]
    test_lines = all_lines[math.floor(len(all_lines) * split_ratio):]

    return train_lines, test_lines


def train_model(data_path, train_lines, image_type, model_filename, save_model, augment=False):
    """
    Loads the images from the training set and uses them to create a KNN model.
    Optionally augments the dataset by rotating images, flipping, scaling, and performing random cropping.

    Args:
        data_path (str): Path to the dataset.
        train_lines (tuple): Tuple of the training data containing (image_number, true_label)
        image_type (str): Image extension to load (e.g. .png, .jpg, .jpeg)
        model_filename (str): Filename to save the trained model.
        save_model (bool): Whether to save the trained model.
        augment (bool): Whether to augment the dataset by rotating images, flipping, scaling, and cropping.

    Returns:
        knn (KNeighborsClassifier): The trained KNN model.
    """

    def rotate_image(image, angle):
        """Rotates an image by the given angle."""
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, matrix, (w, h))

    def scale_image(image, scale_range=(0.8, 1.2)):
        """Scales the image by a random factor within the specified range."""
        h, w = image.shape[:2]
        scale_factor = random.uniform(*scale_range)
        new_w, new_h = int(w * scale_factor), int(h * scale_factor)
        scaled_image = cv2.resize(image, (new_w, new_h))
        if scale_factor > 1.0:
            top = (new_h - h) // 2
            left = (new_w - w) // 2
            scaled_image = scaled_image[top:top + h, left:left + w]
        else:
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            scaled_image = cv2.copyMakeBorder(
                scaled_image, pad_h, h - new_h - pad_h, pad_w, w - new_w - pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )
        return cv2.resize(scaled_image, (25, 33))

    def color_jitter(image, brightness=0.2, contrast=0.2, saturation=0.2):
        """Applies random color jittering to the image."""
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

        # Adjust brightness
        brightness_factor = 1 + random.uniform(-brightness, brightness)
        hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2] * brightness_factor, 0, 255)

        # Adjust saturation
        saturation_factor = 1 + random.uniform(-saturation, saturation)
        hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * saturation_factor, 0, 255)

        # Convert back to BGR
        jittered_image = cv2.cvtColor(hsv_image.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # Adjust contrast
        contrast_factor = 1 + random.uniform(-contrast, contrast)
        jittered_image = np.clip((jittered_image - 127.5) * contrast_factor + 127.5, 0, 255).astype(np.uint8)

        return jittered_image

    # Load and preprocess images
    train_images = []
    train_labels = []

    for i in range(len(train_lines)):
        img_path = data_path + train_lines[i][0] + image_type
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Unable to load image {img_path}")
            continue
        img_resized = cv2.resize(img, (25, 33))
        train_images.append(img_resized.flatten())  # Flatten the image
        train_labels.append(np.int32(train_lines[i][1]))

        # Augment data if enabled
        if augment:
            # Rotate the image
            for angle in [15, 25, 35, 45, 55, 65, 75]:
                rotated_img = rotate_image(img_resized, angle)
                train_images.append(rotated_img.flatten())
                train_labels.append(np.int32(train_lines[i][1]))

            # Perform scaling
            for _ in range(3):
                scaled_img = scale_image(img_resized)
                train_images.append(scaled_img.flatten())
                train_labels.append(np.int32(train_lines[i][1]))

            # Perform color jittering
            jittered_img = color_jitter(img_resized)
            train_images.append(jittered_img.flatten())
            train_labels.append(np.int32(train_lines[i][1]))

    # Convert to numpy arrays
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)

    # Train the KNN model using scikit-learn
    knn = KNeighborsClassifier(n_neighbors=6)  # Default k=3
    knn.fit(train_images, train_labels)

    print("KNN model created!")

    if save_model:
        # Save the trained model using pickle
        with open(model_filename + '.pkl', 'wb') as f:
            pickle.dump(knn, f)
        print(f"KNN model saved to {model_filename}.pkl")

    return knn

def test_model(data_path, test_lines, image_type, knn_model, knn_value, show_img):
    """
    Loads the images and tests the provided KNN model prediction with the dataset label.
    The images and labels must be in the given directory.

    Args:
        data_path (str): Path to the dataset.
        test_lines (tuple): Tuple of the testing data containing (image_number, true_label)
        image_type (str): Image extension to load (e.g. .png, .jpg, .jpeg)
        knn_model (KNeighborsClassifier): The trained KNN model
        knn_value (int): The number of KNN neighbors to consider when classifying
        show_img: A boolean whether to show images as they are processed or not

    Returns:
        None
    """

    test_images = []
    test_labels = []

    for i in range(len(test_lines)):
        img_path = data_path + test_lines[i][0] + image_type
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Unable to load image {img_path}")
            continue
        img_resized = cv2.resize(img, (25, 33))
        test_images.append(img_resized.flatten())
        test_labels.append(np.int32(test_lines[i][1]))

        if show_img:
            cv2.imshow("Original Image", img)
            cv2.imshow("Resized Image", img_resized)
            key = cv2.waitKey()
            if key == 27:  # Esc key to stop
                break

    # Convert to numpy arrays
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)

    # Predict using the trained KNN model
    predictions = knn_model.predict(test_images)

    # Calculate accuracy and confusion matrix
    accuracy = accuracy_score(test_labels, predictions)
    conf_matrix = confusion_matrix(test_labels, predictions)

    print("\n\nTotal accuracy: {:.2f}%".format(accuracy * 100))
    print("Confusion Matrix:")
    print(conf_matrix)

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
    with open('test_model' + '.pkl', 'rb') as f:
        knn_model = pickle.load(f)
    if(test_model_bool):
        test_model(dataset_path[0], test_lines, image_type, knn_model, knn_value, show_img)

if __name__ == "__main__":
    main()
import cv2
import numpy as np
import onnxruntime as ort
import time

# Custom normalization params
mean = np.array([0.66400695, 0.45201, 0.4441439])
std = np.array([0.13950367, 0.15291268, 0.14623028])
IMG_SIZE = 224
IMG_PATH = "combined_imgs/421.png"

def preprocess_image(cv_image):
    # Resize to (224, 224) as expected by ResNet
    img_resized = cv2.resize(cv_image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

    # Convert BGR (OpenCV) to RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    # Convert to float32 and normalize to [0, 1]
    img = img_rgb.astype(np.float32) / 255.0

    # Normalize using mean/std
    img = (img - mean) / std

    # Change shape to CHW (3, 224, 224)
    img = np.transpose(img, (2, 0, 1))

    # Add batch dimension and convert to np.float32 (needed by ONNX)
    input_tensor = np.expand_dims(img, axis=0).astype(np.float32)

    return input_tensor

# Load model
session = ort.InferenceSession("resnet18_trained.onnx")

# Start timing
start_time = time.time()

# Load and preprocess image
img = cv2.imread(IMG_PATH)
input_tensor = preprocess_image(img)

# Run inference
outputs = session.run(None, {"input": input_tensor})
pred = np.argmax(outputs[0])

# End timing
end_time = time.time()
elapsed = (end_time - start_time) * 1000  # convert to ms

print(f"Predicted class: {pred}")
print(f"Inference time: {elapsed:.2f} ms")

#!/usr/bin/env python

import numpy as np
import cv2

# Start video capture
cap = cv2.VideoCapture(0)

# Define the green color range in HSV
# Adjust these values to widen/narrow the margin
lower_green = np.array([35, 50, 50])  # Lower bound of green in HSV
upper_green = np.array([85, 255, 255])  # Upper bound of green in HSV

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Convert the frame to HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask for green color
    mask = cv2.inRange(hsv_frame, lower_green, upper_green)

    # Bitwise-AND mask with the original frame to isolate green areas
    green_areas = cv2.bitwise_and(frame, frame, mask=mask)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours around detected green areas
    for contour in contours:
        # Optional: Filter small contours based on area
        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the original frame with contours
    cv2.imshow('Original Frame with Green Tracking', frame)

    # Display the mask showing only green areas
    cv2.imshow('Green Areas Mask', green_areas)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()

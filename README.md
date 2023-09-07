# CURSOR
import cv2
import numpy as np
import pyautogui

hand_contour = ...

roi = ...

# Set the region of interest (ROI) for hand detection
top, right, bottom, left = 10, 350, 225, 590

# Set the resolution for cursor movement
screen_width, screen_height = pyautogui.size()
mov_resolution = (screen_width, screen_height)

# Initialize the previous centroid position
prev_centroid = None

# Set the sensitivity factor for cursor movement
sensitivity = 2.5

# Start the video capture
cap = cv2.VideoCapture(0)

while True:
    # Read the video frame
    ret, frame = cap.read()

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Extract the region of interest (ROI) for hand detection
    roi = frame[top:bottom, right:left]

    # Convert the ROI to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # Perform thresholding to segment the hand region
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the maximum area (hand)
    if len(contours) > 0:
        hand_contour = max(contours, key=cv2.contourArea)

        # Calculate the centroid of the hand contour
        M = cv2.moments(hand_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroid = (cx, cy)

            # Move the cursor based on the centroid motion
            if prev_centroid is not None:
                dx = int((cx - prev_centroid[0]) * sensitivity)
                dy = int((cy - prev_centroid[1]) * sensitivity)
                pyautogui.moveRel(dx, dy)

            prev_centroid = centroid

    # Draw the hand contour and centroid on the frame
    cv2.drawContours(roi, [hand_contour], 0, (0, 255, 0), 2)
    cv2.circle(roi, centroid, 5, (0, 0, 255), -1)

    # Display the frame
    cv2.imshow("Hand Motion", frame)

    # Check for keypress to exit
    if cv2.waitKey(1) == ord("q"):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()

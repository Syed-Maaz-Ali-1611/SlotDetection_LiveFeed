

import cv2
import numpy as np

def detect_boxes(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Applying a Gaussian Blur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edged = cv2.Canny(blurred, 30, 150)


    # Find contours in the edged image
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # List to hold the status of each box
    box_statuses = []

    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Filter to ensure the contour has four sides and a reasonable area
        if len(approx) == 4 and cv2.contourArea(contour) > 1000:
            x, y, w, h = cv2.boundingRect(approx)

            # Define the ROI for the box
            roi = frame[y:y+h, x:x+w]
            # Convert ROI to grayscale
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # Threshold to separate object from background
            _, roi_thresh = cv2.threshold(roi_gray, 150, 255, cv2.THRESH_BINARY)

            # Calculate the ratio of white to total area to determine if it's empty
            white_area = np.sum(roi_thresh == 255)
            total_area = roi_thresh.size

            if white_area / total_area > 0.5:  # If more than 50% of the area is white, consider it empty
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "booked", (x + 10, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                box_statuses.append('booked')
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "empty", (x + 10, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                box_statuses.append('empty')
                
                

    # Return frame and the status of each box
    return frame, box_statuses

# Open the camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        frame, statuses = detect_boxes(frame)
        print(statuses)  # Print statuses to see what's happening with each box
        cv2.imshow('Camera Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()


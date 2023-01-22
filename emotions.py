import cv2
import numpy as np

# Load the cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the cascade for eyes detection
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Load the cascade for smile detection
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the frames from the webcam
    ret, img = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        # Detect eyes in the face
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            # Draw a rectangle around the eyes
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        # Detect smile in the face
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor= 1.7, minNeighbors=22, minSize=(25, 25))
        for (sx, sy, sw, sh) in smiles:
            # Draw a rectangle around the smile
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Emotion Recognition', img)

    # Exit the webcam feed when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam
cap.release()

# Close all the windows
cv2.destroyAllWindows()

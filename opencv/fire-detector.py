import numpy as np
import cv2
import time

fire_cascade = cv2.CascadeClassifier('cascade.xml')  # Load fire detection classifier

cap = cv2.VideoCapture(0)  # Open the front camera
count = 0

while cap.isOpened():
    ret, img = cap.read()  # Capture a frame
    if not ret:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
    fire = fire_cascade.detectMultiScale(img, 12, 5)  # Fire detection

    for (x, y, w, h) in fire:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Highlight fire detection area
        print('Fire detected..!' + str(count))
        count += 1
        time.sleep(0.2)  # Wait for stability

    cv2.imshow('Fire Detection', img)

    k = cv2.waitKey(100) & 0xFF
    if k == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()

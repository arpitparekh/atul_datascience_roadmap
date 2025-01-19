import cv2
import numpy as np

# color detection

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    while not success:
        break

    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


    # blue color
    # lower_blue = np.array([90, 50, 50])
    # upper_blue = np.array([130, 255, 255])

    # green color
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])

    mask = cv2.inRange(img, lower_green, upper_green)

    result = cv2.bitwise_and(img, img, mask=mask)

    # cv2.imshow("Capturing Video", img)
    cv2.imshow("Capturing Video", result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

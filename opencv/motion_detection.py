# motion detection using cv2

import cv2

cap = cv2.VideoCapture(0)
prev_success,prev_frame = cap.read()
prev_frame = cv2.cvtColor(prev_frame,cv2.COLOR_BGR2GRAY)  # convert into gray

while True:
  next_success,next_frame = cap.read()
  next_frame = cv2.cvtColor(next_frame,cv2.COLOR_BGR2GRAY)  # convert into gray

  diff = cv2.absdiff(prev_frame,next_frame)

  # threshold = cv2.threshold(diff,30,255,cv2.THRESH_BINARY)

  cv2.imshow("Difference",diff)
  # cv2.imshow("Threshold",threshold)

  # cv2.imshow("Motion Detection",threshold)

  prev_frame = next_frame

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()




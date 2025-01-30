# face_detection haar cascade
import cv2
import numpy

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
  success,img = cap.read()
  while not success:
    break

  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

  faces = face_cascade.detectMultiScale(gray,1.1,4)

  for (x,y,w,h) in faces:
    cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.imshow("Capturing Video",gray)


  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()

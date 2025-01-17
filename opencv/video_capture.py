import cv2  # cpp library

cap = cv2.VideoCapture(0)  # 30fps

while True:

  success,img=cap.read()
  while not success:
    break

  img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  # convert image
  cv2.imshow("Capturing Video",img)


  if cv2.waitKey(1) & 0xFF == ord('q'):
    break


cap.release()
cv2.destroyAllWindows()

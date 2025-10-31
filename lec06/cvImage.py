import cv2

cam = cv2.VideoCapture(0)

ret, image = cam.read()
cv2.imshow('preview',image)
cv2.waitKey(0)
cv2.imwrite('/home/pi/cvimage.jpg', image)
cam.release()
cv2.destroyAllWindows()

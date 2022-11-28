import cv2

img = cv2.imread("lena.jpg")
cv2.imshow("Normal", img)
cv2.waitKey(0)
face = img[90:240, 125:225]
cv2.imshow("Face", face)
cv2.waitKey(0)
body = img[20:, 40:240]
cv2.imshow("Body", body)
cv2.waitKey(0)
cv2.destroyAllWindows()

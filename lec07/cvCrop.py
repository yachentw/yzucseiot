import cv2

img = cv2.imread("cat.jpg")
cv2.imshow("Normal", img)
cv2.waitKey(0)
body = img[53:, 89:]
cv2.imshow("Body", body)
cv2.waitKey(0)
face = img[68:233, 97:231]
cv2.imshow("Face", face)
cv2.waitKey(0)
cv2.destroyAllWindows()

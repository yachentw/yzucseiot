import cv2

img = cv2.imread("lena.jpg")
rows, cols = img.shape[:2]
M = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 1)
rotation = cv2.warpAffine(img, M, (cols, rows))
cv2.imshow('Rotation', rotation)
cv2.waitKey(0)
cv2.destroyAllWindows()

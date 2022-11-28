import cv2

img = cv2.imread("lena.jpg")
rows, cols = img.shape[:2]
resize = cv2.resize(img, (2*rows, 2*cols), interpolation = cv2.INTER_CUBIC)
cv2.imshow('Resize', resize)
cv2.waitKey(0)
cv2.destroyAllWindows()

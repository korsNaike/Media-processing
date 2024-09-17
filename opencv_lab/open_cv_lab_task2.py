import cv2

img_first = cv2.imread('files/1.png', flags=cv2.IMREAD_COLOR)
WINDOW_NAME = 'image test'

cv2.namedWindow(WINDOW_NAME, flags=cv2.WINDOW_GUI_NORMAL)
cv2.imshow(WINDOW_NAME, img_first)
cv2.waitKey(0)
cv2.destroyAllWindows()

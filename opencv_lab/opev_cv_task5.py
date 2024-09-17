import cv2

# Загрузка изображения
image = cv2.imread('files/obloga.jpg')

# Конвертация изображения в формат HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Создание окон с размером 1000x600
cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Original Image', 600, 600)
cv2.namedWindow('HSV Image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('HSV Image', 600, 600)

# Отображение изображений
cv2.imshow('Original Image', image)
cv2.imshow('HSV Image', hsv_image)

# Ожидание закрытия окон
cv2.waitKey(0)
cv2.destroyAllWindows()

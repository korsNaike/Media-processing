import numpy as np
import cv2

# Инициализация камеры (0 - это индекс устройства, обычно 0 для встроенной камеры)
cap = cv2.VideoCapture(0)

while True:
    # Считываем кадр с камеры
    ret, frame = cap.read()

    if not ret:
        print("Не удалось захватить кадр")
        break

    # Конвертируем изображение из BGR в HSV
    hsv_frame= cv2.cvtColor(frame, cv2.COLOR_BGR2HSV_FULL)

    # Показываем оригинальное и HSV изображение
    cv2.imshow('Original Image', frame)
    cv2.imshow('HSV Image', hsv_frame)

    # Выход при нажатии клавиши ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()

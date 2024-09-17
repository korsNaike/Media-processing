import cv2
import numpy as np

# Инициализация камеры
cap = cv2.VideoCapture(0)

while True:
    # Чтение кадра с камеры
    ret, frame = cap.read()
    if not ret:
        print("Не удалось захватить изображение с камеры.")
        break

    # Применение размытия к кадру
    blurred_frame = cv2.GaussianBlur(frame, (15, 15), 0)

    # Получение размеров кадра
    height, width, _ = frame.shape

    # Определение координат центра
    center_x, center_y = width // 2, height // 2

    # Установка размеров креста
    cross_width = 150
    cross_height = 20

    # Рисуем пустые прямоугольники (крест)
    # Горизонтальная линия
    cv2.rectangle(blurred_frame,
                  (center_x - cross_width//2, center_y - cross_height//2),
                  (center_x + cross_width//2, center_y + cross_height//2),
                  (0, 0, 255),  # Красный цвет
                  2)  # Толщина линии

    # Вертикальная линия
    cv2.rectangle(blurred_frame,
                  (center_x - cross_height//2, center_y - cross_width//2),
                  (center_x + cross_height//2, center_y + cross_width//2),
                  (0, 0, 255),  # Красный цвет
                  2)  # Толщина линии

    # Отображение результата
    cv2.imshow("Красный крест", blurred_frame)

    # Выход при нажатии клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()

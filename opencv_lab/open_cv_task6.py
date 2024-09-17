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

    # Получение размеров кадра
    height, width, _ = frame.shape

    # Определение координат центра
    center_x, center_y = width // 2, height // 2

    # Установка размеров креста
    cross_width = 150
    cross_height = 20

    # Создание маски для креста
    mask = np.zeros((height, width), dtype=np.uint8)

    # Рисуем пустые прямоугольники (кросс) в маске
    cv2.rectangle(mask,
                  (center_x - cross_width//2, center_y - cross_height//2),
                  (center_x + cross_width//2, center_y + cross_height//2),
                  (255),  # Белый цвет для области креста
                  -1)  # Заполнить
    cv2.rectangle(mask,
                  (center_x - cross_height//2, center_y - cross_width//2),
                  (center_x + cross_height//2, center_y + cross_width//2),
                  (255),  # Белый цвет для области креста,
                  -1)  # Заполнить

    # Применение размытия к кадру
    blurred_frame = cv2.GaussianBlur(frame, (15, 15), 0)

    # Инвертируем маску, чтобы размыть всё, кроме креста
    inverted_mask = cv2.bitwise_not(mask)

    # Применяем маску к размытости и оригинальному изображению
    blurred_background = cv2.bitwise_and(blurred_frame, blurred_frame, mask=mask)
    foreground = cv2.bitwise_and(frame, frame, mask=inverted_mask)

    # Складываем размытое фоновое изображение и крест
    result = cv2.add(blurred_background, foreground)

    # Рисуем красные границы на кресте
    cv2.rectangle(result,
                  (center_x - cross_width // 2, center_y - cross_height // 2),
                  (center_x + cross_width // 2, center_y + cross_height // 2),
                  (0, 0, 255),  # Красный цвет
                  2)  # Толщина линии
    cv2.rectangle(result,
                  (center_x - cross_height // 2, center_y - cross_width // 2),
                  (center_x + cross_height // 2, center_y + cross_width // 2),
                  (0, 0, 255),  # Красный цвет
                  2)  # Толщина линии

    # Отображение результата
    cv2.imshow("Red Cross", result)

    # Выход при нажатии клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()

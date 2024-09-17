import cv2


def get_dominant_color(pixel):
    """Определяет преобладающий цвет по значению пикселя."""
    b, g, r = pixel
    if r > g and r > b:
        return (0, 0, 255)  # Красный
    elif g > r and g > b:
        return (0, 255, 0)  # Зеленый
    else:
        return (255, 0, 0)  # Синий


def draw_cross(frame, center, color):
    """Рисует крест на кадре в заданном месте."""
    center_x, center_y = center
    # Горизонтальная линия
    cv2.rectangle(frame, (center_x - 100, center_y - 10), (center_x + 100, center_y + 10), color, -1)
    # Вертикальная линия
    cv2.rectangle(frame, (center_x - 10, center_y - 100), (center_x + 10, center_y + 100), color, -1)


def main():
    # Открытие камеры
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Получение размеров кадра
        height, width, _ = frame.shape
        # Вычисление координатов центра кадра
        center = (width // 2, height // 2)
        # Получение значения центрального пикселя
        center_pixel = frame[center[1], center[0]]
        # Определение преобладающего цвета центрального пикселя
        dominant_color = get_dominant_color(center_pixel)
        # Рисование креста в центре кадра
        draw_cross(frame, center, dominant_color)

        cv2.imshow('Camera', frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Выход при нажатии ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

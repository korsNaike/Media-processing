import cv2
import numpy as np


def initialize_camera():
    """Инициализация камеры."""
    return cv2.VideoCapture(0)


def get_frame_dimensions(cap):
    """Получение ширины и высоты кадра."""
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return frame_width, frame_height


def create_rectangles():
    """Создание прямоугольных областей."""
    return np.array([
        [[25, 150], [235, 170]],  # Прямоугольник 1
        [[115, 0], [145, 150]],  # Прямоугольник 2
        [[115, 170], [145, 320]]  # Прямоугольник 3
    ])


def calculate_offset(frame_width, frame_height, rectangles):
    """Вычислить смещение для прямоугольников."""
    offset_x = frame_width // 2 - rectangles[:, :, 0].max() // 2
    offset_y = frame_height // 2 - rectangles[:, :, 1].max() // 2
    return offset_x, offset_y


def apply_blur(frame, rectangles, offset_x, offset_y, frame_height, frame_width):
    """Применение размытия в области прямоугольника."""
    x1, y1 = rectangles[0][0]
    x2, y2 = rectangles[0][1]
    mask = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    mask = cv2.rectangle(mask, (x1 + offset_x, y1 + offset_y), (x2 + offset_x, y2 + offset_y), (255, 255, 255), -1)
    blurred_frame = cv2.stackBlur(frame, (63, 63))  # Размытие
    frame[mask == 255] = blurred_frame[mask == 255]


def draw_rectangles(frame, rectangles, offset_x, offset_y):
    """Рисование прямоугольников на кадре."""
    for rectangle in rectangles:
        x1, y1 = rectangle[0]
        x2, y2 = rectangle[1]
        cv2.rectangle(frame, (x1 + offset_x, y1 + offset_y), (x2 + offset_x, y2 + offset_y), (0, 0, 255), 2)


def main():
    cap = initialize_camera()
    frame_width, frame_height = get_frame_dimensions(cap)
    rectangles = create_rectangles()
    offset_x, offset_y = calculate_offset(frame_width, frame_height, rectangles)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        apply_blur(frame, rectangles, offset_x, offset_y, frame_height, frame_width)
        draw_rectangles(frame, rectangles, offset_x, offset_y)

        cv2.imshow("Red Cross", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Выход по нажатию ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

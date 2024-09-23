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


def create_flag_iran(width, height):
    """Создание флага Ирана."""
    # Флаг Ирана состоит из трех горизонтальных полос: зеленая, белая и красная
    flag = np.zeros((height, width, 3), dtype=np.uint8)

    # Наносим цвета
    flag[:height // 3] = [0, 150, 30]  # Зеленый
    flag[height // 3:2 * height // 3] = [255, 255, 255]  # Белый
    flag[2 * height // 3:] = [0, 0, 218]  # Красный

    # Добавляем герб (символы) в центр белой полосы
    # Эти значения могут быть адаптированы в зависимости от разрешения флага
    crest = cv2.imread("files/crest_iran.png", cv2.COLOR_RGBA2RGB)  # Загружаем изображение герба
    crest = cv2.resize(crest, (width // 4, height // 3))  # Изменяем размер

    h, w = crest.shape[:2]
    y_offset = height // 3 + (height // 3 - h) // 2
    x_offset = (width - w) // 2
    flag[y_offset:y_offset + h, x_offset:x_offset + w] = crest

    return flag


def apply_flag(frame, flag, offset_x, offset_y):
    """Применение флага на кадре."""
    h, w = flag.shape[:2]
    frame[offset_y:offset_y + h, offset_x:offset_x + w] = flag


def main():
    cap = initialize_camera()
    frame_width, frame_height = get_frame_dimensions(cap)

    # Создание флага Ирана
    flag = create_flag_iran(frame_width // 4, frame_height // 3)  # Используем 1/4 ширины и 1/3 высоты

    # Определение смещения для размещения флага по центру экрана
    offset_x = (frame_width - flag.shape[1]) // 2
    offset_y = (frame_height - flag.shape[0]) // 2

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        apply_flag(frame, flag, offset_x, offset_y)

        cv2.imshow("Flag of Iran", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Выход по нажатию ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

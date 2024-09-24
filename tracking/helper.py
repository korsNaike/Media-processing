import cv2


def begin():
    return cv2.VideoCapture(0)


def process_stream(cap, process_frame_callback):
    while True:
        ret, frame = cap.read()  # Чтение текущего кадра
        if not ret:  # Проверка на успешное чтение кадра
            break

        process_frame_callback(frame)

        # Ожидание 20 мс и проверка нажатия клавиши 'Esc' (код 27)
        if cv2.waitKey(20) & 0xFF == 27:
            break


def end(cap):
    cap.release()
    cv2.destroyAllWindows()


def do_nothing(n):
    pass


def init_trackbars(window_name, on_change_callback=do_nothing):
    # создаем окно для отображения результата и бегунки
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 300, 200)  # Изменяем размер окна управления
    cv2.createTrackbar("minH", window_name, 119, 179, on_change_callback)
    cv2.createTrackbar("minS", window_name, 140, 255, on_change_callback)
    cv2.createTrackbar("minV", window_name, 3, 255, on_change_callback)
    cv2.createTrackbar("maxH", window_name, 137, 179, on_change_callback)
    cv2.createTrackbar("maxS", window_name, 255, 255, on_change_callback)
    cv2.createTrackbar("maxV", window_name, 255, 255, on_change_callback)

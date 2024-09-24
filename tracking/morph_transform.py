import cv2
import numpy as np

from tracking.filtering_red import create_trackbar_window, get_trackbar_values

def main():
    cap = cv2.VideoCapture(0)
    create_trackbar_window()

    while True:
        ret, frame = cap.read()  # Чтение текущего кадра
        if not ret:  # Проверка на успешное чтение кадра
            break

        minH, minS, minV, maxH, maxS, maxV = get_trackbar_values()
        min_p = (minH, minS, minV)
        max_p = (maxH, maxS, maxV)

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV_FULL)
        hsv_frame[:, :, 0] = (hsv_frame[:, :, 0] + 128) % 0xFF  # смещение красного цвета в центр
        # чтобы не искать спектр мы его приведем в нужные значения самостоятельно

        # применяем фильтр, делаем бинаризацию
        mask = cv2.inRange(hsv_frame, min_p, max_p)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5)))  # erosion + dilation (remove small objects)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5)))  # dilation + erosion (remove small holes)

        hsv_frame_filtered = cv2.bitwise_and(frame, frame, mask=mask)  # создание фильтра
        cv2.imshow('Filtered Web Video', hsv_frame_filtered)

        if cv2.waitKey(20) & 0xFF == 27:  # выйти при нажатии ESC
            break

    cv2.destroyAllWindows()
    cap.release()  # освобождаем захват видео


if __name__ == "__main__":
    main()

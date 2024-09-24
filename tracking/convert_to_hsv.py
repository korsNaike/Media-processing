import cv2
from helper import begin, end, process_stream


def convert_to_hsv(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return hsv_frame


def process_converting(frame):
    hsv_frame = convert_to_hsv(frame)
    cv2.imshow("Web HSV", hsv_frame)  # Отображение кадра
    return hsv_frame


if __name__ == '__main__':
    cap = begin()
    process_stream(cap, process_converting)
    end(cap)

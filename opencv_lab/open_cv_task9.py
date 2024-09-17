import cv2

# URL-адрес камеры телефона
url = 'http://10.200.192.217:8080/video'

cap = cv2.VideoCapture(url)

# Устанавливаем размер окна
window_width = 640
window_height = 480

while True:
    # Считывание кадра с камеры телефона
    ret, frame = cap.read()

    # Уменьшаем размер кадра
    frame = cv2.resize(frame, (window_width, window_height))

    # Отображение кадра
    cv2.imshow('Camera', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

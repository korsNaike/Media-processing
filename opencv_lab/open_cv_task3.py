import cv2

# Открываем видео
cap = cv2.VideoCapture('files/video2.mp4')

# Проверяем, удалось ли открыть видео
if not cap.isOpened():
    print("Не удалось открыть видео")
    exit()

# Получаем информацию о видео
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Создаем окно для отображения видео
cv2.namedWindow('Video', cv2.WINDOW_NORMAL)

# Устанавливаем коэффициент замедления
slow_down_factor = 2

# Отображаем видео
while True:
    # Считываем кадр
    ret, frame = cap.read()

    # Если кадр не удалось считать, выходим из цикла
    if not ret:
        break

    # Меняем размер кадра
    frame = cv2.resize(frame, (1200, 700))

    # Меняем цветовую гамму
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # Задержка для замедления видео
    cv2.waitKey(int(1000 / fps * slow_down_factor))

    # Отображаем кадр
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# Освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()

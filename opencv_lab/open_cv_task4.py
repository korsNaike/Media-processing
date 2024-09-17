import cv2

# Открываем исходное видео
input_video = cv2.VideoCapture('files/video.mp4')

# Получаем информацию о видео
fps = input_video.get(cv2.CAP_PROP_FPS)
width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = input_video.get(cv2.CAP_PROP_FRAME_COUNT)

# Создаем объект для записи выходного видео
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter('files/gray_video.mp4', fourcc, fps, (width, height), isColor=False)

# Отображаем и записываем видео
while True:
    # Считываем кадр
    ret, frame = input_video.read()

    # Если кадр не удалось считать, выходим из цикла
    if not ret:
        break

    # Конвертируем кадр в оттенки серого
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Записываем кадр в выходное видео
    output_video.write(gray_frame)

    # Получаем текущий номер кадра
    current_frame = input_video.get(cv2.CAP_PROP_POS_FRAMES)
    # Вычисляем процент завершения
    progress = (current_frame / total_frames) * 100

    # Выводим процесс перезаписи видео
    print(f"Перезапись видео: {progress:.2f}%")

# Освобождаем ресурсы
input_video.release()
output_video.release()
cv2.destroyAllWindows()

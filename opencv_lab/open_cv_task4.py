import cv2

# Название входного файла
input_file = "files/output.mp4"
# Название выходного файла
output_file = "files/rerecord_video.mp4"

# Открываем входной видеофайл
cap = cv2.VideoCapture(input_file)

# Получаем параметры входного видео
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Создаем выходной видеофайл
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

# Читаем и записываем кадры
while True:
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)

# Освобождаем ресурсы
cap.release()
out.release()

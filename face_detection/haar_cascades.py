import cv2


def haar_face_detection():
    # Загрузка предобученного классификатора Хаара для обнаружения лиц
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Открытие доступа к веб-камере (0 - для первой камеры)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Ошибка: не удалось открыть камеру.")
        exit()

    while True:
        # Считывание кадра с камеры
        ret, frame = cap.read()
        if not ret:
            print("Ошибка: не удалось считать кадр.")
            break

        # Преобразование изображения в градации серого
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Обнаружение лиц на кадре
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Рисование прямоугольников вокруг обнаруженных лиц
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow('Haar Face Detection', frame)

        # Выход по нажатию клавиши ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    haar_face_detection()

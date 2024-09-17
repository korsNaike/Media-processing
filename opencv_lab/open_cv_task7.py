import cv2


def initialize_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Ошибка: не удалось открыть вебкамеру.")
    return cap


def initialize_video_writer(filepath, fps, width, height):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(filepath, fourcc, fps, (width, height))


def record_video(cap, out):
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Ошибка: не удалось прочитать кадр.")
            break

        display_information(frame)
        out.write(frame)
        cv2.imshow('Video from camera', frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            break


def display_information(frame):
    cv2.putText(frame, "Press 'ESC' to finish recording", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                cv2.LINE_AA)


def play_saved_video(filepath):
    cap_video = cv2.VideoCapture(filepath)
    while cap_video.isOpened():
        ret, frame = cap_video.read()
        if not ret:
            break
        cv2.imshow('Saved Video', frame)
        if cv2.waitKey(30) & 0xFF == 27:  # ESC key
            break
    cap_video.release()


def main():
    FILEPATH = 'files/output.mp4'
    fps = 32.0
    cap = initialize_camera()
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = initialize_video_writer(FILEPATH, fps, width, height)

    try:
        record_video(cap, out)
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    play_saved_video(FILEPATH)


if __name__ == "__main__":
    main()

import logging

import cv2
import numpy as np


class MotionDetector:
    threshold : float = None
    contour_area : float = None

    kernel_size : int = None
    sigma_x : int = None
    sigma_y : int = None

    def __init__(
            self,
            threshold: float = 15.0,
            contour_area: float = 800.0,
            kernel_size: int = 15,
            sigma_x: int = 5,
            sigma_y: int = 5
    ):
        self.threshold = threshold
        self.contour_area = contour_area
        self.kernel_size = kernel_size
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y

    def get_video_with_motions(
            self,
            input_path: str,
            output_path: str,
    ):
        video_capture = cv2.VideoCapture(input_path)
        ret, frame = video_capture.read()
        if not ret:
            logging.error('Не удалось открыть видеофайл.')
            return

        # Читаем первый кадр в чб, применяем размытие Гаусса
        processed_frame = self.prepare_frame(frame)

        # Подготовка файла для записи нового видео
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_writer = cv2.VideoWriter(output_path, fourcc, 24, (frame_width, frame_height))
        logging.info(f"Видео будет сохранено по адресу: {output_path}")

        while True:
            previous_frame = processed_frame.copy()
            ret, frame = video_capture.read()
            if not ret:
                break

            # Преобразование текущего кадра в оттенки серого и размытие
            processed_frame = self.prepare_frame(frame)

            # Вычисление разницы между текущим и предыдущим кадром
            frame_difference = cv2.absdiff(previous_frame, processed_frame)

            # Проводим операцию двоичного разделения:
            # проводим бинаризацию изображения по пороговому значению (оставляем либо 255, либо 0)
            _, thresholded_frame = cv2.threshold(frame_difference, self.threshold, 255, cv2.THRESH_BINARY)

            # Поиск контуров объектов
            contours, _ = cv2.findContours(thresholded_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                contour_area = cv2.contourArea(contour)
                if contour_area < self.contour_area: # Ищем контур больше заданного значения
                    continue
                # Запись исходного кадра, если найдены значимые изменения
                video_writer.write(frame)
                break

            # Прерывание по нажатию клавиши 'Esc'
            if cv2.waitKey(1) & 0xFF == 27:
                break

        video_capture.release()
        video_writer.release()
        logging.info("Видео успешно записано")
        cv2.destroyAllWindows()

    def prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Подготовить фрейм к обработке, привести к ЧБ, применить фильтр Гаусса
        :param frame: Кадр д
        :return:
        """
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(
            gray_frame,
            (self.kernel_size, self.kernel_size),
            sigmaX=self.sigma_x,
            sigmaY=self.sigma_y
        )

    def __str__(self):
        return f'detected-kernel={self.kernel_size}-sigma={(self.sigma_x, self.sigma_y)}-threshold={self.threshold}-contour={self.contour_area}'


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

if __name__ == '__main__':
    detector = MotionDetector(threshold=1, contour_area=1000)
    detector.get_video_with_motions(f'../videos/ЛР4_main_video.mov', f'../videos/{detector}.mp4')


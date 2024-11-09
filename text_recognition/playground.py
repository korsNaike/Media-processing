import re

import numpy as np
import easyocr
import cv2

from text_recognition.dataset_preparing.images_crop import get_image_filenames

# Создайте экземпляр рецензента
reader = easyocr.Reader(['en'])  # Укажите языки, которые вы хотите использовать

def clear_from_chars(string: str) -> str:
    """
    Удалить все символы кроме цифр из строки
    :param string:
    :return:
    """
    return re.sub('[^0-9]', '', string)

def recognize_text(image_path: str):
    image = cv2.imread(image_path)
    # Распознавание текста, включая цифры
    results = reader.readtext(image, allowlist='0123456789')
    # Вывод результатов
    for (bbox, text, prob) in results:
        return clear_from_chars(text)

if __name__ == '__main__':
    # Точность без обучения без предобработки фото - 0,30
    # Точность без обучения с предобработкой фото - 0,475
    images_names = get_image_filenames('./dataset/preprocessed')
    all_num = len(images_names)
    correct_num = 0
    for image_name in images_names:
        correct_answer = clear_from_chars(image_name)
        result = recognize_text('./dataset/preprocessed/' + image_name)
        if result == correct_answer:
            correct_num += 1
    print(f'Всего: {all_num}, верных: {correct_num}, процент: {100 * float(correct_num / all_num)}%')
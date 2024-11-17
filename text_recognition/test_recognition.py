import csv

import cv2
import numpy as np
import pytesseract

from text_recognition.dataset_preparing.images_crop import get_image_filenames
from text_recognition.models.BaseModel import BaseModel
from text_recognition.models.ClassicPytesseract import ClassicPytesseract
from text_recognition.playground import clear_from_chars
from text_recognition.val_types.BaseValType import BaseValType
from text_recognition.val_types.FullValType import FullValType
from text_recognition.val_types.LevensteinValType import LevensteinValType


def test_recognition(rec_type: BaseModel, val_type: BaseValType, need_write_answers: bool = True):
    test_csv_file = f"{str(rec_type.__class__.__name__)}-test.csv"

    answers = []
    for img_filename in get_image_filenames('./dataset/formatted/val'):
        img = cv2.imread(f'./dataset/formatted/val/{img_filename}')
        text_from_model = rec_type.image_to_string(img)
        answers.append(
            {
                "model": text_from_model,
                "correct": clear_from_chars(img_filename),
            }
        )

    if need_write_answers:
        write_answers(test_csv_file, answers)
    calc_accuracy(answers, val_type)

def write_answers(csv_file: str, answers: list[dict[str, str]]):

    with open(csv_file, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['model', 'correct'])
        writer.writeheader()
        for label in answers:
            writer.writerow(label)

def calc_accuracy(answers: list[dict[str, str]] , val_type: BaseValType):
    accuracy_values = []
    for answer in answers:
        accuracy_values.append(val_type.check_value(answer['model'], answer['correct']))

    print(f"Точность: {np.mean(accuracy_values)}")

if __name__ == '__main__':
    test_recognition(ClassicPytesseract(), LevensteinValType(), need_write_answers=True)
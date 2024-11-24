import csv
import logging

import cv2
import numpy as np
import pytesseract
import easyocr

from text_recognition.dataset_preparing.images_crop import get_image_filenames
from text_recognition.models.BaseModel import BaseModel
from text_recognition.models.ClassicEasyOCR import ClassicEasyOCR
from text_recognition.models.ClassicPytesseract import ClassicPytesseract
from text_recognition.models.TesseractWithAugmentation import TesseractWithAugmentation
from text_recognition.models.TesseractWithPostProcessing import TesseractWithPostProcessing
from text_recognition.playground import clear_from_chars
from text_recognition.val_types.BaseValType import BaseValType
from text_recognition.val_types.FullValType import FullValType
from text_recognition.val_types.LevensteinValType import LevensteinValType


def test_recognition(rec_type: BaseModel, val_type: BaseValType, path_to_ds: str, csv_prefix: str, need_write_answers: bool = True):
    test_csv_file = f"{str(rec_type.__class__.__name__)}-{csv_prefix}-test.csv"

    answers = []
    for img_filename in get_image_filenames(path_to_ds):
        img = cv2.imread(f'{path_to_ds}/{img_filename}')
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

def calc_accuracy_by_answers_file(answers_file: str , val_type: BaseValType):
    accuracy_values = []
    with open(answers_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            accuracy_values.append(val_type.check_value(row['model'], row['correct']))
    print(f"Точность: {np.mean(accuracy_values)}")

logging.basicConfig(level=logging.INFO)

def start_test():
    test_recognition(
        ClassicEasyOCR(
            easyocr.Reader(['en'],
                        model_storage_directory='custom_EasyOCR/model',
                        user_network_directory='custom_EasyOCR/user_network',
                        recog_network='custom_example')
        ),
        FullValType(),
        path_to_ds='./dataset/formatted-v2/val',
        csv_prefix='trained-v2',
        need_write_answers=True
    )

def start_calc():
    calc_accuracy_by_answers_file(
        './ClassicEasyOCR-trained-v2-test.csv',
        LevensteinValType()
    )

if __name__ == '__main__':
    # start_test()
    start_calc()

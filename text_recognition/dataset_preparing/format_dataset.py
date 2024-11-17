import csv
import os
import shutil
import numpy as np
from text_recognition.dataset_preparing.images_crop import get_image_filenames
from text_recognition.playground import clear_from_chars


def format_dataset(prepared_images_folder: str):
    root_dataset_folder = prepared_images_folder + '/../'
    prepared_images = np.array(get_image_filenames(prepared_images_folder))

    split_ratio = 0.8
    split_index = int(len(prepared_images) * split_ratio)

    # Разделение массива
    train_images = prepared_images[:split_index]
    val_images = prepared_images[split_index:]

    train_folder = root_dataset_folder + '/formatted-v2/train'
    val_folder = root_dataset_folder + '/formatted-v2/val'

    clear_folders([train_folder, val_folder])

    write_labels(train_folder, train_images)
    write_labels(val_folder, val_images)

    copy_images_in_another_folder(prepared_images_folder, train_folder, train_images)
    copy_images_in_another_folder(prepared_images_folder, val_folder, val_images)



def write_labels(folder_with_data: str, prepared_images: np.array):
    labels_dict: list[dict[str, str]] = []
    for img in prepared_images:
        labels_dict.append({'filename': img, 'words': clear_from_chars(img)})

    with open(folder_with_data + '/labels.csv', 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'words'])
        writer.writeheader()
        for label in labels_dict:
            writer.writerow(label)

def copy_images_in_another_folder(
        original_folder: str,
        folder_for_copy: str,
        images_names_to_copy: np.array
):
    for img in images_names_to_copy:
        shutil.copy(original_folder + '/' + img, folder_for_copy + '/' + img)

def clear_folders(folders: list[str]):
    for folder in folders:
        shutil.rmtree(folder)
        os.makedirs(folder)

if __name__ == '__main__':
    format_dataset('../dataset/v2')
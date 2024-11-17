import os
import cv2

from text_recognition.dataset_preparing.images_crop import get_image_filenames


def augment_image(image_path: str, folder_to_save: str):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Не удалось загрузить изображение по пути: {image_path}")

    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)

    if ext.lower() not in ('.jpg', '.jpeg', '.png', '.bmp'):
        raise ValueError("Допустимые форматы изображений: jpg, jpeg, png, bmp")

    joined_str = 'a'
    for angle in range(-20, 21):
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        rotated_image = cv2.warpAffine(image, M, (w, h))

        new_name = f"{name}_rotated_{joined_str}{ext}"
        joined_str += 'a'

        cv2.imwrite(folder_to_save + '/' + new_name, rotated_image)

if __name__ == '__main__':
    for image in get_image_filenames('../dataset/preprocessed'):
        augment_image(
            '../dataset/preprocessed/' + image,
            folder_to_save='../dataset/v2'
        )
import cv2
import numpy as np

from text_recognition.dataset_preparing.images_crop import get_image_filenames


def preprocess_image(img: np.ndarray) -> np.ndarray:
    """
    Провести предобработку изображения, привести к ЧБ, увеличить контрастность и наложить размытие
    :param img:
    :return:
    """
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    normalized_image = cv2.normalize(gray_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(normalized_image)

    blurred = cv2.GaussianBlur(enhanced_image, (5, 5), 3, sigmaY=3)
    return blurred

if __name__ == '__main__':
    images_names = get_image_filenames('../dataset/cropped')
    for image_name in images_names:
        image = cv2.imread('../dataset/cropped/' + image_name)
        # cv2.imshow('image', image)
        edges = preprocess_image(image)
        cv2.imwrite('../dataset/preprocessed/' + image_name, edges)
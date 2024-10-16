import numpy as np
import cv2

from gauss.convolution_matrix import conv_matrix, normalize_matrix


def conv_operation(
        element_x: int,
        element_y: int,
        original_matrix: np.ndarray,
        ker_matrix: np.ndarray,
        ker_size: int
) -> float:
    """
    Провести операцию свёртки
    :param element_x: Индекс элемента исходной матрицы по горизонтали
    :param element_y: Индекс элемента исходной матрицы по вертикали
    :param original_matrix: Исходная матрица
    :param ker_matrix: Ядро свёртки
    :param ker_size: Размер ядра свёртки
    :return:
    """
    value = 0
    half_size = int(ker_size // 2)
    for k in range(-(ker_size // 2), ker_size // 2 + 1):
        for l in range(-(ker_size // 2), ker_size // 2 + 1):
            value += original_matrix[element_y + k, element_x + l] + ker_matrix[k + half_size, l + half_size]

    return value


def gaussian_blur(img: np.ndarray, ker_size: int, ms_deviation: int | float) -> np.ndarray:
    """
    Провести операцию размытия Гаусса
    :param img: Фото (матрица Numpy)
    :param ker_size: Размер ядра
    :param ms_deviation: Среднеквадратичное отклонение
    :return: Размытое фото
    """
    ker_matrix = normalize_matrix(conv_matrix(ker_size, ms_deviation))

    blurred_img = img.copy()
    h, w = img.shape[:2]
    half_ker_size = int(ker_size // 2)
    for y in range(half_ker_size, h - half_ker_size):  # Проход по матрице вертикально
        for x in range(half_ker_size, w - half_ker_size):  # Проход по матрице горизонтально
            # Операция свёртки
            blurred_val = 0
            for k in range(-(ker_size // 2), ker_size // 2 + 1):  # Проходим по матрице свёртки
                for l in range(-(ker_size // 2), ker_size // 2 + 1):
                    blurred_val += img[y + k, x + l] * ker_matrix[k + half_ker_size, l + half_ker_size]
            blurred_img[y, x] = blurred_val

    return blurred_img


if __name__ == '__main__':
    img = cv2.imread('./../opencv_lab/files/cat.jpg')
    img = cv2.resize(img, [i // 2 for i in img.shape[1::-1]])
    print("Уменьшили размер фото..")
    ker_size = 15
    ms_deviation = 5
    print("Начата обработка..")
    blurred_img = gaussian_blur(img, ker_size, ms_deviation)
    print("Обработка закончена")
    cv2.imshow(f'Blurred photo (conv matrix size={ker_size}, deviation={ms_deviation})', blurred_img)

    img_blur_cv2 = cv2.GaussianBlur(img, (ker_size, ker_size), ms_deviation)
    cv2.imshow(f'cv2 Gaussian blur photo', img_blur_cv2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

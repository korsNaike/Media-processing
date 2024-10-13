import numpy as np


def gauss(x: int, y: int, sigma: float | int, a: float | int, b: float | int) -> float:
    """
    Функция Гаусса
    :param x: Первый индекс матрицы
    :param y: Второй индекс матрицы
    :param sigma: Среднеквадратичное отклонение
    :param a: Мат. ожидание двумерной случайной величины
    :param b: Мат. ожидание двумерной случайной величины
    :return: Значение гауссовой функции
    """
    double_sigma_squared = 2 * sigma * sigma
    return np.exp(-((x - a) ** 2 + (y - b) ** 2) / double_sigma_squared) / (np.pi * double_sigma_squared)


def conv_matrix(matrix_size: int, ms_deviation: float | int) -> np.ndarray:
    """
    Создание и заполнение матрицы свёртки
    :param matrix_size: Размер матрицы
    :param ms_deviation: Среднеквадратичное отклонение
    :return: Матрица numpy
    """
    matrix = np.zeros((matrix_size, matrix_size))  # Инициализируем матрицу нулями
    a = b = matrix_size // 2  # Считаем математическое ожидание двумерной случайной величины

    # Заполяем матрицу свёртки
    for y in range(matrix_size):
        for x in range(matrix_size):
            matrix[y, x] = gauss(x, y, ms_deviation, a, b)

    return matrix


def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Нормируем матрицу, чтобы сумма элементов была равна 1
    :param matrix:
    :return:
    """
    return matrix / np.sum(matrix)


if __name__ == '__main__':
    ms_deviation = 3
    for matrix_size in (3, 5, 7):
        print(f'\nРазмер матрицы: {matrix_size}')
        print(f'Среднеквадратичное отклонение: {ms_deviation}')
        matrix = conv_matrix(matrix_size, ms_deviation)
        matrix = normalize_matrix(matrix)
        print(matrix)
        print(f'Сумма элементов матрицы: {np.sum(matrix)}')
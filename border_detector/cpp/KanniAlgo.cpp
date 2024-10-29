#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// Функция для вычисления длины градиента изображения
void computeGradientMagnitude(const Mat& inputImage, Mat& gradientMagnitude) {
    Mat gradientX, gradientY;
    // Вычисляем градиенты по осям X и Y
    Sobel(inputImage, gradientX, CV_32FC1, 1, 0);
    Sobel(inputImage, gradientY, CV_32FC1, 0, 1);
    // Вычисляем величину градиента
    magnitude(gradientX, gradientY, gradientMagnitude);
}

// Функция для вычисления угла градиента изображения
void computeGradientAngle(const Mat& inputImage, Mat& gradientAngle) {
    Mat gradientX, gradientY;
    // Вычисляем градиенты по осям X и Y
    Sobel(inputImage, gradientX, CV_32FC1, 1, 0);
    Sobel(inputImage, gradientY, CV_32FC1, 0, 1);
    // Вычисляем угол градиента
    phase(gradientX, gradientY, gradientAngle);
}

// Функция для подавления не максимальных значений
void applyNonMaximumSuppression(const Mat& gradientMagnitude, const Mat& gradientAngle) {
    Mat suppressedImage = gradientMagnitude.clone();

    for (int i = 1; i < gradientMagnitude.rows - 1; i++) {
        for (int j = 1; j < gradientMagnitude.cols - 1; j++) {
            float angle = gradientAngle.at<float>(i, j);
            float magnitudeValue = gradientMagnitude.at<float>(i, j);

            // Отсекаем границы для дальнейших вычислений
            if (i == 0 || i == gradientMagnitude.rows - 1 || j == 0 || j == gradientMagnitude.cols - 1) {
                suppressedImage.at<float>(i, j) = 0;
            }
            else {
                int xOffset = 0;
                int yOffset = 0;

                // Определение смещений в зависимости от угла градиента
                if (angle == 0 || angle == 4) {
                    xOffset = 0;
                }
                else if (angle > 0 && angle < 4) {
                    xOffset = 1;
                }
                else {
                    xOffset = -1;
                }

                if (angle == 2 || angle == 6) {
                    yOffset = 0;
                }
                else if (angle > 2 && angle < 6) {
                    yOffset = -1;
                }
                else {
                    yOffset = 1;
                }

                // Проверяем, является ли текущее значение градиента максимальным
                bool isLocalMaxima = magnitudeValue >= gradientMagnitude.at<float>(i + yOffset, j + xOffset) &&
                    magnitudeValue >= gradientMagnitude.at<float>(i - yOffset, j - xOffset);
                suppressedImage.at<float>(i, j) = isLocalMaxima ? magnitudeValue : 0;
            }
        }
    }
    imshow("Non_Maximum_Suppression", suppressedImage);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cout << "Использование: " << argv[0] << " <путь_к_изображению>" << endl;
        return -1;
    }

    string imagePath = argv[1];
    Mat cannyOutput;

    setlocale(LC_ALL, "Russian");
    Mat inputImage = imread(imagePath, IMREAD_GRAYSCALE);

    if (inputImage.empty()) {
        cout << "Ошибка: Не удалось загрузить изображение: " << imagePath << endl;
        return -1;
    }

    Mat gradientMagnitude(inputImage.size(), CV_32FC1);
    Mat gradientAngleImage(inputImage.size(), CV_32FC1);

    computeGradientMagnitude(inputImage, gradientMagnitude);
    computeGradientAngle(inputImage, gradientAngleImage);

    applyNonMaximumSuppression(gradientMagnitude, gradientAngleImage);
    Canny(inputImage, cannyOutput, 50, 150);

    // Отображение исходного изображения и результатов обработки

    imshow("Original_Image", inputImage);
    // imshow("Gradient_Magnitude", gradientMagnitude / 255.0f);
    // imshow("Gradient_Angle", gradientAngleImage / CV_PI / 2.0f);
    imshow("Canny_Output", cannyOutput);

    waitKey(0); // Ожидание нажатия клавиши

    return 0;
}

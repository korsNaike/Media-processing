import os

import cv2


def crop_center(image_path, output_path, crop_width=220, crop_height=120):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    img_height, img_width, _ = image.shape

    center_x, center_y = img_width // 2, img_height // 2


    start_x = center_x - crop_width // 2 - 140
    start_y = center_y - crop_height // 2

    cropped_image = image[start_y:start_y + crop_height, start_x:start_x + crop_width]
    cv2.imwrite(output_path, cropped_image)


def get_image_filenames(directory):
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')

    image_files = []

    for filename in os.listdir(directory):
        if filename.lower().endswith(image_extensions):
            image_files.append(filename)

    return image_files



if __name__ == "__main__":
    images_names = get_image_filenames('../dataset/raw')
    for image_name in images_names:
        crop_center('../dataset/raw/' + image_name, '../dataset/cropped/' + image_name)
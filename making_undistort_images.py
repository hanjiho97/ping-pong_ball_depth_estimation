import glob
import os
import cv2
import numpy as np
import yaml


YAML_PATH = 'calibration_data.yaml'
OUTPUT_FOLDER= './output/'
IMAGE_PATH = './images/'
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640


def parse_calibration_data(file_path):
    with open(file_path, 'r') as file:
        calibration_data = yaml.load(file)
    camera_matrix = np.array([[0]*3 for _ in range(3)])
    for row in range(calibration_data['CAMERA_MATRIX']['ROW']):
        for col in range(calibration_data['CAMERA_MATRIX']['COL']):
            index = row * calibration_data['CAMERA_MATRIX']['ROW'] + col
            camera_matrix[row][col] = calibration_data['CAMERA_MATRIX']['DATA'][index]
    distortion_coefficients = np.array(calibration_data['DISTORTION_COEFFICIENTS']['DATA'])
    return camera_matrix, distortion_coefficients


def make_calibration_images(image_file_path_list, mapx, mapy, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for image_file_path in image_file_path_list:
        image_file_name = image_file_path.split('/')[-1]
        image = cv2.imread(image_file_path, cv2.IMREAD_COLOR)
        image_undistort = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
        cv2.imwrite(output_folder+image_file_name, image_undistort)


if __name__ == '__main__':
    camera_matrix, distortion_coefficients = parse_calibration_data(YAML_PATH)
    image_size = (IMAGE_WIDTH, IMAGE_HEIGHT)
    mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, distortion_coefficients, None, None, image_size, cv2.CV_32FC1)
    image_file_path_list = glob.glob(IMAGE_PATH + '*.jpg')
    make_calibration_images(image_file_path_list, mapx, mapy, OUTPUT_FOLDER)

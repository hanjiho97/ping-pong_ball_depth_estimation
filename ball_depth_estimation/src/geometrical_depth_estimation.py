#!/usr/bin/env python3
import math
import rospy
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sensor_msgs.msg import Image
from yolov3_trt_ros.msg import BoundingBox, BoundingBoxes

calibration_image = np.empty(shape=[0])
bbox_list_raw = []
CAMERA_HEIGHT = 0.16
FOV_V = 150 #need to change
FX = 346.25953
FY = 346.5049
CX = 317.36334
CY = 204.58251
FOV_H = 170
FOV_V = 160

VISUAL_IMAGE_PATH = '/home/hanjiho97/xycar_ws/src/ball_depth_estimation/src/car_fleid.jpg'


def image_callback(data):
    global calibration_image
    calibration_image = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
    calibration_image = cv2.cvtColor(calibration_image, cv2.COLOR_RGB2BGR)


def bbox_callback(data):
    global bbox_list_raw
    bbox_list_raw = []
    for bbox in data.bounding_boxes:
        bbox.xmin = max(min(639, bbox.xmin), 0)
        bbox.ymin = max(min(479, bbox.ymin), 0)
        bbox.xmax = max(min(639, bbox.xmax), 0)
        bbox.ymax = max(min(479, bbox.ymax), 0)
        box = [bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax, 0]
        bbox_list_raw.append(box)


def get_depth(bbox_list, carmera_height, fx, fy, cx, cy, fov_v, fov_h):
    for index, bbox in enumerate(bbox_list):
        # normalized Image plane
        xmin = bbox[0]
        ymin = bbox[1]
        xmax = bbox[2]
        ymax = bbox[3]
        y_norm = (ymax - cy) / fy
        azimuth = (((xmin+xmax) / 2) - 320) * (fov_v / 2)
        distance = 1 * carmera_height / y_norm
        depth = distance * math.cos(azimuth)
        # m -> mm
        distance = int(distance * 1000) - 150 # minus camera to bumper size
        bbox_list[index][4] = distance
    return bbox_list


def draw_box_and_depth(image, bbox_list):
    for bbox in bbox_list:
        x1, y1, x2, y2, depth = bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(image, f"{depth*0.1:.4}cm", (x2, y2+10), 1, 1, (255, 255, 0), 2)
    cv2.imshow('result', image)
    cv2.waitKey(1)


def draw_position(bbox_list, image_path):
    visual_fleid = cv2.imread(image_path, cv2.IMREAD_COLOR)
    visual_fleid = cv2.resize(visual_fleid, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    for bbox in bbox_list:
        distance = bbox[4]
        cv2.circle(visual_fleid, (257, 675-int(distance*17/45)), 20, (0, 127, 255), -1, cv2.LINE_AA)
        cv2.putText(visual_fleid, f"x: 0 y: {distance*0.1:.4}cm", (277, 655-int(distance*17/45)), 1, 1, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('visual_fleid', visual_fleid)
    cv2.waitKey(1)


def start_depth_estimation():
    rate = rospy.Rate(10)
    image_sub = rospy.Subscriber('/usb_cam/cailbration_image', Image, image_callback)
    bbox_sub = rospy.Subscriber('/yolov3_ros/detections', BoundingBoxes, bbox_callback)
    while not rospy.is_shutdown():
        rate.sleep()
        if calibration_image.shape[0] == 0:
            continue
        bbox_list = get_depth(bbox_list_raw, CAMERA_HEIGHT, FX, FY, CX, CY, FOV_V, FOV_H)
        draw_box_and_depth(calibration_image, bbox_list)
        draw_position(bbox_list, VISUAL_IMAGE_PATH)
        # # FuncAnimation(plt.gcf(), draw_position, fargs=bbox_list, interval=100)
        # plt.tight_layout()


if __name__ == '__main__':
    rospy.init_node('geometrical_depth_estimation', anonymous=True)
    start_depth_estimation()

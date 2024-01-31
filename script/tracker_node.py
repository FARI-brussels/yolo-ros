#!/usr/bin/env python3

import cv2
import cv_bridge
import numpy as np
import roslib.packages
import rospy
from sensor_msgs.msg import Image
from ultralytics import YOLO
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from ultralytics_ros.msg import YoloResult, RobotPositionArray, RobotPosition
from geometry_msgs.msg import Point
from roboflow import Roboflow
from datetime import datetime
import json



def load_coefficients(path):
    """
    Load camera matrix (K) and distortion coefficients (D) from a file.

    Parameters:
    - path (str): Path to the file containing camera calibration coefficients.

    Returns:
    - mtx (numpy.ndarray): Camera matrix (intrinsic parameters).
    - dist (numpy.ndarray): Distortion coefficients.
    """
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    mtx = cv_file.getNode("K").mat()
    dist = cv_file.getNode("D").mat()
    H = cv_file.getNode("H").mat()
    cv_file.release()
    
    return mtx, dist, H

def point_coordinates_to_world_coordinates(img_point, H):
    """
    Transform a point from image coordinates to world coordinates using a given homography matrix.
    
    Parameters:
    - img_point (list or tuple): Image point in the format [x, y].
    - H (numpy.ndarray): Homography matrix.
    
    Returns:
    - world_point (list): Transformed point in world coordinates in the format [X, Y].
    """
    # Convert point to homogeneous coordinates
    img_point_homogeneous = np.array([img_point[0], img_point[1], 1])
    
    # Use the homography matrix to get world coordinates in homogeneous form
    world_point_homogeneous = np.dot(H, img_point_homogeneous)
    
    # Convert back from homogeneous coordinates to 2D
    world_point = world_point_homogeneous[:2] / world_point_homogeneous[2]
    
    return world_point.tolist()



class TrackerNode:
    def __init__(self):
        path = roslib.packages.get_pkg_dir("ultralytics_ros")
        with open(f"{path}/keys.json", 'r') as file:
            ROBOFLOW_API_KEY = json.load(file)["ROBOFLOW_API_KEY"]
            RF = Roboflow(api_key=ROBOFLOW_API_KEY)
        yolo_model = rospy.get_param("~yolo_model", "yolov8n.pt")
        camera_calibration_file = rospy.get_param("~camera_calibration_file", "calibration.yml")
        self.positions_topic = rospy.get_param("~positions_topic", "positions")
        self.result_image_topic = rospy.get_param("~result_image_topic", "yolo_image")
        self.conf_thres = rospy.get_param("~conf_thres", 0.25)
        self.iou_thres = rospy.get_param("~iou_thres", 0.45)
        self.max_det = rospy.get_param("~max_det", 300)
        self.rate = rospy.Rate(rospy.get_param("~frame_rate", 30))
        self.classes = rospy.get_param("~classes", None)
        self.tracker = rospy.get_param("~tracker", "bytetrack.yaml")
        self.device = rospy.get_param("~device", None)
        self.roboflow_project = RF.workspace(rospy.get_param("~roboflow_workspace", None)).project(rospy.get_param("~roboflow_project", None))
        self.send_data_to_roboflow = rospy.get_param("~send_data_to_roboflow", False)
        self.roboflow_send_frequency = rospy.get_param("~roboflow_send_frequency", 0.1)
        self.last_sent_time = rospy.Time.now()
        self.result_conf = rospy.get_param("~result_conf", True)
        self.result_line_width = rospy.get_param("~result_line_width", None)
        self.result_font_size = rospy.get_param("~result_font_size", None)
        self.result_font = rospy.get_param("~result_font", "Arial.ttf")
        self.result_labels = rospy.get_param("~result_labels", True)
        self.result_boxes = rospy.get_param("~result_boxes", True)
        
        self.MTX, self.DIST, self.H = load_coefficients(f"{path}/{camera_calibration_file}")
        self.model = YOLO(f"{path}/models/{yolo_model}")
                #Instanciate roboflow workspace
        self.model.fuse()
        self.bridge = cv_bridge.CvBridge()

        # Initialize OpenCV video capture
        self.cap = cv2.VideoCapture(0)  # Adjust '0' if your camera index is different
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


        # Other publishers remain the same
        self.robot_positions_pub = rospy.Publisher(self.positions_topic, RobotPositionArray, queue_size=10)
        self.result_image_pub = rospy.Publisher(
            self.result_image_topic, Image, queue_size=1
        )


    def start(self):
        while not rospy.is_shutdown():
            # Capture frame-by-frame
            ret, frame = self.cap.read()
            if not ret:
                rospy.logerr("Failed to capture image")
                continue


            # Process and publish the frame
            self.image_callback(frame)

            # Sleep to maintain the desired rate
            self.rate.sleep()

    def image_callback(self, frame):
        cv_image = cv2.warpPerspective(cv2.undistort(frame, self.MTX, self.DIST), self.H, (1286, 1008))
        current_time = rospy.Time.now()
        if (current_time - self.last_sent_time).to_sec() >= 1.0 / self.roboflow_send_frequency and self.send_data_to_roboflow:
            print("sending to roboflow")
            save_path = f"arena_image_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
            cv2.imwrite(save_path, cv_image)
            self.roboflow_project.upload(save_path)
            self.last_sent_time = current_time  # Update the last sent time

        results = self.model.track(
            source=cv_image,
            conf=self.conf_thres,
            iou=self.iou_thres,
            max_det=self.max_det,
            classes=self.classes,
            tracker=self.tracker,
            device=self.device,
            verbose=False,
            retina_masks=True,
        )

        if results is not None:
            yolo_result_msg = YoloResult()
            yolo_result_image_msg = Image()
            yolo_result_msg.detections = self.create_detections_array(results)
            yolo_result_image_msg = self.create_result_image(results)
            robot_positions_msg = RobotPositionArray()
            self.result_image_pub.publish(yolo_result_image_msg)
                # Create a RobotPositionArray message
            positions = self.get_detection_centers(results)
            robot_positions_msg = RobotPositionArray()
            robot_positions_msg.positions = [
                RobotPosition(robot_id=rid, position=Point(x=pos[0], y=pos[1])) for rid, pos in positions.items()
            ]

            # Publish the message
            self.robot_positions_pub.publish(robot_positions_msg)

    def create_detections_array(self, results):
        self.get_detection_centers(results)
        detections_msg = Detection2DArray()
        bounding_box = results[0].boxes.xywh
        classes = results[0].boxes.cls
        confidence_score = results[0].boxes.conf
        for bbox, cls, conf in zip(bounding_box, classes, confidence_score):
            detection = Detection2D()
            detection.bbox.center.x = float(bbox[0])
            detection.bbox.center.y = float(bbox[1])
            detection.bbox.size_x = float(bbox[2])
            detection.bbox.size_y = float(bbox[3])
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.id = int(cls)
            hypothesis.score = float(conf)
            detection.results.append(hypothesis)
            detections_msg.detections.append(detection)
        return detections_msg


    def get_detection_centers(self, results):
        """
        Process the detection results and output a dictionary with the position of the centers of the bounding boxes.

        :param results: The output from the detection model.
        :return: A dictionary where keys are class IDs and values are lists of (x, y) tuples representing the centers of the bounding boxes.
        """
        detections = {}
        if results is not None:
            bounding_boxes = results[0].boxes.xywh
            classes = results[0].boxes.cls

            for i, bbox in enumerate(bounding_boxes):
                x_center = float(bbox[0] + bbox[2] / 2)  # x_center = x + width/2
                y_center = float(bbox[1] + bbox[3] / 2)  # y_center = y + height/2
                x_center, y_center = point_coordinates_to_world_coordinates((x_center, y_center), self.H)
                detections[i] = (x_center, y_center)
        return detections


    def create_result_image(self, results):
        plotted_image = results[0].plot(
            conf=self.result_conf,
            line_width=self.result_line_width,
            font_size=self.result_font_size,
            font=self.result_font,
            labels=self.result_labels,
            boxes=self.result_boxes,
        )
        result_image_msg = self.bridge.cv2_to_imgmsg(plotted_image, encoding="bgr8")
        return result_image_msg

if __name__ == "__main__":
    rospy.init_node("tracker_node")
    node = TrackerNode()
    node.start()  # Start the image capture and publishing loop

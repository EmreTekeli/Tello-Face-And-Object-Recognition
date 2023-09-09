import sys
sys.path.append("/tello-face-tracker-main")

from typing import Tuple
import cv2
import math
import time
from djitellopy import Tello
from numpy import array
from ultralytics import YOLO
from cvzone import cornerRect, putTextRect
from face_detection import get_caffe_net, get_most_confident_face


classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]


model = YOLO("yolov8n.pt")

class FaceTracker:
    def __init__(
        self,
        proto_file: str = 'deploy.prototxt.txt',
        model_file: str = 'res10_300x300_ssd_iter_140000.caffemodel',
        image_size: Tuple[int, int] = (800, 600)
    ) -> None:
        self.tello = Tello()

        self.tello.connect()
        self.tello.streamon()

        print('Battery: ', self.tello.get_battery())

        self.tello.takeoff()
        self.tello.send_rc_control(0, 0, 0, 0)

        self.net = get_caffe_net(
            proto_file=proto_file,
            model_file=model_file
        )

        self.image_size = image_size

    def get_frame(self) -> array:
        image = self.tello.get_frame_read().frame


        image = cv2.resize(image, self.image_size)

        return image

    def get_rc_controls(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int
    ) -> Tuple[int, int, int, int]:
        left_right_velocity = 0
        forward_backward_velocity = self.get_forward_backward_velocity(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2
        )

        if forward_backward_velocity == 0:
            up_down_velocity = self.get_up_down_velocity(
                y1=y1,
                y2=y2
            )
        else:
            up_down_velocity = 0

        yaw_velocity = self.get_yaw_velocity(
            x1=x1,
            x2=x2
        )

        return (
            left_right_velocity,
            forward_backward_velocity,
            up_down_velocity,
            yaw_velocity
        )

    def get_forward_backward_velocity(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int
    ) -> int:
        height = y2 - y1
        width = x2 - x1
        area = height * width

        if (area == 0) or (6000 < area < 17000):
            forward_backward_velocity = 0
        elif area <= 6000:
            forward_backward_velocity = 20
        else:
            forward_backward_velocity = -20

        return forward_backward_velocity

    def get_up_down_velocity(
        self,
        y1: int,
        y2: int
    ) -> int:
        y_mid = (y1 + y2) // 2

        if (y_mid == 0) or (70 < y_mid < 135):
            up_down_velocity = 0
        elif y_mid <= 70:
            up_down_velocity = 15
        else:
            up_down_velocity = -15

        return up_down_velocity

    def get_yaw_velocity(
        self,
        x1: int,
        x2: int
    ) -> int:
        x_mid = (x1 + x2) // 2

        if (x_mid == 0) or (120 < x_mid < 180):
            yaw_velocity = 0
        elif x_mid <= 120:
            yaw_velocity = -25
        else:
            yaw_velocity = 25

        return yaw_velocity

    def track_face(self) -> None:
        prev_frame_time = 0
        while True:
            new_frame_time = time.time()
            image = self.get_frame()
            results = model(image, stream=True)

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1
                    cornerRect(image, (x1, y1, w, h))

                    imgCrop = image[y1:y1 + h, x1:x1 + w]
                    imgBlur = cv2.blur(imgCrop, (35, 35))
                    image[y1:y1 + h, x1:x1 + w] = imgBlur

                    conf = math.ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])

                    putTextRect(image, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2


                    image_width = self.image_size[0]
                    image_height = self.image_size[1]
                    mid_x = image_width // 2
                    mid_y = image_height // 2


                    if cx < mid_x - 50:
                        self.tello.send_rc_control(-20, 0, 0, 0)
                    elif cx > mid_x + 50:
                        self.tello.send_rc_control(20, 0, 0, 0)
                    else:
                        self.tello.send_rc_control(0, 0, 0, 0)


                    if cy < mid_y - 30:
                        self.tello.send_rc_control(0, 0, 20, 0)
                    elif cy > mid_y + 30:
                        self.tello.send_rc_control(0, 0, -20, 0)
                    else:
                        self.tello.send_rc_control(0, 0, 0, 0)

            self.tello.send_rc_control(0, 0, 0, 0)

            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            print("fps: ", fps)

            cv2.imshow("Yüz Tanıma/Takip Sistemi", image)
            cv2.waitKey(1)

if __name__ == "__main__":

    new_image_size = (800, 600)


    face_tracker = FaceTracker(image_size=new_image_size)
    face_tracker.track_face()

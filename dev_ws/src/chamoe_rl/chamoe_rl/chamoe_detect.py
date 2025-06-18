#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import numpy as np
import cv2
from ultralytics import YOLO

class ChamoeDetect(Node):
    def __init__(self):
        super().__init__('chamoe_detect')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            CompressedImage,
            '/color_camera/color_camera/image_raw/compressed',
            self.image_callback,
            10
        )
        self.publisher = self.create_publisher(
            Float32MultiArray,
            '/chamoe_detect/bbox_xy',
            10
        )
        self.model = YOLO('/home/dockeruser/y11n_gz_chamoebest.pt')
        self.get_logger().info('YOLOv11 model loaded and chamoe_detect node started.')

    def image_callback(self, msg):
        # CompressedImage -> OpenCV 이미지 변환
        np_arr = np.frombuffer(msg.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            self.get_logger().warn('Failed to decode image')
            return
        # YOLO 추론
        results = self.model(image)
        found = False
        x_center, y_center = -999.0, -999.0
        x1, y1, x2, y2 = -999.0, -999.0, -999.0, -999.0
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                if cls == 0:  # chamoe 클래스 인덱스
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    """x_center = (x1 + x2) / 2
                    y_center = (y1 + y2) / 2"""
                    # 사각형 그리기 및 라벨 출력
                    cv2.rectangle(
                        image,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (0, 255, 0),
                        2
                    )
                    cv2.putText(image, 'Chamoe', (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                
                    found = True
                    break
            if found:
                break
        # Publish
        msg_out = Float32MultiArray()
        msg_out.data = [x1, y1, x2, y2]
        self.publisher.publish(msg_out)
        self.get_logger().debug(f'Published chamoe bbox: {x1}, {y1}, {x2}, {y2}')
        # 실시간 이미지 출력
        cv2.imshow('Chamoe Detection', image)
        cv2.waitKey(1)  # 필요 (콜백마다 갱신)

def main(args=None):
    rclpy.init(args=args)
    node = ChamoeDetect()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 
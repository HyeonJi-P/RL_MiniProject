import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class DepthVisualizer(Node):
    def __init__(self):
        super().__init__('depth_visualizer')
        self.bridge = CvBridge()
        self.depth_sub = self.create_subscription(
            Image,
            '/depth_camera/depth_camera/depth/image_raw',
            self.depth_callback,
            10
        )
        self.get_logger().info('Depth Visualizer Node Started')

    def depth_callback(self, msg):
        # depth 이미지를 numpy array로 변환
        depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        # 시각화용 이미지 생성 (0~255로 정규화)
        vis_img = np.clip(depth_img, 0.028, 3.0)
        vis_img = (vis_img - 0.028) / (3.0 - 0.028) * 255
        vis_img = vis_img.astype(np.uint8)
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2BGR)

        # ROI 좌표 (예시: 이미지 하단 중앙 부분)
        x1, y1 = 200, 100
        x2, y2 = 1100, 360

        # ROI 내에서만 유효한 값 찾기
        roi = depth_img[y1:y2, x1:x2]
        valid = (roi > 0.0) & (roi < 10) & np.isfinite(roi)
        if np.any(valid):
            min_val = np.min(roi[valid])
            min_idx_roi = np.unravel_index(np.argmin(np.where(valid, roi, np.inf)), roi.shape)
            # ROI 내 좌표를 전체 이미지 좌표로 변환
            min_idx = (min_idx_roi[0] + y1, min_idx_roi[1] + x1)
            # ROI 네모 그리기
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0,255,0), 2)
            # ROI 내 최소값 위치에 빨간 원 표시
            cv2.circle(vis_img, (min_idx[1], min_idx[0]), 8, (0,0,255), 2)
            cv2.putText(vis_img, f"min: {min_val:.2f}m", (min_idx[1], min_idx[0]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            cv2.imshow("Depth Min Visualizer", vis_img)
            cv2.waitKey(1)
            print(f"ROI shape: {roi.shape}")
            print(f"ROI min: {np.min(roi)}, max: {np.max(roi)}")
            print(f"ROI unique values: {np.unique(roi)[:10]} ...")
            print(f"ROI finite count: {np.sum(np.isfinite(roi))}")
            print(f"ROI >0 count: {np.sum(roi > 0)}")
        else:
            # ROI 네모만 그리기
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.imshow("Depth Min Visualizer", vis_img)
            cv2.waitKey(1)
            print("No valid depth values in ROI.")

def main(args=None):
    rclpy.init(args=args)
    node = DepthVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
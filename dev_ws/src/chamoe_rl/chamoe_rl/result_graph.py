#!/usr/bin/env python3
import sys
import threading
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow
import pyqtgraph
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

class GraphSubscriber(Node):
    def __init__(self, window):
        super().__init__('chamoe_rl_result_graph')
        self.window = window
        self.subscription = self.create_subscription(
            Float32MultiArray, '/result', self.data_callback, 10
        )
    def data_callback(self, msg):
        self.window.receive_data(msg)

class Window(QMainWindow):
    def __init__(self):
        super(Window, self).__init__()
        self.setWindowTitle('Result')
        self.setGeometry(50, 50, 600, 650)
        self.ep = []
        self.data_list = []
        self.rewards = []
        self.count = 1
        self.plot()
        self.ros_subscriber = GraphSubscriber(self)
        self.ros_thread = threading.Thread(
            target=rclpy.spin, args=(self.ros_subscriber,), daemon=True
        )
        self.ros_thread.start()
    def receive_data(self, msg):
        self.data_list.append(msg.data[0])
        self.ep.append(self.count)
        self.count += 1
        self.rewards.append(msg.data[1])
    def plot(self):
        self.qValuePlt = pyqtgraph.PlotWidget(self, title='Average max Q-value')
        self.qValuePlt.setGeometry(0, 320, 600, 300)
        self.rewardsPlt = pyqtgraph.PlotWidget(self, title='Total reward')
        self.rewardsPlt.setGeometry(0, 10, 600, 300)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(200)
        self.show()
    def update(self):
        self.rewardsPlt.showGrid(x=True, y=True)
        self.qValuePlt.showGrid(x=True, y=True)
        self.rewardsPlt.plot(self.ep, self.data_list, pen=(255, 0, 0), clear=True)
        self.qValuePlt.plot(self.ep, self.rewards, pen=(0, 255, 0), clear=True)
    def closeEvent(self, event):
        if self.ros_subscriber is not None:
            self.ros_subscriber.destroy_node()
        rclpy.shutdown()
        event.accept()

def main():
    rclpy.init()
    app = QApplication(sys.argv)
    win = Window()
    sys.exit(app.exec())

if __name__ == '__main__':
    main() 
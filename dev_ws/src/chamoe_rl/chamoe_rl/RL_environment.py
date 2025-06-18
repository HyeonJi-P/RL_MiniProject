#!/usr/bin/env python3
#################################################################################
# Copyright 2019 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################
#
# Authors: Ryan Shim, Gilbert, ChanHyeong Lee

import math

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import numpy
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_srvs.srv import Empty
from std_msgs.msg import Float32MultiArray

from turtlebot3_msgs.srv import Dqn
from turtlebot3_msgs.srv import Goal


class RLEnvironment(Node):

    def __init__(self):
        super().__init__('rl_environment')
        self.bridge = CvBridge()
        self.depth_image = None
        self.depth_image_sub = self.create_subscription(
            Image,
            '/depth_camera/depth_camera/depth/image_raw',
            self.depth_image_sub_callback,
            qos_profile_sensor_data
        )

        self.min_obstacle_distance = float('inf')
 
        self.train_mode = True
        self.goal_pose_x = 0.0
        self.goal_pose_y = 0.0
        self.robot_pose_x = 0.0
        self.robot_pose_y = 0.0
        self.robot_linear_vel = 0.0
        self.robot_angular_vel = 0.0
        self.is_near = False
        self.chamoe_bbox = [-999.0, -999.0, -999.0, -999.0]
        self.chamoe_center = [-999.0, -999.0]
        
        self.action_size = 6
        self.max_step = 800

        self.done = False
        self.fail = False
        self.succeed = False

        self.goal_angle = 0.0
        self.goal_distance = 1.0
        #self.init_goal_distance = 0.5
        
        self.local_step = 0
        self.stop_cmd_vel_timer = None
        self.angular_vel = [1.5, 0.75, 0.0, -0.75, -1.5, 0.0]
        self.linear_vel = [0.15, 0.15, 0.15, 0.15, 0.15, 0.0]

        qos = QoSProfile(depth=10)

        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', qos)

        self.odom_sub = self.create_subscription(
            Odometry,
            'odom',
            self.odom_sub_callback,
            qos
        )
        

        self.clients_callback_group = MutuallyExclusiveCallbackGroup()
        self.task_succeed_client = self.create_client(
            Goal,
            'task_succeed',
            callback_group=self.clients_callback_group
        )
        self.task_failed_client = self.create_client(
            Goal,
            'task_failed',
            callback_group=self.clients_callback_group
        )
        self.initialize_environment_client = self.create_client(
            Goal,
            'initialize_env',
            callback_group=self.clients_callback_group
        )

        self.rl_agent_interface_service = self.create_service(
            Dqn,
            'rl_agent_interface',
            self.rl_agent_interface_callback
        )
        self.make_environment_service = self.create_service(
            Empty,
            'make_environment',
            self.make_environment_callback
        )
        self.reset_environment_service = self.create_service(
            Dqn,
            'reset_environment',
            self.reset_environment_callback
        )

        self.chamoe_center = [-999.0, -999.0]

        self.chamoe_center_sub = self.create_subscription(
            Float32MultiArray,
            '/chamoe_detect/bbox_xy',
            self.chamoe_center_callback,
            10
        )

    def make_environment_callback(self, request, response):
        while not self.initialize_environment_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(
                'service for initialize the environment is not available, waiting ...'
            )
        future = self.initialize_environment_client.call_async(Goal.Request())
        rclpy.spin_until_future_complete(self, future)
        response_goal = future.result()
        if not response_goal.success:
            self.get_logger().error('initialize environment request failed')
        else:
            self.goal_pose_x = response_goal.pose_x
            self.goal_pose_y = response_goal.pose_y
            self.get_logger().info(
                'goal initialized at [%f, %f]' % (self.goal_pose_x, self.goal_pose_y)
            )

        return response

    def reset_environment_callback(self, request, response):
        state = self.calculate_state()
        #self.init_goal_distance = state[0]
        #self.prev_goal_distance = self.init_goal_distance
        response.state = state

        return response

    def call_task_succeed(self):
        while not self.task_succeed_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('service for task succeed is not available, waiting ...')
        future = self.task_succeed_client.call_async(Goal.Request())
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            response = future.result()
            self.goal_pose_x = response.pose_x
            self.goal_pose_y = response.pose_y
            self.get_logger().info('service for task succeed finished')
        else:
            self.get_logger().error('task succeed service call failed')

    def call_task_failed(self):
        while not self.task_failed_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('service for task failed is not available, waiting ...')
        future = self.task_failed_client.call_async(Goal.Request())
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            response = future.result()
            self.goal_pose_x = response.pose_x
            self.goal_pose_y = response.pose_y
            self.get_logger().info('service for task failed finished')
        else:
            self.get_logger().error('task failed service call failed')

    def depth_image_sub_callback(self, msg: Image):
        # 32FC1 single-channel depth map으로 변환
        depth_array = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        # ROI 좌표 설정
        x1, y1 = 200, 100
        x2, y2 = 1100, 360
        roi = depth_array[y1:y2, x1:x2]
        # 유효 거리만 필터링 (0 < depth < inf)
        valid = roi[(roi > 0.0) & numpy.isfinite(roi)]
        if valid.size > 0:
            self.min_obstacle_distance = float(numpy.min(valid))
        else:
            self.min_obstacle_distance = float('inf')
        self.get_logger().debug(f"collision__min_obstacle_distance = {self.min_obstacle_distance:.3f}")

    def odom_sub_callback(self, msg):
        self.robot_pose_x = msg.pose.pose.position.x
        self.robot_pose_y = msg.pose.pose.position.y
        self.robot_linear_vel = msg.twist.twist.linear.x
        self.robot_angular_vel = msg.twist.twist.angular.z
        _, _, self.robot_pose_theta = self.euler_from_quaternion(msg.pose.pose.orientation)

        goal_distance = math.sqrt(
            (self.goal_pose_x - self.robot_pose_x) ** 2
            + (self.goal_pose_y - self.robot_pose_y) ** 2)
        path_theta = math.atan2(
            self.goal_pose_y - self.robot_pose_y,
            self.goal_pose_x - self.robot_pose_x)

        goal_angle = path_theta - self.robot_pose_theta
        if goal_angle > math.pi:
            goal_angle -= 2 * math.pi

        elif goal_angle < -math.pi:
            goal_angle += 2 * math.pi

        self.goal_distance = goal_distance
        self.goal_angle = goal_angle

    def calculate_state(self):
        state = []
        """state.append(float(self.goal_distance))
        state.append(float(self.goal_angle))
        
        state.append(self.min_obstacle_distance)
        print(f"[calculate_state] state: {state}")"""
        self.local_step += 1

        distance_to_chamoe = math.sqrt(
            (self.goal_pose_x - self.robot_pose_x) ** 2
            + (self.goal_pose_y - self.robot_pose_y) ** 2)
        is_near = distance_to_chamoe < 0.35 and distance_to_chamoe > 0.25
        is_stopped = abs(self.robot_linear_vel) < 0.02 and abs(self.robot_angular_vel) < 0.02
        self.is_near = is_near  # 참외 근처 여부 저장

        if is_near and is_stopped:
            self.get_logger().info('Goal Reached and Stopped')
            self.succeed = True
            self.done = True
            self.cmd_vel_pub.publish(Twist())
            self.local_step = 0
            self.call_task_succeed()

        if self.min_obstacle_distance < 0.15:
            self.get_logger().info('Collision happened')
            self.fail = True
            self.done = True
            self.cmd_vel_pub.publish(Twist())
            self.local_step = 0
            self.call_task_failed()

        if self.local_step == self.max_step:
            self.get_logger().info('Time out!')
            self.fail = True
            self.done = True
            self.cmd_vel_pub.publish(Twist())
            self.local_step = 0
            self.call_task_failed()

        # chamoe_center를 그대로 state로 사용 + 장애물 거리 추가
        #state = self.chamoe_bbox.copy()
        state = self.chamoe_center.copy()
        state.append(self.min_obstacle_distance)
        print(f'[calculate_state] state: {state}')

        return state

    def calculate_reward(self):
        if self.train_mode:
            # 화면 크기 
            center_x = 160  # 예시: 320*240 이미지라면
            x, y = self.chamoe_center

            align_reward = 0.0
            not_found_reward = 0.0  # 반드시 기본값으로 초기화

            if x == -999.0 or y == -999.0:
                not_found_reward = -5.0
            else:
                if abs(x - center_x) <= 5:
                    if abs(self.robot_angular_vel)<=0.75: #직진 혹은 좌우 회전 중 하나만 있을 때
                        align_reward = 20.0  # 중앙 수직선 근처면 보상
                    else:
                        align_reward = max(10.0 - 3 * (abs(self.robot_linear_vel)+abs(self.robot_angular_vel)), 0.0)  # 속도가 빠를수록 보상감소
                else:
                    align_reward = max(5.0 - 0.01 * abs(x - center_x), 0.0)  # x축 중앙에서 멀어질수록 보상 감소

            #yaw_reward = (1 - 2 * math.sqrt(math.fabs(self.goal_angle / math.pi)))

            obstacle_reward = 0.0
            if self.min_obstacle_distance < 0.30:
                obstacle_reward = -10.0

            if self.min_obstacle_distance < 0.20:
                self.fail = True
                self.done = True
                
            # 시간 패널티 추가 (예: step마다 -0.01)
            time_penalty = -0.1 # 1?  

            collision_penalty = 0.0
            distance_to_chamoe = math.sqrt(
                (self.goal_pose_x - self.robot_pose_x) ** 2
                + (self.goal_pose_y - self.robot_pose_y) ** 2)
            if distance_to_chamoe < 0.10:  # 10cm 이내면 충돌로 간주
                collision_penalty = -500.0  # 원하는 패널티 값
                self.get_logger().info('Collision with chamoe!')

            reward = not_found_reward + align_reward + obstacle_reward + time_penalty + collision_penalty
            if self.is_near:
                reward += max(0, 2.0 - abs(self.robot_linear_vel)*10)

            if self.succeed:
                reward = 1500.0
            elif self.fail:
                reward = -1000.0

            print(f'[calculate_reward] reward: {reward, not_found_reward, align_reward, obstacle_reward, time_penalty, collision_penalty}')
        else:
            if self.succeed:
                reward = 5.0
            elif self.fail:
                reward = -5.0
            else:
                reward = 0.0

            print(f'[calculate_reward] reward: {reward}')

        return reward

    def rl_agent_interface_callback(self, request, response):
        action = request.action
        if not (0 <= action < len(self.linear_vel)):
            self.get_logger().error(f"Received invalid action index: {action}")
            response.state = self.calculate_state()
            response.reward = -1000.0  # or another penalty
            response.done = True
            return response
        twist = Twist()
        twist.linear.x = self.linear_vel[action]
        twist.angular.z = self.angular_vel[action]
        self.cmd_vel_pub.publish(twist)
        if self.stop_cmd_vel_timer is None:
            #self.prev_goal_distance = self.init_goal_distance
            self.stop_cmd_vel_timer = self.create_timer(1.8, self.timer_callback)
        else:
            self.destroy_timer(self.stop_cmd_vel_timer)
            self.stop_cmd_vel_timer = self.create_timer(1.8, self.timer_callback)

        response.state = self.calculate_state()
        
        response.reward = self.calculate_reward()
        response.done = self.done

        if self.done is True:
            self.done = False
            self.succeed = False
            self.fail = False

        return response

    def timer_callback(self):
        self.get_logger().info('Stop called')
        self.cmd_vel_pub.publish(Twist())
        self.destroy_timer(self.stop_cmd_vel_timer)

    def euler_from_quaternion(self, quat):
        x = quat.x
        y = quat.y
        z = quat.z
        w = quat.w

        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = numpy.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        pitch = numpy.arcsin(sinp)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = numpy.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def chamoe_center_callback(self, msg):
        if len(msg.data) == 4:
            if msg.data[0] != -999.0 and msg.data[1] != -999.0 and msg.data[2] != -999.0 and msg.data[3] != -999.0:
                self.chamoe_bbox = [msg.data[0], msg.data[1], msg.data[2], msg.data[3]]
                self.chamoe_center = [(msg.data[0] + msg.data[2]) / 2, (msg.data[1] + msg.data[3]) / 2]
            else:
                self.chamoe_bbox = [-999.0, -999.0, -999.0, -999.0]
                self.chamoe_center = [-999.0, -999.0]
        else:
            self.chamoe_bbox = [-999.0, -999.0, -999.0, -999.0]
            self.chamoe_center = [-999.0, -999.0]


def main(args=None):
    rclpy.init(args=args)
    rl_environment = RLEnvironment()
    try:
        while rclpy.ok():
            rclpy.spin_once(rl_environment, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        rl_environment.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

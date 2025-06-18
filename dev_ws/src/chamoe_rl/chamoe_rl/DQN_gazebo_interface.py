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
# # Authors: Ryan Shim, Gilbert, ChanHyeong Lee

import os
import random
import sys
import time
import math

from ament_index_python.packages import get_package_share_directory
from gazebo_msgs.srv import DeleteEntity
from gazebo_msgs.srv import SpawnEntity
from geometry_msgs.msg import Pose
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from std_srvs.srv import Empty
from gazebo_msgs.msg import ModelStates

from turtlebot3_msgs.srv import Goal

class GazeboInterface(Node):
    def __init__(self):
        super().__init__('gazebo_interface_chamoe')
        self.stage = int(1)
        self.entity_name = 'ChamoeGoal'
        self.entity = None
        self.open_entity()
        self.entity_pose_x = 0.5
        self.entity_pose_y = 0.0
        self.delete_entity_client = self.create_client(DeleteEntity, 'delete_entity')
        self.spawn_entity_client = self.create_client(SpawnEntity, 'spawn_entity')
        self.reset_simulation_client = self.create_client(Empty, 'reset_simulation')
        self.callback_group = MutuallyExclusiveCallbackGroup()
        self.initialize_env_service = self.create_service(
            Goal,
            'initialize_env',
            self.initialize_env_callback,
            callback_group=self.callback_group
        )
        self.task_succeed_service = self.create_service(
            Goal,
            'task_succeed',
            self.task_succeed_callback,
            callback_group=self.callback_group
        )
        self.task_failed_service = self.create_service(
            Goal,
            'task_failed',
            self.task_failed_callback,
            callback_group=self.callback_group
        )
        self.chamoe_pose = None
        self.model_states_sub = self.create_subscription(
            ModelStates, '/gazebo/model_states', self.model_states_callback, 10
        )
    def open_entity(self):
        try:
            package_share = get_package_share_directory('turtlebot3_gazebo')
            # chamoe 모델을 사용하도록 경로 변경
            model_path = os.path.join(
                package_share, 'models', 'chamoe', 'model.sdf'
            )
            with open(model_path, 'r') as f:
                self.entity = f.read()
            self.get_logger().info('Loaded chamoe entity from: ' + model_path)
        except Exception as e:
            self.get_logger().error('Failed to load chamoe entity file: {}'.format(e))
            raise e
    def reset_simulation(self):
        reset_req = Empty.Request()
        while not self.reset_simulation_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('service for reset_simulation is not available, waiting ...')
        self.reset_simulation_client.call_async(reset_req)
    def delete_entity(self):
        delete_req = DeleteEntity.Request()
        delete_req.name = self.entity_name
        while not self.delete_entity_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('service for delete_entity is not available, waiting ...')
        future = self.delete_entity_client.call_async(delete_req)
        rclpy.spin_until_future_complete(self, future)
        self.get_logger().info('A chamoe goal deleted.')
    def spawn_entity(self):
        entity_pose = Pose()
        entity_pose.position.x = self.entity_pose_x
        entity_pose.position.y = self.entity_pose_y
        spawn_req = SpawnEntity.Request()
        spawn_req.name = self.entity_name
        spawn_req.xml = self.entity
        spawn_req.initial_pose = entity_pose
        while not self.spawn_entity_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('service for spawn_entity is not available, waiting ...')
        future = self.spawn_entity_client.call_async(spawn_req)
        rclpy.spin_until_future_complete(self, future)
    def task_succeed_callback(self, request, response):
        self.delete_entity()
        time.sleep(0.2)
        self.reset_simulation()  # 성공 시에도 시뮬레이터 전체 리셋
        time.sleep(0.2)
        self.generate_goal_pose()
        time.sleep(0.2)
        self.spawn_entity()
        response.pose_x = self.entity_pose_x
        response.pose_y = self.entity_pose_y
        response.success = True
        self.get_logger().info('A new chamoe goal generated (with reset).')
        return response
    def task_failed_callback(self, request, response):
        self.delete_entity()
        time.sleep(0.2)
        self.reset_simulation()
        time.sleep(0.2)
        self.generate_goal_pose()
        time.sleep(0.2)
        self.spawn_entity()
        response.pose_x = self.entity_pose_x
        response.pose_y = self.entity_pose_y
        response.success = True
        self.get_logger().info('Environment reset (chamoe goal)')
        return response
    def initialize_env_callback(self, request, response):
        self.delete_entity()
        time.sleep(0.2)
        self.reset_simulation()
        time.sleep(0.2)
        self.spawn_entity()
        response.pose_x = self.entity_pose_x
        response.pose_y = self.entity_pose_y
        response.success = True
        self.get_logger().info('Environment initialized (chamoe goal)')
        return response
    def generate_goal_pose(self): # 참외 스폰 위치 고정
        self.entity_pose_x = 2.0  # 원하는 x좌표
        self.entity_pose_y = -1.5  # 원하는 y좌표
        """
        if self.stage != 4:
            self.entity_pose_x = random.randrange(-23, 23) / 10
            self.entity_pose_y = random.randrange(-23, 23) / 10
        else:
            goal_pose_list = [
                [1.0, 0.0], [2.0, -1.5], [0.0, -2.0], [2.0, 2.0], [0.8, 2.0], [-1.9, 1.9],
                [-1.9, 0.2], [-1.9, -0.5], [-2.0, -2.0], [-0.5, -1.0], [-0.5, 2.0], [2.0, -0.5]
            ]
            rand_index = random.randint(0, 11)
            self.entity_pose_x = goal_pose_list[rand_index][0]
            self.entity_pose_y = goal_pose_list[rand_index][1]"""
    def model_states_callback(self, msg):
        if 'ChamoeGoal' in msg.name:
            idx = msg.name.index('ChamoeGoal')
            self.chamoe_pose = msg.pose[idx]
    def calculate_reward(self):
        # chamoe와의 거리 계산
        if self.chamoe_pose is not None:
            chamoe_dist = math.sqrt(
                (self.entity_pose_x - self.chamoe_pose.position.x) ** 2 +
                (self.entity_pose_y - self.chamoe_pose.position.y) ** 2
            )
            if chamoe_dist < 0.2:
                reward = 100.0
                return reward
        # 장애물 충돌 판정(기존 방식)
        if self.min_obstacle_distance < 0.2:
            reward = -100.0
            return reward
        # ... 나머지 보상 ...
def main(args=None):
    rclpy.init(args=sys.argv)
    gazebo_interface = GazeboInterface()
    try:
        while rclpy.ok():
            rclpy.spin_once(gazebo_interface, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        gazebo_interface.destroy_node()
        rclpy.shutdown()
if __name__ == '__main__':
    main() 
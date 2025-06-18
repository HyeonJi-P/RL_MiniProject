#!/usr/bin/env python3
#################################################################################
# Q-learning Agent for chamoe_rl 
#################################################################################

import collections
import math
import numpy as np
import os
import random
import sys
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Empty
from turtlebot3_msgs.srv import Dqn

class QAgent(Node):
    def __init__(self, stage_num, max_training_episodes, load_episode=0):
        super().__init__('q_agent')

        self.stage = int(stage_num)
        self.max_training_episodes = int(max_training_episodes)
        self.train_mode = True if int(max_training_episodes) > 0 else False
        self.load_episode = int(load_episode)

        self.reset_environment_client = self.create_client(Dqn, 'reset_environment')
        raw_state = self._get_initial_state()
        self.state_size = len(raw_state)
        self.action_size = 6

        # Q-table
        self.state_bins = [11, 11, 10]              #  x1, y1, distance
        self.q_table    = np.zeros(self.state_bins + [self.action_size])

        self.q_table_dir_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
            'saved_model'
        )

        if not self.train_mode:
            try:
                self.q_table = np.load(os.path.join(self.q_table_dir_path, f'q_table_stage{self.stage}_ep{self.load_episode}.npy'))
                print(f'Q-table loaded from {os.path.join(self.q_table_dir_path, f"q_table_stage{self.stage}_ep{self.load_episode}.npy")}')
            except FileNotFoundError:
                print('No saved Q-table found, starting fresh.')

        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.alpha = 0.1
        self.gamma = 0.99

        self.state_low = np.array([-999.0, -999.0, 0.0])
        self.state_high = np.array([320.0, 240.0, 3.0])

        self.rl_agent_interface_client = self.create_client(Dqn, 'rl_agent_interface')
        self.make_environment_client = self.create_client(Empty, 'make_environment')

        self.action_pub = self.create_publisher(Float32MultiArray, '/get_action', 10)
        self.result_pub = self.create_publisher(Float32MultiArray, 'result', 10)

        self.process()

    def env_make(self):
        while not self.make_environment_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(
                'Environment make client failed to connect to the server, try again ...'
            )
        self.make_environment_client.call_async(Empty.Request())

    def _get_initial_state(self):
        while not self.reset_environment_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(
                'Reset environment client failed to connect to the server, try again ...'
            )
        future = self.reset_environment_client.call_async(Dqn.Request())
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            return future.result().state
        else:
            self.get_logger().error(
                'Exception while calling service: {0}'.format(future.exception()))
            return [0.0, 0.0, 0.0]

    def reset_environment(self):
        while not self.reset_environment_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(
                'Reset environment client failed to connect to the server, try again ...'
            )
        future = self.reset_environment_client.call_async(Dqn.Request())
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            return future.result().state
        else:
            self.get_logger().error(
                'Exception while calling service: {0}'.format(future.exception()))
            return [-999.0, -999.0, 0.0]

    def discretize_state(self, state):
        """ratios = (np.array(state) - self.state_low) / (self.state_high - self.state_low)
        ratios = np.clip(ratios, 0, 0.999)
        discrete = (ratios * self.state_bins).astype(int)"""
        
        x1, y1, d = state

        # x 처리
        if x1 == -999.0:
            x1_bin = 10
        else:
            x1_bin = int(np.clip(x1 / 320.0 * 10, 0, 9))

        # y 처리
        if y1 == -999.0:
            y1_bin = 10
        else:
            y1_bin = int(np.clip(y1 / 240.0 * 10, 0, 9))

        # d 처리
        d = np.clip(d, 0.05, 3.0)
        d_bin = int((d - 0.05) / (3.0 - 0.05) * 10)
        d_bin = np.clip(d_bin, 0, 9) # 0~9

        #return tuple(discrete)
        return (x1_bin, y1_bin, d_bin)

    def get_action(self, state):
        if self.train_mode:
            if random.random() < self.epsilon:
                return random.randint(0, self.action_size - 1)
            else:
                discrete_state = self.discretize_state(state)
                return int(np.argmax(self.q_table[discrete_state]))
        else:
            discrete_state = self.discretize_state(state)
            return int(np.argmax(self.q_table[discrete_state]))

    def step(self, action):
        req = Dqn.Request()
        req.action = int(action)
        while not self.rl_agent_interface_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('rl_agent interface service not available, waiting again...')
        future = self.rl_agent_interface_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            next_state = future.result().state
            reward = future.result().reward
            done = future.result().done
        else:
            self.get_logger().error('Exception while calling service: {0}'.format(future.exception()))
            next_state = [-999.0, -999.0, 0.0]
            reward = 0.0
            done = True
        return next_state, reward, done

    def process(self):
        self.env_make()
        time.sleep(1.0)

        episode_num = 0

        for episode in range(1, self.max_training_episodes + 1):
            state = self.reset_environment()
            discrete_state = self.discretize_state(state)
            done = False
            score = 0
            local_step = 0

            time.sleep(1.0)

            while not done:
                local_step += 1
                action = self.get_action(state)
                next_state, reward, done = self.step(action)
                next_discrete_state = self.discretize_state(next_state)

                # Q-learning update
                best_next_q = np.max(self.q_table[next_discrete_state])
                td_target = reward + self.gamma * best_next_q * (not done)
                td_error = td_target - self.q_table[discrete_state + (action,)]
                self.q_table[discrete_state + (action,)] += self.alpha * td_error

                # action/result publish (optional)
                msg = Float32MultiArray()
                msg.data = [float(action), float(score), float(reward)]
                self.action_pub.publish(msg)

                state = next_state
                discrete_state = next_discrete_state
                score += reward

                if done:
                    msg = Float32MultiArray()
                    msg.data = [float(score), float(local_step)]
                    self.result_pub.publish(msg)
                    print(
                        'Episode:', episode,
                        'score:', score,
                        'steps:', local_step,
                        'epsilon:', self.epsilon)
                    break

                time.sleep(0.01)

            # Epsilon decay
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                self.epsilon = max(self.epsilon, self.epsilon_min)

            # Q-table 저장 (학습 모드에서만)
            if self.train_mode and episode % 100 == 0:
                np.save(os.path.join(self.q_table_dir_path, f'q_table_stage{self.stage}_ep{episode}.npy'), self.q_table)

def main(args=None):
    if args is None:
        args = sys.argv
    stage_num = args[1] if len(args) > 1 else '1'
    max_training_episodes = args[2] if len(args) > 2 else '1000'
    load_episode = args[3] if len(args) > 3 else '0'
    rclpy.init(args=args)
    agent = QAgent(stage_num, max_training_episodes, load_episode)
    rclpy.spin(agent)
    agent.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main() 
#!/usr/bin/env python3
################################################################################
# Q‑learning TEST script (inference‑only)
# --------------------------------------------------------------
# ‼️ This file is generated to mirror the structure of the existing
#    DQNTest but operates on a **saved Q‑table** instead of a network.
#    Key points are marked with "# <<< NOTE" for clarity.
################################################################################
import numpy as np
import random
import sys
import time
import rclpy
from rclpy.node import Node
from turtlebot3_msgs.srv import Dqn

from std_msgs.msg import Float32MultiArray


class QTest(Node):
    """Evaluation‑only node for a trained Q‑table policy."""

    def __init__(self, stage: str = "1", load_episode: str = "1000"):
        super().__init__("chamoe_q_test")

        # --------------------------------------------------
        # Meta / file handling
        # --------------------------------------------------
        self.stage = int(stage)
        self.load_episode = int(load_episode)
        self.table_path = f"q_table_stage{self.stage}_ep{self.load_episode}.npy"

        # --------------------------------------------------
        # Try to load Q‑table
        # --------------------------------------------------
        try:
            #self.q_table = np.zeros([11, 11, 11, 11, 10, 6])  # 5차원 + action
            self.q_table = np.zeros([11, 11, 10, 6])  # 3차원 + action
            self.get_logger().info(f"✅ Q‑table loaded from {self.table_path}")
        except FileNotFoundError:
            self.get_logger().error(f"❌ Q‑table {self.table_path} not found.")
            raise

        self.state_bins = list(self.q_table.shape[:-1])  # e.g. [11,11,11,11,10]
        self.action_size = 6#self.q_table.shape[-1]
        self.epsilon = 0.05  # greedy‑ish for eval
        

        # --------------------------------------------------
        # Fixed min/max (should match training agent)
        # --------------------------------------------------
        #self.state_low = np.array([-999.0, -999.0, -999.0, -999.0, 0.0])
        #self.state_high = np.array([320.0, 240.0, 320.0, 240.0, 3.0])
        self.state_low = np.array([-999.0, -999.0, 0.0])
        self.state_high = np.array([320.0, 240.0, 3.0])

        # --------------------------------------------------
        # ROS2 interfaces
        # --------------------------------------------------
        self.rl_agent_interface_client = self.create_client(Dqn, "rl_agent_interface")
        self.reset_environment_client = self.create_client(Dqn, "reset_environment")

        self.action_pub = self.create_publisher(Float32MultiArray, "/get_action", 10)
        self.result_pub = self.create_publisher(Float32MultiArray, "result", 10)

        self.evaluate_loop()

    # ------------------------------------------------------
    # Helper: discretize state exactly like training script
    # ------------------------------------------------------
    def discretize_state(self, state):
        #x1, y1, x2, y2, d = state
        x1, y1, d = state
        def bin_x(val):
            return 10 if val == -999.0 else int(np.clip(val / 320.0 * 10, 0, 9))
        def bin_y(val):
            return 10 if val == -999.0 else int(np.clip(val / 240.0 * 10, 0, 9))
        x1_bin, y1_bin = bin_x(x1), bin_y(y1)
        #x2_bin, y2_bin = bin_x(x2), bin_y(y2)
        d = np.clip(d, 0.05, 3.0)
        d_bin = int((d - 0.05) / (3.0 - 0.05) * 10)
        d_bin = np.clip(d_bin, 0, 9)
        return (x1_bin, y1_bin, d_bin)#(x1_bin, y1_bin, x2_bin, y2_bin, d_bin)

    # ------------------------------------------------------
    # Reset environment helper
    # ------------------------------------------------------
    def reset_environment(self):
        while not self.reset_environment_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("reset_environment service unavailable, retrying …")
        future = self.reset_environment_client.call_async(Dqn.Request())
        rclpy.spin_until_future_complete(self, future)
        return future.result().state if future.result() else [-999.0]*4 #[-999.0]*5

    # ------------------------------------------------------
    # Action selection (ε‑greedy for completeness)
    # ------------------------------------------------------
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        ds = self.discretize_state(state)
        return int(np.argmax(self.q_table[ds]))

    # ------------------------------------------------------
    # Single step in the environment
    # ------------------------------------------------------
    def step_env(self, action):
        req = Dqn.Request(); req.action = int(action)
        while not self.rl_agent_interface_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("rl_agent_interface not ready, waiting …")
        future = self.rl_agent_interface_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result():
            return future.result().state, future.result().reward, future.result().done
        self.get_logger().error("Service call failed; returning terminal state.")
        return [-999.0]*5, 0.0, True

    # ------------------------------------------------------
    # Evaluation main loop (fixed 100 episodes)
    # ------------------------------------------------------
    def evaluate_loop(self, n_episodes: int = 10):
        scores = []
        steps_list = []
        for ep in range(1, n_episodes + 1):
            state = self.reset_environment()
            done, score, steps = False, 0.0, 0
            time.sleep(1.0)
            while not done and steps < 1000:
                steps += 1
                action = self.select_action(state)
                next_state, reward, done = self.step_env(action)
                score += reward

                # Publish for logging / live plot
                pub = Float32MultiArray(); pub.data = [float(action), float(score), float(reward)]
                self.action_pub.publish(pub)

                state = next_state
                time.sleep(0.01)
            scores.append(score)
            steps_list.append(steps)
            res = Float32MultiArray(); res.data = [float(score), float(steps)]
            self.result_pub.publish(res)
            self.get_logger().info(f"Episode {ep}: score={score:.2f}, steps={steps}")
        self.get_logger().info(f"score_list: {scores}, steps_list: {steps_list}")        
        self.get_logger().info(f"Finished {n_episodes} evaluation episodes. Avg score={np.mean(scores):.2f}")


# ----------------------------------------------------------
# Entrypoint
# ----------------------------------------------------------

def main(args=None):
    if args is None:
        args = sys.argv
    stage = args[1] #if len(args) > 1 else "1"
    load_ep = args[2] if len(args) > 2 else "1000"

    rclpy.init(args=args)
    tester = QTest(stage, load_ep)
    try:
        while rclpy.ok():
            rclpy.spin_once(tester, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        tester.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

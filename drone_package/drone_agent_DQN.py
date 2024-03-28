import numpy as np
from tensorflow import keras
import math
import cv2
import time
from .submodules.Environment import Environment
from .submodules.AgentMemory import RewardStorage
from .submodules.Control import Control

# ROS2 dependencies
import rclpy
from rclpy.node import Node
from drone_sim_messages.msg import DroneControl
from drone_sim_messages.msg import DroneSensors
from drone_sim_messages.msg import SessionInfo
from drone_sim_messages.msg import DroneStatus
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float32

from .submodules.DQN import Agent

class DroneAgent_DQN(Node):
   def __init__(self, drone_id, agent, control, environment, n_episodes = 10, save_path=None):
      super().__init__('drone_agent_DQN')
      pub_topic = "/" + drone_id + "/cmd"
      self.control_publisher = self.create_publisher(DroneControl, pub_topic, 10)
      timer_period = 0.25 # seconds
      self.timer = self.create_timer(timer_period, self.control_timer_callback)
      self.i = 0
      pub_topic = "/" + drone_id + "/status"
      self.status_publisher = self.create_publisher(DroneStatus, pub_topic, 10)
      #timer_period = 0.25 # seconds
      #self.timer = self.create_timer(timer_period, self.status_timer_callback)
      #self.i = 0
      sub_topic = "/" + drone_id + "/data"
      self.data_sub = self.create_subscription(DroneSensors, sub_topic, self.sensor_listener_callback, 10)
      self.data_sub # prevent unused variable warning
      sub_topic = "/" + drone_id + "/reward"
      self.rwrd_sub = self.create_subscription(Float32, sub_topic, self.reward_listener_callback, 10)
      self.rwrd_sub # prevent unused variable warning
      sub_topic = "/" + drone_id + "/status"
      self.status_sub = self.create_subscription(DroneStatus, sub_topic, self.status_listener_callback, 10)
      self.status_sub # prevent unused variable warning
      sub_topic = "/session/data"
      self.session_sub = self.create_subscription(SessionInfo, sub_topic, self.session_listener_callback, 10)
      self.session_sub # prevent unused variable warning

      self.control_params = control

      self.id = drone_id

      self.n_episodes = n_episodes
      self.episode_count = 1

      self.agent = agent
      self.environment = environment
      self.reward_storage = RewardStorage()

      self.state  = self.environment.reset()
      self.done = self.environment.done
      self.trained = False

      self.save_path = save_path
      return
   
   # Function to send control data
   def control_timer_callback(self):
      if not self.environment.done:
         self.trained = False
         action = self.agent.choose_action(self.state)                  # Choose action
         self.decode_action(action)                                     # Execute action
         time.sleep(0.1)                                                # Wait for changes
         state_, reward, done = self.environment.step()                 # Get observation
         self.agent.remember(self.state, action, reward, state_, done)  # Store experience
         self.state = state_                                            # Store old state
         #self.agent.learn(verbose=2)
         self.reward_storage.add_reward(reward)                         # Save reward
      elif not self.trained:
         self.reward_storage.store_episode_rewards()                    # Save episode rewards
         self.check_for_end()
         print("Preparing the agent...")
         self.agent.learn(verbose=2)                                    # Train the agent
         self.status_timer_callback()                                   # Send status to the simulator
         self.episode_count += 1                                        # Increase episode count
         self.trained = True
         print("Agent ready!")
      return
   
   # Function to send drone agent status
   def status_timer_callback(self):
      print("Sending status...")
      msg = DroneStatus()
      msg.id = self.id
      msg.active = True
      self.status_publisher.publish(msg)
      print("Status sent!")
      return
   
   # Function to get the drone state
   def sensor_listener_callback(self, msg):
      # Direct conversion to CV2 (decode the image)
      np_arr = np.frombuffer(msg.camera_image, np.uint8)
      decoded_image = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
      
      input_size = (84,84) # This should be provided by the sim
      
      resized_image = cv2.resize(decoded_image, input_size) # Resize the image
      normalized_image = resized_image / 255.0              # Normalize pixel values to range [0,1]
      
      state = np.expand_dims(normalized_image, axis=1)
      state = np.transpose(state, (2,0,1))
      self.environment.state = state
      return
   
   # Function to get reward value
   def reward_listener_callback(self, msg):
      self.environment.reward = msg.data
      return
   
   # Function to get drone status
   def status_listener_callback(self, msg):
      self.active = msg.active
      if not self.active:
         self.status_sent = False
      return
   
   def session_listener_callback(self, msg):
      self.active = msg.session_ended
      self.environment.done = msg.session_ended
      return
   
   def check_for_end(self):
      if self.episode_count > self.n_episodes:
         print("Reached episode ", self.episode_count, " out of ", self.n_episodes)
         if self.save_path != None:
            self.agent.save_model()
            self.reward_storage.save_reward_chart(self.save_path+"DQN_rewards.jpg")
         exit()
      return
   
   # Function to decode the action number into action
   def decode_action(self, action, verbose=0):
      msg = DroneControl()
      #FORWARD
      if action == 0:
         if verbose > -1:
            print("Selected action: FORWARD, ", self.agent.choice_maker)
         msg.twist.linear.x = 1.0
         msg.twist.linear.y = 0.0
         msg.twist.linear.z = 0.0
         msg.twist.angular.x = 0.0
         msg.twist.angular.y = 0.0
         msg.twist.angular.z = 0.0
      #BACKWARD
      if action == 1:
         if verbose > -1:
            print("Selected action: BACKWARD, ", self.agent.choice_maker)
         msg.twist.linear.x = -1.0
         msg.twist.linear.y = 0.0
         msg.twist.linear.z = 0.0
         msg.twist.angular.x = 0.0
         msg.twist.angular.y = 0.0
         msg.twist.angular.z = 0.0
      #LEFT
      if action == 2:
         if verbose > -1:
            print("Selected action: LEFT, ", self.agent.choice_maker)
         msg.twist.linear.x = 0.0
         msg.twist.linear.y = 1.0
         msg.twist.linear.z = 0.0
         msg.twist.angular.x = 0.0
         msg.twist.angular.y = 0.0
         msg.twist.angular.z = 0.0
      #RIGHT
      if action == 3:
         if verbose > -1:
            print("Selected action: RIGHT, ", self.agent.choice_maker)
         msg.twist.linear.x = 0.0
         msg.twist.linear.y = -1.0
         msg.twist.linear.z = 0.0
         msg.twist.angular.x = 0.0
         msg.twist.angular.y = 0.0
         msg.twist.angular.z = 0.0
      msg.speed.x = self.control_params.roll
      msg.speed.y = self.control_params.yaw
      msg.speed.z = self.control_params.throttle
      self.control_publisher.publish(msg)
      return

# Main function
def main(args=None):
   d_id = input("Drone ID please: ")

   agent = Agent(gamma=0.99, epsilon=1.0, alpha=0.0005, input_dims=(84,84,1), n_actions=4, mem_size=1000, batch_size=10, epsilon_end=0.01, fname='/home/blue02/Desktop/Results/DQN/dqn_model.keras')
   control = Control(yaw=10.0, roll=10.0, throttle=10.0)
   environment = Environment(state_shape=(84,84,1))

   rclpy.init(args=args)
   drone_agent = DroneAgent_DQN(drone_id=d_id, agent=agent, control=control, environment=environment, n_episodes=50, save_path="/home/blue02/Desktop/Results/")

   try:
      rclpy.spin(drone_agent)
   except KeyboardInterrupt:
      # Destroy the node explicitly
      # (optional - otherwise it will be done automatically
      # when the garbage collector destroys the node object)
      drone_agent.destroy_node()
      rclpy.shutdown()
   return 0

if __name__ == '__main__':
    main()
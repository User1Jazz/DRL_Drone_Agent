import numpy as np
from tensorflow import keras
import math
import cv2
import time

# ROS2 dependencies
import rclpy
from rclpy.node import Node
from drone_sim_messages.msg import DroneControl
from drone_sim_messages.msg import DroneSensors
from drone_sim_messages.msg import SessionInfo
from drone_sim_messages.msg import DroneStatus
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float32

from .submodules.DoubleDQN import Agent
from .submodules.AgentMemory import RewardStorage, LossMemory, PositionMemory
from .submodules.Environment import Environment
from .submodules.Control import Control

class DroneAgent_DoubleDQN(Node):
   def __init__(self, drone_id, agent, control, environment, n_episodes = 10, save_every_n = 5, save_path=None, training=True):
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

      self.position_memory = PositionMemory()

      self.control_params = control

      self.id = drone_id

      self.training = training
      self.n_episodes = n_episodes
      self.episode_count = 1
      self.save_every_n = save_every_n

      self.agent = agent
      self.environment = environment
      self.reward_storage = RewardStorage()
      self.loss_memory = LossMemory()

      self.state  = self.environment.reset()
      self.done = self.environment.done
      self.trained = False

      self.pos_x = 0.0
      self.pos_y = 0.0
      self.pos_z = 0.0

      self.save_path = save_path
      return
   
   # Function to send control data
   def control_timer_callback(self):
      if not self.environment.done:
         self.trained = False
         if not self.training:
            self.position_memory.add_position(self.pos_x, self.pos_y, self.pos_z)
         action = self.agent.choose_action(self.state)                     # Choose action
         self.decode_action(action)                                        # Execute action
         time.sleep(0.1)                                                   # Wait for changes
         state_, reward, done = self.environment.step()                    # Get observation
         if self.training:
            self.agent.remember(self.state, action, reward, state_, done)  # Store experience
         self.state = state_                                               # Store old state
         if self.training:
            self.loss_memory.add_loss(self.agent.learn(verbose=0))
         self.reward_storage.add_reward(reward)                         # Save reward
      elif not self.training:
         self.position_memory.save_position_chart(self.save_path+"DoubleDQN_positions.png", chart_type="2d")
         self.position_memory.save_position_values(self.save_path+"DoubleDQN_positions.txt")
         self.reward_storage.save_rewards_list(self.save_path+"DoubleDQN_rewards_list.txt")
         print("Done")
         exit()
      elif not self.trained:
         self.reward_storage.store_episode_rewards()                       # Save episode rewards
         self.check_for_end()
         print("Preparing the agent...")
         if self.episode_count % self.save_every_n == 0:
            self.save_data()
         #self.agent.learn(verbose=2)                                      # Train the agent
         self.status_timer_callback()                                      # Send status to the simulator
         self.episode_count += 1                                           # Increase episode count
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
      if not self.training:
         self.pos_x = msg.local_position.x
         self.pos_y = msg.local_position.y
         self.pos_z = msg.local_position.z
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
         print("Reached episode ", (self.episode_count-1), " out of ", self.n_episodes)
         self.save_data(final_save=True)
         exit()
      return
   
   def save_data(self, final_save=False):
      if self.save_path != None:
         if final_save:
            self.agent.fname = '/home/kristijan.segulja/Desktop/Results/DoubleDQN/double_dqn_final_model.weights.h5'
         self.agent.save_model()
         self.reward_storage.save_reward_chart_mean(self.save_path+"DoubleDQN_rewards_mean.jpg", reward_gap=50)
         self.reward_storage.save_reward_chart_median(self.save_path+"DoubleDQN_rewards_median.jpg", reward_gap=50)
         self.reward_storage.save_rewards_list(self.save_path+"DobleDQN_rewards_list.txt")
         self.loss_memory.save_loss_chart(self.save_path+"DoubleDQN_losses.jpg", gap=1000)
         self.loss_memory.save_loss_list(self.save_path+"DoubleDQN_losses_list.txt")
         print("Data saved!")
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

   agent = Agent(gamma=0.99, epsilon=1.0, alpha=0.0005, input_dims=(84,84,1), n_actions=4, mem_size=1000, batch_size=100, epsilon_end=0.01,
                 replace_target=10, fname='/home/kristijan.segulja/Desktop/Results/DoubleDQN/ddouble_dqn_model.weights.h5', training=False)
   agent.load_model("/home/kristijan.segulja/Desktop/Results/DoubleDQN/RealDroneDDQN/ddouble_dqn_model.weights.h5")
   control = Control(yaw=10.0, roll=10.0, throttle=10.0)
   environment = Environment(state_shape=(84,84,1))

   rclpy.init(args=args)
   drone_agent = DroneAgent_DoubleDQN(drone_id=d_id,
                                      agent=agent,
                                      control=control,
                                      environment=environment,
                                      n_episodes=400,
                                      save_every_n=5,
                                      save_path="/home/kristijan.segulja/Desktop/Results/DoubleDQN/",
                                      training=True)

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

import numpy as np
from tensorflow import keras
import math
import cv2

# ROS2 dependencies
import rclpy
from rclpy.node import Node
from drone_sim_messages.msg import DroneControl
from drone_sim_messages.msg import DroneSensors
from drone_sim_messages.msg import SessionInfo
from drone_sim_messages.msg import DroneStatus
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float32

from .submodules.SAC import SAC

class DroneAgent_SAC(Node):
   def __init__(self, drone_id, p_net = None, q_net = None, v_net = None, max_episodes = 10, save_path=None):
      super().__init__('drone_agent')
      pub_topic = "/" + drone_id + "/cmd"
      self.control_publisher = self.create_publisher(DroneControl, pub_topic, 10)
      timer_period = 0.5 # seconds
      self.timer = self.create_timer(timer_period, self.control_timer_callback)
      self.i = 0
      pub_topic = "/" + drone_id + "/status"
      self.status_publisher = self.create_publisher(DroneStatus, pub_topic, 10)
      timer_period = 0.5 # seconds
      self.timer = self.create_timer(timer_period, self.status_timer_callback)
      self.i = 0
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

      self.yaw_speed = 12.0
      self.roll_speed = 10.0
      self.throttle_speed = 2.0

      self.id = drone_id
      self.active = False
      self.status_sent = False

      self.state = np.zeros((1,84,84,1))
      self.action = 0
      self.reward = 0
      self.done = 0

      self.agent = SAC(agent=self, P_net=p_net, Q_net=q_net, V_net=v_net)
      self.agent.set_hyperparams(no_actions=9, experience_buffer_size=2000, learning_rate=0.01, metrics=['mae'], discount_factor=0.5)
      self.agent.compile_networks()

      self.episode_count = 1
      self.max_episodes = max_episodes

      self.save_path = save_path
      return
   
   # Function to send control data (and train the agent)
   def control_timer_callback(self):
      if self.active and self.status_sent:
         self.agent.run(update_network=False, store_experience=True, verbose=0)
      else:
         self.agent.store_episode_rewards()
         if self.episode_count > self.max_episodes:
            if self.save_path != None:
               self.agent.save_reward_chart(save_path=self.save_path+"SAC_rewards.jpg")
               self.agent.save_networks(policy_path=self.save_path+"SAC_P_net.keras", q_path=self.save_path+"SAC_Q_net.keras", v_path=self.save_path+"SAC_V_net.keras", target_q_path=self.save_path+"SAC_target_Q_net.keras")
            print("Reached episode ", self.episode_count, " out of ", self.max_episodes)
            exit()
         print("Preparing the agent...")
         self.agent.train(no_exp=1000, verbose=2)
         self.active = True
         self.status_timer_callback()
         self.episode_count += 1
         print("Agent ready")
      return
   
   # Function to send drone agent status
   def status_timer_callback(self):
      if self.active and not self.status_sent:
         print("Sending status...")
         msg = DroneStatus()
         msg.id = self.id
         msg.active = True
         self.status_publisher.publish(msg)
         self.status_sent = True
      return
   
   # Function to get the drone state
   def sensor_listener_callback(self, msg):
      # Direct conversion to CV2 (decode the image)
      np_arr = np.frombuffer(msg.camera_image, np.uint8)
      decoded_image = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
      
      input_size = (84,84) # This should be provided by the sim
      
      resized_image = cv2.resize(decoded_image, input_size) # Resize the image
      normalized_image = resized_image / 255.0              # Normalize pixel values to range [0,1]
      
      self.state = np.expand_dims(normalized_image, axis=1)
      self.state = np.transpose(self.state, (1,0,2))
      return
   
   # Function to get reward value
   def reward_listener_callback(self, msg):
      self.reward = msg.data
      return
   
   # Function to get drone status
   def status_listener_callback(self, msg):
      self.active = msg.active
      if not self.active:
         self.status_sent = False
      return
   
   def session_listener_callback(self, msg):
      self.done = msg.session_ended
      return
   
   # Function to decode the action number into action
   def decode_action(self, verbose=0):
      msg = DroneControl()
      #FORWARD
      if self.agent.current_action == 0:
         if verbose > -1:
            print("Selected action: FORWARD, ", self.agent.choice_maker)
         msg.twist.linear.x = 1.0
         msg.twist.linear.y = 0.0
         msg.twist.linear.z = 0.0
         msg.twist.angular.x = 0.0
         msg.twist.angular.y = 0.0
         msg.twist.angular.z = 0.0
      #BACKWARD
      if self.agent.current_action == 1:
         if verbose > -1:
            print("Selected action: BACKWARD, ", self.agent.choice_maker)
         msg.twist.linear.x = -1.0
         msg.twist.linear.y = 0.0
         msg.twist.linear.z = 0.0
         msg.twist.angular.x = 0.0
         msg.twist.angular.y = 0.0
         msg.twist.angular.z = 0.0
      #LEFT
      if self.agent.current_action == 2:
         if verbose > -1:
            print("Selected action: LEFT, ", self.agent.choice_maker)
         msg.twist.linear.x = 0.0
         msg.twist.linear.y = 1.0
         msg.twist.linear.z = 0.0
         msg.twist.angular.x = 0.0
         msg.twist.angular.y = 0.0
         msg.twist.angular.z = 0.0
      #RIGHT
      if self.agent.current_action == 3:
         if verbose > -1:
            print("Selected action: RIGHT, ", self.agent.choice_maker)
         msg.twist.linear.x = 0.0
         msg.twist.linear.y = -1.0
         msg.twist.linear.z = 0.0
         msg.twist.angular.x = 0.0
         msg.twist.angular.y = 0.0
         msg.twist.angular.z = 0.0
      #UP
      if self.agent.current_action == 4:
         if verbose > -1:
            print("Selected action: UP, ", self.agent.choice_maker)
         msg.twist.linear.x = 0.0
         msg.twist.linear.y = 0.0
         msg.twist.linear.z = 1.0
         msg.twist.angular.x = 0.0
         msg.twist.angular.y = 0.0
         msg.twist.angular.z = 0.0
      #DOWN
      if self.agent.current_action == 5:
         if verbose > -1:
            print("Selected action: DOWN, ", self.agent.choice_maker)
         msg.twist.linear.x = 0.0
         msg.twist.linear.y = 0.0
         msg.twist.linear.z = -1.0
         msg.twist.angular.x = 0.0
         msg.twist.angular.y = 0.0
         msg.twist.angular.z = 0.0
      msg.speed.x = self.roll_speed
      msg.speed.y = self.yaw_speed
      msg.speed.z = self.throttle_speed
      self.control_publisher.publish(msg)
      return
   
   # Function to write the summary of the agent
   def summary(self):
      print("---------------")
      print("Policy Values:")
      print(self.agent.current_policy)
      print("---------------")
      print("P Network:")
      print(self.agent.P_net.summary())
      print("---------------")
      print("Q Network:")
      print(self.agent.Q_net.summary())
      print("---------------")
      print("V Network:")
      print(self.agent.V_net.summary())
      print("---------------")
      return

# Main function
def main(args=None):
   d_id = input("Drone ID please: ")

   p_net = keras.models.Sequential()
   p_net.add(keras.layers.Conv2D(32, (6,6), activation='relu', input_shape=(84,84,1)))
   p_net.add(keras.layers.Conv2D(64, (4,4), activation='relu', kernel_initializer='he_uniform'))
   p_net.add(keras.layers.Flatten())
   p_net.add(keras.layers.Dense(25, activation='relu', kernel_initializer='he_uniform'))
   p_net.add(keras.layers.Dense(6))

   q_net = keras.models.Sequential()
   q_net.add(keras.layers.Conv2D(32, (6,6), activation='relu', input_shape=(84,84,1)))
   q_net.add(keras.layers.Conv2D(64, (4,4), activation='relu', kernel_initializer='he_uniform'))
   q_net.add(keras.layers.Flatten())
   q_net.add(keras.layers.Dense(25, activation='relu', kernel_initializer='he_uniform'))
   q_net.add(keras.layers.Dense(6))

   v_net = keras.models.Sequential()
   v_net.add(keras.layers.Conv2D(32, (6,6), activation='relu', input_shape=(84,84,1)))
   v_net.add(keras.layers.Conv2D(64, (4,4), activation='relu', kernel_initializer='he_uniform'))
   v_net.add(keras.layers.Flatten())
   v_net.add(keras.layers.Dense(25, activation='relu', kernel_initializer='he_uniform'))
   v_net.add(keras.layers.Dense(1))

   rclpy.init(args=args)
   drone_agent = DroneAgent_SAC(drone_id=d_id, p_net=p_net, q_net=q_net, v_net=v_net, max_episodes=40, save_path="/home/blue02/Desktop/Results/")

   try:
      rclpy.spin(drone_agent)
   except KeyboardInterrupt:
      print("---------------")
      print("Results:")
      drone_agent.summary()
      # Destroy the node explicitly
      # (optional - otherwise it will be done automatically
      # when the garbage collector destroys the node object)
      drone_agent.destroy_node()
      rclpy.shutdown()
   return 0

if __name__ == '__main__':
    main()
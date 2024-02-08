import numpy as np
from tensorflow import keras
import math

# ROS2 dependencies
import rclpy
from rclpy.node import Node
from drone_sim_messages.msg import DroneControl
from drone_sim_messages.msg import DroneSensors

from .submodules.agent import Agent

class DroneAgent(Node):
  def __init__(self, drone_id, _adv_net = None, _state_val_net = None):
    # ROS2 Setup BELOW!
    super().__init__('drone_agent')    
    # Publishing stuff
    pub_topic = "/" + drone_id + "/cmd"
    self.publisher_ = self.create_publisher(DroneControl, pub_topic, 10)
    timer_period = 0.5  # seconds
    self.timer = self.create_timer(timer_period, self.timer_callback)
    self.i = 0
    # Subscription stuff
    sub_topic = "/" + drone_id + "/data"
    self.subscription = self.create_subscription(
        DroneSensors,
        sub_topic,
        self.listener_callback,
        10)
    self.subscription  # prevent unused variable warning
    # ROS2 Setup ABOVE!

    # Setup runtime vars
    self.imu_data = np.zeros(9)
    self.height_data = np.zeros(1)
    self.target_position = np.zeros(3)
    self.reward = 0.0
    
    # Setup the agent now!
    self.DRLagent = Agent(drone_id, _adv_net, _state_val_net)

  # Send drone control data
  def timer_callback(self):
    self.train()

  # Listen to incoming data; This function should update the observation data
  def listener_callback(self, msg):
    self.get_logger().info("Received sensors data")
    euler_angles = self.quaternion_2_euler(msg.orientation)
    self.imu_data = np.array([euler_angles[0],
                              euler_angles[1],
                              euler_angles[2],
                              msg.angular_velocity.x,
                              msg.angular_velocity.y,
                              msg.angular_velocity.z,
                              msg.linear_acceleration.x,
                              msg.linear_acceleration.y,
                              msg.linear_acceleration.z])             # (pitch, roll, yaw, linear acceleration x 3, angular velocity x 3)
    self.height_data = np.array([msg.height])                         # Single value for height
    self.target_position = np.array([0.0, 0.0, 0.0])                  # X, Y and Z coordinates for target position
  
  def train(self):
        # Get observation
        # next_state = np.array([np.concatenate((self.imu_data, self.height_data, self.target_position))])
        next_state = np.array([np.concatenate([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,], [0.0], [0.0, 0.0, 0.0]])])
        next_reward = self.reward
        self.DRLagent.update_params(next_state, next_reward)

        # Choose an action
        self.DRLagent.estimate_adv_values()
        self.DRLagent.estimate_state_val()
        self.DRLagent.get_q_values()
        self.DRLagent.choose_action()

        # Perform an action
        self.get_logger().info("Sending control data")
        self.i += 1
        self.publisher_.publish(self.decode_action())

        # Store observation, action, and reward (AND state value) to the experience buffer
        self.DRLagent.store_experience()

        # Get observation (and reward)
        # next_state = np.array([np.concatenate((self.imu_data, self.height_data, self.target_position))])
        next_state = np.array([np.concatenate([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,], [0.0], [0.0, 0.0, 0.0]])])
        next_reward = self.reward
        self.DRLagent.update_params(next_state, next_reward)

        # Bellman equation
        self.DRLagent.calculate_target()
        
        # Update networks
        self.DRLagent.update_networks(1)
        
        # Reset params for next train cycle
        self.DRLagent.reset_params()
        return
  
  def decode_action(self):
     msg = DroneControl()

     # IDLE
     if self.DRLagent.current_action is 0:
        msg.twist.linear.x = 0.0
        msg.twist.linear.y = 0.0
        msg.twist.linear.z = 0.0
        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = 0.0

     # FORWARD
     if self.DRLagent.current_action is 1:
        msg.twist.linear.x = 1.0
        msg.twist.linear.y = 0.0
        msg.twist.linear.z = 0.0
        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = 0.0
     
     # BACKWARD
     if self.DRLagent.current_action is 2:
        msg.twist.linear.x = -1.0
        msg.twist.linear.y = 0.0
        msg.twist.linear.z = 0.0
        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = 0.0
     
     # LEFT
     if self.DRLagent.current_action is 3:
        msg.twist.linear.x = 0.0
        msg.twist.linear.y = 1.0
        msg.twist.linear.z = 0.0
        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = 0.0
     
     # RIGHT
     if self.DRLagent.current_action is 4:
        msg.twist.linear.x = 0.0
        msg.twist.linear.y = -1.0
        msg.twist.linear.z = 0.0
        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = 0.0
     
     # UP
     if self.DRLagent.current_action is 5:
        msg.twist.linear.x = 0.0
        msg.twist.linear.y = 0.0
        msg.twist.linear.z = 1.0
        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = 0.0
     
     # DOWN
     if self.DRLagent.current_action is 6:
        msg.twist.linear.x = 0.0
        msg.twist.linear.y = 0.0
        msg.twist.linear.z = -1.0
        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = 0.0
     
     # YAW LEFT
     if self.DRLagent.current_action is 7:
        msg.twist.linear.x = 0.0
        msg.twist.linear.y = 0.0
        msg.twist.linear.z = 0.0
        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = -1.0
     
     # YAW RIGHT
     if self.DRLagent.current_action is 8:
        msg.twist.linear.x = 0.0
        msg.twist.linear.y = 0.0
        msg.twist.linear.z = 0.0
        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = 1.0
     return msg
  
  def quaternion_2_euler(self, quaternion):
      """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
      """
      t0 = +2.0 * (quaternion.w * quaternion.x + quaternion.y * quaternion.z)
      t1 = +1.0 - 2.0 * (quaternion.x * quaternion.x +quaternion.y * quaternion.y)
      roll_x = math.degrees(math.atan2(t0, t1))
     
      t2 = +2.0 * (quaternion.w * quaternion.y - quaternion.z * quaternion.x)
      t2 = +1.0 if t2 > +1.0 else t2
      t2 = -1.0 if t2 < -1.0 else t2
      pitch_y = math.degrees(math.asin(t2))
     
      t3 = +2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y)
      t4 = +1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z)
      yaw_z = math.degrees(math.atan2(t3, t4))
     
      return [roll_x, pitch_y, yaw_z] # in degrees

# Main function
def main(args=None):
    d_id = input("Drone ID please: ")

    rclpy.init(args=args)
    
    # Setting up an advantage values Neural Network model
    total_feature_dimensions = 13
    adv_model = keras.Sequential([
        keras.layers.Flatten(input_shape=(total_feature_dimensions,)),             # Input layer; The number of neurons is the same as the number of input parameters (obviously)
        keras.layers.Dense(128, activation='relu'),                                # Hidden layer after the input layer
        keras.layers.Dense(128, activation='relu'),                                # hidden layer (2); The number of neurons are chosen by us intuitivelly; RELU - "Rectified Linear Unit"
        keras.layers.Dense(9, activation='softmax')                                # output layer (3); 9 output neurons (one for each action the drone can take)
    ])
    adv_model.compile(optimizer='adam',                                            # Adam optimisation algorithm (stochastic gradient descent)
              loss='mse',                                                          # Mean Squared Error for Q-learning
              metrics=['mae'])                                                     # Mean Absolute Error could be used as a metric
    
    # Setting up a state value Neural Network model
    total_feature_dimensions = 13
    stat_val_model = keras.Sequential([
        keras.layers.Flatten(input_shape=(total_feature_dimensions,)),             # Input layer; The number of neurons is the same as the number of input parameters (obviously)
        keras.layers.Dense(128, activation='relu'),                                # Hidden layer after the input layer
        keras.layers.Dense(128, activation='relu'),                                # hidden layer (2); The number of neurons are chosen by us intuitivelly; RELU - "Rectified Linear Unit"
        keras.layers.Dense(1, activation='softmax')                                # output layer (3); 1 output neuron (for state value)
    ])
    stat_val_model.compile(optimizer='adam',                                       # Adam optimisation algorithm (stochastic gradient descent)
              loss='mse',                                                          # Mean Squared Error for Q-learning
              metrics=['mae'])                                                     # Mean Absolute Error could be used as a metric

    drone_agent = DroneAgent(d_id, adv_model, stat_val_model)

    rclpy.spin(drone_agent)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    drone_agent.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == '__main__':
    main()
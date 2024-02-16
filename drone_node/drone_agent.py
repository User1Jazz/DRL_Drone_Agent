import numpy as np
from tensorflow import keras
import math

# ROS2 dependencies
import rclpy
from rclpy.node import Node
from drone_sim_messages.msg import DroneControl
from drone_sim_messages.msg import DroneSensors
from drone_sim_messages.msg import SessionInfo
from drone_sim_messages.msg import DroneStatus
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float32

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
    pub_topic = "/" + drone_id + "/status"
    self.status_publisher_ = self.create_publisher(DroneStatus, pub_topic, 10)
    timer_period = 0.5  # seconds
    self.timer = self.create_timer(timer_period, self.status_timer_callback)
    self.i = 0
    ## Subscription stuff
    sub_topic = "/" + drone_id + "/data"
    self.sens_sub = self.create_subscription(
         DroneSensors,
         sub_topic,
         self.sensor_listener_callback,
         10)
    self.sens_sub  # prevent unused variable warning
    sub_topic = "/" + drone_id + "/target"
    self.tgt_sub = self.create_subscription(
         Vector3,
         sub_topic,
         self.target_listener_callback,
         10)
    self.tgt_sub
    sub_topic = "/" + drone_id + "/reward"
    self.rwrd_sub = self.create_subscription(
         Float32,
         sub_topic,
         self.reward_listener_callback,
         10)
    self.rwrd_sub
    sub_topic = "/" + drone_id + "/status"
    self.stat_sub = self.create_subscription(
         DroneStatus,
         sub_topic,
         self.status_listener_callback,
         10)
    self.rwrd_sub
    # ROS2 Setup ABOVE!

    # Setup runtime vars
    self.active = False
    self.status_sent = False
    self.imu_data = np.zeros(6)
    self.height_data = np.zeros(1)
    self.target_position = np.zeros(3)
    self.reward = 0.0
    
    # Agent Hyperparameters
    self.id = drone_id
    self.learning_rate = 0.1
    self.discount_factor = 0.5
    self.exploration_prob = 0.75
    self.exploration_decrease = 0.02
    # Setup the agent now!
    self.DRLagent = Agent(drone_id, _adv_net, _state_val_net)
    self.DRLagent.set_hypers(learn_rate=self.learning_rate,
                             discount_fac=self.discount_factor,
                             exp_prob=self.exploration_prob,
                             exp_dec=self.exploration_decrease)

  # Send drone control data
  def timer_callback(self):
    # Make sure the drone is acive and the simulator is ready
    if self.active and self.status_sent:
      self.train()
    else:
       print("Preparing the agent...")
       self.DRLagent.update_exploration_probability()
       #self.reset()
       self.active = True
       print("Agent ready")
  
  # Function to send the 'ready' signal to the simulator
  def status_timer_callback(self):
     if self.active and not self.status_sent:
        print("Sending status...")
        msg = DroneStatus()
        msg.id = self.id
        msg.active = True
        self.status_publisher_.publish(msg)
        self.status_sent = True
     return

  # Function to pick the drone status message (active/disabled)
  def status_listener_callback(self, msg):
     self.active = msg.active
     if not self.active:
        self.status_sent = False

  # Listen to incoming data; This function should update the observation data
  def sensor_listener_callback(self, msg):
      #self.get_logger().info("Received sensors data")
      euler_angles = self.quaternion_2_euler(msg.orientation)
      self.imu_data = np.array([msg.world_position.x,
                                msg.world_position.y,
                                msg.world_position.z,
                                euler_angles[0],
                                euler_angles[1],
                                euler_angles[2]])     # (Position x 3, rotation x 3)
      self.height_data = np.array([msg.height])       # Single value for height
  
  def target_listener_callback(self, msg):
      #self.get_logger().info("Received target data")
      self.target_position = np.array([msg.x, msg.y, msg.z])  # X, Y and Z coordinates for target position
   
  def reward_listener_callback(self, msg):
      #self.get_logger().info("Received reward data")
      self.reward = msg.data

  def train(self):
        # Get observation
        next_state = np.array([np.concatenate([self.imu_data, self.height_data, self.target_position])])
        next_reward = self.reward
        self.DRLagent.update_params(next_state, next_reward)

        # Choose an action
        self.DRLagent.estimate_adv_values()
        self.DRLagent.estimate_state_val()
        self.DRLagent.get_q_values()
        self.DRLagent.choose_action()

        # Perform an action
        #self.get_logger().info("Sending control data")
        self.i += 1
        self.publisher_.publish(self.decode_action())

        # Store observation, action, and reward (AND state value) to the experience buffer
        self.DRLagent.store_experience()

        # Get observation (and reward)
        next_state = np.array([np.concatenate([self.imu_data, self.height_data, self.target_position])])
        next_reward = self.reward
        self.DRLagent.update_params(next_state, next_reward)

        # Bellman equation
        self.DRLagent.calculate_target()
        
        # Update networks
        self.DRLagent.update_networks(1)
        
        # Reset params for next train cycle
        self.DRLagent.reset_params()
        return
  
  def reset(self):
     self.DRLagent.set_hypers(learn_rate=self.learning_rate,
                             discount_fac=self.discount_factor,
                             exp_prob=self.exploration_prob,
                             exp_dec=self.exploration_decrease)
     return
  
  def decode_action(self):
     msg = DroneControl()

     # IDLE
     if self.DRLagent.current_action == 0:
        print("Selected action: IDLE, ", self.DRLagent.choice_maker)
        msg.twist.linear.x = 0.0
        msg.twist.linear.y = 0.0
        msg.twist.linear.z = 0.0
        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = 0.0

     # FORWARD
     if self.DRLagent.current_action == 1:
        print("Selected action: FORWARD, ", self.DRLagent.choice_maker)
        msg.twist.linear.x = 1.0
        msg.twist.linear.y = 0.0
        msg.twist.linear.z = 0.0
        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = 0.0
     
     # BACKWARD
     if self.DRLagent.current_action == 2:
        print("Selected action: BACKWARD, ", self.DRLagent.choice_maker)
        msg.twist.linear.x = -1.0
        msg.twist.linear.y = 0.0
        msg.twist.linear.z = 0.0
        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = 0.0
     
     # LEFT
     if self.DRLagent.current_action == 3:
        print("Selected action: LEFT, ", self.DRLagent.choice_maker)
        msg.twist.linear.x = 0.0
        msg.twist.linear.y = 1.0
        msg.twist.linear.z = 0.0
        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = 0.0
     
     # RIGHT
     if self.DRLagent.current_action == 4:
        print("Selected action: RIGHT, ", self.DRLagent.choice_maker)
        msg.twist.linear.x = 0.0
        msg.twist.linear.y = -1.0
        msg.twist.linear.z = 0.0
        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = 0.0
     
     # UP
     if self.DRLagent.current_action == 5:
        print("Selected action: UP,", self.DRLagent.choice_maker)
        msg.twist.linear.x = 0.0
        msg.twist.linear.y = 0.0
        msg.twist.linear.z = 1.0
        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = 0.0
     
     # DOWN
     if self.DRLagent.current_action == 6:
        print("Selected action: DOWN, ", self.DRLagent.choice_maker)
        msg.twist.linear.x = 0.0
        msg.twist.linear.y = 0.0
        msg.twist.linear.z = -1.0
        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = 0.0
     
     # YAW LEFT
     if self.DRLagent.current_action == 7:
        print("Selected action: YAW LEFT, ", self.DRLagent.choice_maker)
        msg.twist.linear.x = 0.0
        msg.twist.linear.y = 0.0
        msg.twist.linear.z = 0.0
        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = -1.0
     
     # YAW RIGHT
     if self.DRLagent.current_action == 8:
        print("Selected action: YAW RIGHT, ", self.DRLagent.choice_maker)
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
  
  def summary(self):
      print("---------------")
      print("Q Values:")
      print(self.DRLagent.q_values)
      print("---------------")
      print("State Value:")
      print(self.DRLagent.current_state_val)
      print("---------------")
      print("Adv Values:")
      print(self.DRLagent.current_adv_vals)
      print("---------------")
      print("State Val Net:")
      print(self.DRLagent.state_val_net.summary())
      print("---------------")
      print("Adv Net:")
      print(self.DRLagent.adv_net.summary())
      print("---------------")

# Main function
def main(args=None):
    d_id = input("Drone ID please: ")

    rclpy.init(args=args)
    
    # Setting up an advantage values Neural Network model
    total_feature_dimensions = 10
    adv_model = keras.Sequential([
        keras.layers.Flatten(input_shape=(total_feature_dimensions,)),             # Input layer; The number of neurons is the same as the number of input parameters (obviously)
        keras.layers.Dense(128, activation='relu'),                                # Hidden layer after the input layer
        keras.layers.Dense(128, activation='relu'),                                # hidden layer (2); The number of neurons are chosen by us intuitivelly; RELU - "Rectified Linear Unit"
        keras.layers.Dense(9, activation='softmax')                                # output layer (3); 9 output neurons (one for each action the drone can take)
    ])
    adv_model.compile(optimizer='adam',                                            # Adam optimisation algorithm (stochastic gradient descent)
              loss='mse',                                     # Categorical Crossentropy
              metrics=['mae'])                                                # Accuracy could be used as a metric
    
    # Setting up a state value Neural Network model
    total_feature_dimensions = 10
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
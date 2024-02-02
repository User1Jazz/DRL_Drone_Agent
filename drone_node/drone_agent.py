import numpy as np
from tensorflow import keras

# Manual control dependencies
import keyboard

# ROS2 dependencies
import rclpy
from rclpy.node import Node
from drone_sim_messages.msg import DroneControl
from drone_sim_messages.msg import DroneSensors

class DroneAgent:
  def __init__(self, drone_id):
    # ROS2 Setup BELOW!
    super().__init__('drone_agent')    
    # Publishing stuff
    pub_topic = "/" + drone_id + "/cmd"
    self.publisher_ = self.create_publisher(String, pub_topic, 10)
    timer_period = 0.5  # seconds
    self.timer = self.create_timer(timer_period, self.timer_callback)
    self.i = 0
    # Subscription stuff
    sub_topic = "/" + drone_id + "/sensors"
    self.subscription = self.create_subscription(
        String,
        sub_topic,
        self.listener_callback,
        10)
    self.subscription  # prevent unused variable warning
    # ROS2 Setup ABOVE!

    # Setup the agent now!

    return

  # Send drone control data
  def timer_callback(self):
    msg = DroneControl()
    msg.twist.linear.x = 0
    msg.twist.linear.y = 0
    msg.twist.linear.z = 0
    msg.twist.angular.x = 0
    msg.twist.angular.y = 0
    msg.twist.angular.z = 0

    # Forward/Backward
    if keyboard.is_pressed('w'):
       msg.twist.linear.x = 1
    if keyboard.is_pressed('s'):
       msg.twist.linear.x = -1
    
    # Left/Right
    if keyboard.is_pressed('a'):
       msg.twist.linear.y = -1
    if keyboard.is_pressed('d'):
       msg.twist.linear.y = 1
    
    # Up/Down
    if keyboard.is_pressed('r'):
       msg.twist.linear.z = 1
    if keyboard.is_pressed('f'):
       msg.twist.linear.z = -1
    
    #Yaw Left/Right
    if keyboard.is_pressed('e'):
       msg.twist.angular.z = 1
    if keyboard.is_pressed('q'):
       msg.twist.angular.z = -1

    self.get_logger().info("Sending control data")
    self.i += 1
    self.publisher_.publish(msg)
    return

  # Listen to incoming data; This function should update the observation data
  def listener_callback(self, msg):
    self.get_logger().info("Received sensors data")
    self.imu_data = np.array([0.0,
                              0.0,
                              0.0,
                              msg.imu.angular_velocity.x,
                              msg.imu.angular_velocity.y,
                              msg.imu.angular_velocity.z,
                              msg.linear_acceleration.x,
                              msg.linear_acceleration.y,
                              msg.linear_acceleration.z])             # (pitch, roll, yaw, linear acceleration x 3, angular velocity x 3)
    self.height_data = np.array([msg.height])                         # Single value for height
    self.target_position = np.array([0.0, 0.0, 0.0])                  # X, Y and Z coordinates for target position
    
    return

# Main function
def main(args=None):
    d_id = input("Drone ID please: ")

    rclpy.init(args=args)

    drone_agent = DroneAgent(d_id)

    rclpy.spin(drone_agent)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    drone_agent.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == '__main__':
    main()
import numpy as np
from tensorflow import keras

# Manual control dependencies
import keyboard

# ROS2 dependencies
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

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
    return

  # Send vehicle control data
  def timer_callback(self):
    msg = String()
    self.get_logger().info("Sending something")
    self.i += 1
    return

  # Listen to incoming data
  def listener_callback(self, msg):
    self.get_logger().info("eceived something")
    # This function should update the observation data
    return
  
def main(args=None):
    rclpy.init(args=args)

    d_id = input("Drone ID please: ")

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
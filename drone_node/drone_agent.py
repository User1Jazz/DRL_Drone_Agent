import numpy as np
from tensorflow import keras

# ROS2 dependencies
import rclpy
from rclpy.node import Node
from drone_sim_messages.msg import DroneControl
from drone_sim_messages.msg import DroneSensors

class DroneAgent(Node):
  def __init__(self, drone_id):
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
    
    # Setup the agent now!

  # Send drone control data
  def timer_callback(self):
    msg = DroneControl()
    msg.twist.linear.x = 1.0
    msg.twist.linear.y = 0.0
    msg.twist.linear.z = 1.0
    msg.twist.angular.x = 0.0
    msg.twist.angular.y = 0.0
    msg.twist.angular.z = 0.0

    self.get_logger().info("Sending control data")
    self.i += 1
    self.publisher_.publish(msg)

  # Listen to incoming data; This function should update the observation data
  def listener_callback(self, msg):
    self.get_logger().info("Received sensors data")
    self.imu_data = np.array([0.0,
                              0.0,
                              0.0,
                              msg.angular_velocity.x,
                              msg.angular_velocity.y,
                              msg.angular_velocity.z,
                              msg.linear_acceleration.x,
                              msg.linear_acceleration.y,
                              msg.linear_acceleration.z])             # (pitch, roll, yaw, linear acceleration x 3, angular velocity x 3)
    self.height_data = np.array([msg.height])                         # Single value for height
    self.target_position = np.array([0.0, 0.0, 0.0])                  # X, Y and Z coordinates for target position

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
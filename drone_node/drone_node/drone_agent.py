import numpy as np
from tensorflow import keras

# ROS2 dependencies
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class DroneAgent:
  def __init__(self):
    # ROS2 Setup BELOW!
    super().__init__('path_planning')    
    # Publishing stuff
    self.publisher_ = self.create_publisher(String, '/cmd', 10)
    timer_period = 0.5  # seconds
    self.timer = self.create_timer(timer_period, self.timer_callback)
    self.i = 0
    # Subscription stuff
    self.subscription = self.create_subscription(
        String,
        '/sensors',
        self.listener_callback,
        10)
    self.subscription  # prevent unused variable warning
    # ROS2 Setup ABOVE!

    # Setting up action variables
    self.action = np.array([False, False, False, False, False, False, False, False])    # Forward, Backward, Left, Right, Up, Down, Yaw left, Yaw Right
    # Setting up observation (state) variables
    self.imu_data = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])             # (pitch, roll, yaw, linear acceleration x 3, angular velocity x 3)
    self.height_data = np.array([0.0])                                                  # Single value for height
    self.target_position = np.array([0.0, 0.0, 0.0])                                    # X, Y and Z coordinates for target position

    # Q-learning hyperparameters
    self.learning_rate = 0.1
    self.discount_factor = 0.5
    self.exploration_prob = 0.5
    #self.experience_buffer = np.array()

    # Previous state, action, and reward for Q-learning update
    self.prev_state = None
    self.prev_action = None
    self.prev_reward = None

    # Setting up a Neural Network model
    self.total_feature_dimensions = len(self.imu_data) + len(self.height_data) + len(self.target_position)
    self.model = keras.Sequential([
        keras.layers.Flatten(input_shape=(self.total_feature_dimensions,)),             # Input layer; The number of neurons is the same as the number of input parameters (obviously)
        keras.layers.Dense(128, activation='relu'),                                     # Hidden layer after the input layer
        keras.layers.Dense(128, activation='relu'),                                     # hidden layer (2); The number of neurons are chosen by us intuitivelly; RELU - "Rectified Linear Unit"
        keras.layers.Dense(8, activation='softmax')                                     # output layer (3); 8 output layers (one for each action the drone can take)
    ])
    self.model.compile(optimizer='adam',                                                # Adam optimisation algorithm (stochastic gradient descent)
              loss='mse',                                                               # Mean Squared Error for Q-learning
              metrics=['mae'])                                                          # Mean Absolute Error could be used as a metric


  # Send vehicle control data
  def timer_callback(self):
    msg = String()
    self.get_logger().info('Sending something')
    self.i += 1


  # Listen to incoming data
  def listener_callback(self, msg):
    self.get_logger().info('Received something')
    # This function should update the observation data


  def train(self, num_episodes, verbose= True):
    for episode in range(num_episodes):
      # Reset the environment for a new episode (i.e. reset the sim world)
      state = self.reset_environment()
      total_reward = 0
      done = False

      while not done:
        # Choose an action
        action = self.choose_action(state)

        # Take the action and observe the next state and reward
        next_state, reward, done = self.take_environment_action(action)

        # Learn from the experience (update Q-values)
        self.learn_from_experience(next_state, reward)

        total_reward += reward
        state = next_state

        # Optionally, log or visualize episode-specific information
        if verbose:
          print(f"Episode: {episode + 1}, Total Reward: {total_reward}")


  def reset_environment(self):
    # Reset the environment for a new episode and return the initial state (i.e. send the reset command to the simulator and reset the agent parameters [except its neural net])
    # Implement this function based on how your environment is set up
    return initial_state


  # Returns the numpy array of actions (action with the highest number is the chosen action)
  def choose_action(self, state):
    q_values = np.array([np.zeros(len(self.action))])
    # Implement your action selection strategy (epsilon-greedy, softmax, etc.); Below is epsilon-greedy policy (i.e. flip the coin and take either random action or the 'predicted' one)
    if np.random.rand() < self.exploration_prob:
      rand_int = np.random.choice(len(self.action))
      q_values[0][rand_int] = 1
    else:
      q_values = self.model.predict(state)
    # Return the chosen action
    return q_values


  # Send action to the simulator and observe the next state (i.e. the simulator sends the new set of observations)
  def take_environment_action(self, action):
    # Implement taking the chosen action in the environment (i.e. send the action to the simulator)
    # Return the next state, reward, and done status (i.e. receive sensor data from the simulator)
    return next_state, reward, done


  def learn_from_experience(self, next_state, reward):
    # Implement the learning process (update Q-values or train the neural network)
    # based on the observed experience
    # This is where you would call the update_q_values method or train the neural network
    pass
  
def main(args=None):
    rclpy.init(args=args)

    drone_agent = DroneAgent()

    rclpy.spin(drone_agent)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    drone_agent.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
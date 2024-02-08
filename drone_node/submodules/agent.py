import numpy as np
import tensorflow as tf
from tensorflow import keras

class Agent():
    def __init__(self, id, _adv_net = None, _state_val_net = None):
        self.id = id
        
        # Setting up action variables
        self.action = np.array([False, False, False, False, False, False, False, False])            # Forward, Backward, Left, Right, Up, Down, Yaw left, Yaw Right
        # Setting up observation (state) variables
        self.imu_data = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])                     # (pitch, roll, yaw, linear acceleration x 3, angular velocity x 3)
        self.height_data = np.array([0.0])                                                          # Single value for height
        self.target_position = np.array([0.0, 0.0, 0.0])                                            # X, Y and Z coordinates for target position
        self.experience_buffer = np.array([np.zeros(24)])

        # Hyperparameters
        self.learning_rate = 0.1
        self.discount_factor = 0.5
        self.exploration_prob = 0.25
        self.exploration_decrease = 0.1

        # Runtime parameters
        self.prev_state = None
        self.prev_action = None
        self.prev_reward = None
        self.prev_state_val = None
        self.prev_adv_val = None

        self.current_state = None
        self.current_action = None
        self.current_reward = None
        self.current_state_val = None
        self.current_adv_val = None

        self.q_values = None
        self.target = None

        # Setting up a Neural Network model(s)
        if _adv_net is not None:
            self.adv_net = _adv_net
        else:
            print("Advantage Neural Network not set")
            exit()
        if _state_val_net is not None:
            self.state_val_net = _state_val_net
        else:
            print("State Value Neural Network not set")
            exit()
        return
    
    def choose_action(self):
        # Epsilon-greedy policy
        if np.random.rand() < self.exploration_prob:
            self.exploration_prob -= self.exploration_decrease
            self.current_action = np.random.choice(len(self.action))
        else:
            self.current_action = np.argmax(self.q_values)
        return
    
    # Returns the state value
    def get_state_value(self, state):
        state_val = self.state_val_net.predict(state.reshape(1, -1))
        return state_val
    
    # Function to store the experience of a single train cycle
    def store_experience(self):
        experience = np.concatenate((self.current_state, self.current_adv_val, self.current_state_val, [np.array([self.current_reward])]), axis=1)
        np.append(self.experience_buffer, experience, axis=0)
        return
    
    # Function to get the new input values and store the old ones
    def update_params(self, next_state, next_reward):
        # Store current params if it contains data
        if self.current_state is not None:
            self.prev_state = self.current_state
        if self.current_action is not None:
            self.prev_action = self.current_action
        if self.current_reward is not None:
            self.prev_reward = self.current_reward
        if self.current_state_val is not None:
            self.prev_state_val = self.current_state_val
        if self.current_adv_val is not None:
            self.prev_adv_val = self.current_adv_val
        # Update current state and reward
        self.current_state = next_state
        self.current_reward = next_reward
        return

    # Function to update (fit) the networks    
    def update_networks(self, num_epoch):
        self.adv_net.fit(self.prev_state, self.target, epochs=num_epoch, verbose=0)
        self.state_val_net.fit(self.prev_state, self.target, epochs=num_epoch, verbose=0)
        return
    
    # Function to calculate the target
    def calculate_target(self):
        target_q_values = self.current_reward + self.discount_factor * self.current_adv_val[np.argmax(self.current_adv_val)]
        target_state_value = self.current_reward + self.discount_factor * self.current_state_val
        self.target = target_q_values + target_state_value
        return

    # Function to reset runtime parameters    
    def reset_params(self):
        self.current_reward = None
        self.current_action = None
        self.current_adv_val = None
        self.current_state = None
        self.current_state_val = None
        return
    
    # Function to estimate advantage values
    def estimate_adv_values(self):
        self.current_adv_val = self.adv_net.predict(self.current_state.reshape(1, -1))
        return

    # Function to estimate the state value    
    def estimate_state_val(self):
        self.current_state_val = self.state_val_net.predict(self.current_state.reshape(1,-1))
        return
    
    # Function to get the q values (actions) using the following equation: [ q = v + a - a.mean() ]
    def get_q_values(self):
        self.q_values = keras.layers.Add()([self.current_state_val, keras.layers.Subtract()([self.current_adv_val, keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1, keepdims=True))(self.current_adv_val)])])
        return
    
    # OLD TRAIN FUNCTION
    def train(self):
        # Get observation
        self.current_state = np.concatenate(self.imu_data, self.height_data, self.target_position)
        
        # Choose an action (and estimate the state value)
        adv_values = self.choose_action(self.current_state)
        self.current_state_val = self.get_state_value(self.current_state)
        
        # Act (i.e. combine advantage values and the state value)
        q_values = keras.layers.Add()([self.current_state_val, keras.layers.Subtract()([adv_values, keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1, keepdims=True))(adv_values)])])
        action = np.argmax(q_values)
        
        # Get reward
        self.curent_reward = 0.0
        
        # Store observation, action, and reward (AND state value) to the experience buffer
        np.append(self.experience_buffer, [[self.current_state, q_values, self.current_state_val, self.current_reward]])
        
        # Get next observation
        self.prev_state = self.current_state
        self.current_state = np.concatenate(self.imu_data, self.height_data, self.target_position)
        
        # Store new observation
        
        # Bellman equation
        target_q_values = self.current_reward + self.discount_factor * q_values[np.argmax(q_values)]
        target_state_value = self.current_reward + self.discount_factor * self.current_state_val
        
        # Update networks
        self.update_network(self.adv_net, self.prev_state.reshape(1, -1), target_q_values, 1)
        self.update_network(self.state_val_net, self.prev_state.reshape(1, -1), target_state_value, 1)
        return
    
    # OLD CHOOSE ACTION FUNCTION
    # Returns q values (in dueling DQNs these are the advantage values)
    def old_choose_action(self, state):
        # Epsilon-greedy policy
        if np.random.rand() < self.exploration_prob:
            self.exploration_prob -= self.exploration_decrease
            return np.random.choice(len(self.action))
        else:
            q_values = self.adv_net.predict(state.reshape(1, -1)) # this will give us action values
            #return np.argmax(q_values)
            return q_values
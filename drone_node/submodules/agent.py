import numpy as np
import tensorflow as tf
from tensorflow import keras

class Agent():
    def __init__(self, id, _adv_net = None, _state_val_net = None):
        self.id = id
        
        # Setting up action variables
        self.action = np.array([False, False, False, False, False, False, False, False])            # Forward, Backward, Left, Right, Up, Down, Yaw left, Yaw Right

        self.experience_buffer = np.array([np.zeros(21)])

        # Set hyperparameters
        self.set_hypers()

        # Runtime parameters
        self.prev_state = None
        self.prev_action = None     #Not used yet
        self.prev_reward = None     #Not used yet
        self.prev_state_val = None  #Not used yet
        self.prev_adv_val = None    #Not used yet

        self.current_state = None
        self.current_action = None
        self.current_reward = None
        self.current_state_val = None
        self.current_adv_vals = None

        self.q_values = None
        self.target_adv_vals = None
        self.target_state_val = None

        self.choice_maker = None

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
            self.current_action = np.random.choice(len(self.action))
            self.choice_maker = "[RANDOM]"
        else:
            self.current_action = np.argmax(self.q_values)
            self.choice_maker = "[ESTIMATED]"
        return
     
    # Function to estimate advantage values
    def estimate_adv_values(self):
        self.current_adv_vals = self.adv_net.predict(self.current_state.reshape(1, -1), verbose=0)
        return

    # Function to estimate the state value    
    def estimate_state_val(self):
        self.current_state_val = self.state_val_net.predict(self.current_state.reshape(1,-1), verbose=0)
        return
    
    # Returns the state value
    def get_state_value(self, state):
        state_val = self.state_val_net.predict(state.reshape(1, -1))
        return state_val 
    
    # Function to get the q values (actions) using the following equation: [ q = v + a - a.mean() ]
    def get_q_values(self):
        self.q_values = self.current_state_val + self.current_adv_vals - np.mean(self.current_adv_vals)
        return
    
    # Function to calculate the target
    def calculate_target(self):
        #target_q_values = self.current_reward + self.discount_factor * self.current_adv_vals
        #target_state_value = self.current_reward + self.discount_factor * self.current_state_val
        target_state_values = self.current_reward + self.discount_factor * self.prev_adv_val
        target_adv_values = self.current_reward + self.discount_factor * self.current_state_val - self.prev_state_val
        self.target_adv_vals = target_adv_values
        self.target_state_val = np.array([target_state_values[0][np.argmax(self.current_adv_vals)]])
        #print("Target state values: ", target_state_values)
        #print("Target adv values: ", target_adv_values)
        #print("Target state value: ", self.target_state_val)
        return
    
    # Function to update (fit) the networks    
    def update_networks(self, num_epoch):
        self.adv_net.fit(self.prev_state, self.target_adv_vals, epochs=num_epoch, verbose=0)
        self.state_val_net.fit(self.prev_state, self.target_state_val, epochs=num_epoch, verbose=0)
        return
    
    # Function to store the experience of a single train cycle
    def store_experience(self):
        experience = np.concatenate((self.current_state, self.current_adv_vals, self.current_state_val, [np.array([self.current_reward])]), axis=1)
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
        if self.current_adv_vals is not None:
            self.prev_adv_val = self.current_adv_vals
        # Update current state and reward
        self.current_state = next_state
        self.current_reward = next_reward
        return
    
    def update_exploration_probability(self):
        self.exploration_prob -= self.exploration_decrease
    
    # Function to set agent's hyperparameters
    def set_hypers(self, learn_rate=0.1, discount_fac=0.5, exp_prob=0.5, exp_dec=0.1):
        self.learning_rate = learn_rate
        self.discount_factor = discount_fac
        self.exploration_prob = exp_prob
        self.exploration_decrease = exp_dec
        return

    # Function to reset runtime parameters    
    def reset_params(self):
        self.current_reward = None
        self.current_action = None
        self.current_adv_vals = None
        self.current_state = None
        self.current_state_val = None
        return
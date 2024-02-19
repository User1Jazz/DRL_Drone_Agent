import numpy as np
import tensorflow as tf
from tensorflow import keras

class Agent():
    def __init__(self, id, _adv_net = None, _state_val_net = None):
        self.id = id
        
        # Setting up action variables
        self.action = np.array([False, False, False, False, False, False, False, False])            # Forward, Backward, Left, Right, Up, Down, Yaw left, Yaw Right

        self.experience = np.array([np.array(np.zeros(11)),np.array(np.zeros(1)),np.array(np.zeros(1)),np.array(np.zeros(11))], dtype=object)
        self.experience_buffer = np.array(np.array([self.experience]), dtype=object)

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

        self.no_exp = 0
        self.exp_count = 0

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
        #print("Current adv vals: ", self.current_adv_vals)
        return

    # Function to estimate the state value    
    def estimate_state_val(self):
        self.current_state_val = self.state_val_net.predict(self.current_state.reshape(1,-1), verbose=0)
        #print("Current state val: ", self.current_state_val)
        return
    
    # Returns the state value
    def get_state_value(self, state):
        state_val = self.state_val_net.predict(state.reshape(1, -1))
        return state_val 
    
    # Function to get the q values (actions) using the following equation: [ q = v + a - a.mean() ]
    def get_q_values(self):
        self.q_values = self.current_state_val + self.current_adv_vals - np.mean(self.current_adv_vals)
        return
    
    # Function to calculate the target values
    def calculate_target(self):
        target_state_value = self.current_reward + self.discount_factor * np.mean(self.q_values)
        target_adv_values = self.q_values - self.current_adv_vals
        self.target_adv_vals = target_adv_values
        self.target_state_val = np.array([target_state_value])
        return
    
    # Function to update (fit) the networks    
    def update_networks(self, num_epoch, set_verbose=0):
        self.adv_net.fit(self.prev_state.reshape(1,-1), self.target_adv_vals.reshape(1,-1), epochs=num_epoch, verbose=set_verbose)
        self.state_val_net.fit(self.prev_state.reshape(1,-1), self.target_state_val.reshape(1,-1), epochs=num_epoch, verbose=set_verbose)
        return
    
    def train(self, no_exp, verbose=0):
        print("Optimisation in progress...")
        # Sample no_exp experiences from experience replay buffer
        sample_experience = None
        if no_exp >= len(self.experience_buffer):
            sample_experience = self.experience_buffer[np.random.choice(len(self.experience_buffer), size = len(self.experience_buffer), replace = True)]
        else:
            sample_experience = self.experience_buffer[np.random.choice(len(self.experience_buffer), size = no_exp, replace = True)]
        # Update neural networks
        for i in range(len(sample_experience)):
            self.prev_state = sample_experience[i][0]       # Get s
            self.current_action = sample_experience[i][1]   # Get a
            self.current_reward = sample_experience[i][2]   # Get r
            self.current_state = sample_experience[i][3]    # Get s'
            #print("Prev state: ", self.prev_state)
            #print("Current action: ", self.current_action)
            #print("Current reward: ", self.current_reward)
            #print("Current state: ", self.current_state)
            self.estimate_state_val()                       # Estimate state value
            self.estimate_adv_values()                      # Estimate advantage values
            self.get_q_values()                             # Calculate Q values
            self.calculate_target()                         # Calculate target state value and target advantage values
            self.update_networks(1, set_verbose=verbose)    # Update networks
        return
    
    # Function to store the experience of a single train cycle (e = (s, a, r, s'))
    def store_experience(self):
        experience = np.array([np.array(np.zeros(11)),np.array(np.zeros(1)),np.array(np.zeros(1)),np.array(np.zeros(11))], dtype=object)
        if self.exp_count >= self.replay_buffer_size:
            self.exp_count = 0

        # If first experience: store initial state twice
        if self.prev_state is None:
            experience[0] = self.current_state # Initial state as the current state
            experience[1] = np.array([self.current_action])
            experience[2] = np.array([self.current_reward])
            experience[3] = self.current_state # Initial state as the next state
            if self.no_exp < self.replay_buffer_size:
                self.experience_buffer = np.append(self.experience_buffer, [experience], axis=0)
            else:
                #print("Buffer max size reached. Placing new experience to index ", self.exp_count)
                self.experience_buffer[self.exp_count] = experience
                self.exp_count += 1
        else:
            experience[0] = self.prev_state
            experience[1] = self.current_action
            experience[2] = np.array([self.current_reward])
            experience[3] = self.current_state
            if self.no_exp < self.replay_buffer_size:
                self.experience_buffer = np.append(self.experience_buffer, [experience], axis=0)
            else:
                #print("Buffer max size reached. Placing new experience to index ", self.exp_count)
                self.experience_buffer[self.exp_count] = experience
                self.exp_count += 1
        self.no_exp += 1
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
        #print("New state: ", self.current_state)
        #print("New reward: ", self.current_reward)
        return
    
    def update_exploration_probability(self):
        self.exploration_prob -= self.exploration_decrease
    
    # Function to set agent's hyperparameters
    def set_hypers(self, replay_buffer_size = 200, learn_rate=0.1, discount_fac=0.5, exp_prob=0.5, exp_dec=0.1):
        self.learning_rate = learn_rate
        self.discount_factor = discount_fac
        self.exploration_prob = exp_prob
        self.exploration_decrease = exp_dec
        self.replay_buffer_size = replay_buffer_size
        return

    # Function to reset runtime parameters    
    def reset_params(self):
        self.current_reward = None
        self.current_action = None
        self.current_adv_vals = None
        self.current_state = None
        self.current_state_val = None
        return
import numpy as np
import tensorflow as tf
from tensorflow import keras
import random

class DQNagent():
    def __init__(self, id, deep_q_net = None):
        self.id = id
        
        # Setting up action variables
        self.action = np.array([False, False, False, False, False, False, False, False])            # Forward, Backward, Left, Right, Up, Down, Yaw left, Yaw Right

        self.experience = [np.zeros((1,224,224,3)), np.zeros(1), np.zeros(1), np.zeros((1,224,224,3))]
        self.experience_buffer = []

        # Set hyperparameters
        self.set_hypers()

        # Runtime parameters
        self.prev_state = None
        self.prev_action = None
        self.prev_reward = None
        self.prev_q_values = None

        self.current_state = None
        self.current_action = None
        self.current_reward = None
        self.current_q_values = None

        self.choice_maker = None

        self.no_exp = 0
        self.exp_count = 0

        # Setting up a Neural Network model(s)
        if deep_q_net is not None:
            self.deep_q_net = deep_q_net
        else:
            print("Deep Q Network not set")
            exit()
        return
    
    def choose_action(self):
        # Epsilon-greedy policy
        if np.random.rand() < self.exploration_prob:
            self.current_action = np.random.choice(len(self.action))
            self.choice_maker = "[RANDOM]"
        else:
            self.current_action = np.argmax(self.current_q_values)
            self.choice_maker = "[ESTIMATED]"
        return
    
    # Function to estimate the q values
    def estimate_q_values(self):
        self.current_q_values = self.deep_q_net.predict(self.current_state, verbose=0)
        return
    
    # Function to calculate the target Q values using the Bellman equation: Q(s,a)target = R + y * mean(Q(s,a))
    def calculate_target_q(self, reward, discount_factor, q_values):
        self.target_q_values = reward + discount_factor * np.mean(q_values)
        return
    
    # Function to update (fit) the networks
    def update_networks(self, num_epoch, current_state, target_q_values, set_verbose=0):
        self.deep_q_net.fit(current_state, target_q_values, epochs=num_epoch, verbose=set_verbose)
        return
    
    def train(self, no_exp, verbose=0):
        print("Optimisation in progress...")
        # Sample no_exp experiences from experience replay buffer
        sample_experience = None
        if no_exp >= len(self.experience_buffer):
            sample_experience = random.sample(self.experience_buffer, len(self.experience_buffer))
        else:
            sample_experience = random.sample(self.experience_buffer, no_exp)
        # Update neural networks
        for i in range(len(sample_experience)):
            print("Selected experience:")
            print("Previous state: ", sample_experience[i][0].shape)
            print("Current action: ", sample_experience[i][1])
            print("Current reward: ", sample_experience[i][2])
            print("Current state: ", sample_experience[i][3].shape)
            self.prev_state = sample_experience[i][0]       # Get s
            self.current_action = sample_experience[i][1]   # Get a
            self.current_reward = sample_experience[i][2]   # Get r
            self.current_state = sample_experience[i][3]    # Get s'
            self.estimate_q_values()                        # Estimate Q values
            self.calculate_target_q(self.current_reward, self.discount_factor, self.current_q_values)                       # Calculate target Q values
            self.update_networks(1, self.prev_state, self.target_q_values, set_verbose=verbose)    # Update networks
        return
    
    # Function to store the experience of a single train cycle (e = (s, a, r, s'))
    def store_experience(self):
        experience = [np.zeros((1,224,224,3)), np.zeros(1), np.zeros(1), np.zeros((1,224,224,3))]
        if self.exp_count >= self.replay_buffer_size:
            self.exp_count = 0

        # If first experience: store initial state twice
        if self.prev_state is None:
            experience[0] = self.current_state # Initial state as the current state
            experience[1] = np.array([self.current_action])
            experience[2] = np.array([self.current_reward])
            experience[3] = self.current_state # Initial state as the next state
            if self.no_exp < self.replay_buffer_size:
                self.experience_buffer.append(experience)
            else:
                self.experience_buffer[self.exp_count] = experience
                self.exp_count += 1
        else:
            experience[0] = self.prev_state
            experience[1] = self.current_action
            experience[2] = np.array([self.current_reward])
            experience[3] = self.current_state
            if self.no_exp < self.replay_buffer_size:
                self.experience_buffer.append(experience)
            else:
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
        if self.current_q_values is not None:
            self.prev_q_values = self.current_q_values
        # Update current state and reward
        self.current_state = next_state
        self.current_reward = next_reward
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
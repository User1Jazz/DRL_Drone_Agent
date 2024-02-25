import numpy as np
import random
import time
from tensorflow import keras

class DQN():
    def __init__(self, agent=None, main_net=None, target_net=None):
        # Make sure the required parameters are provided
        if agent == None:
            print("Agent not set")
            exit()
        if main_net == None or target_net == None:
            print("Main net OR target net not set\nMain net: ", main_net, "\nTarget net: ", target_net)
            exit()
        
        self.agent = agent          # Get the initializer
        self.main_net = main_net    # Get the main Q network
        self.target_net=target_net  # Get the target Q network
        self.experience_buffer = [] # Initialize experience buffer
        self.current_state=None     # Initialize current state
        self.update_state()         # Initialize other runtime variables
        self.set_hyperparams()      # Initialize hyperparameters
        self.compile_networks()     # Compile the networks
        self.choice_maker = "[UNKNOWN]"
        self.exp_count = 0          # Initialize experience counter
        return
    
    # Function to estimate the q values using the main Q network
    def estimate_q_values(self, state):
        self.q_values = self.main_net.predict(state, verbose=0)
        return
    
    # Function to estimate the target q values using the target Q network
    def estimate_target_q_values(self, state):
        self.target_q_values = self.target_net.predict(state, verbose=0)
        return
    
    # Function to store the experience (state, action, reward, next state, done)
    def store_experience(self):
        # Make sure exp_count (index pointer) does not go out of bounds
        if self.exp_count >= self.experience_buffer_size:
            self.exp_count = 0
        # Get experience record
        experience = [self.previous_state, self.current_action, self.current_reward, self.current_state, self.done]
        if len(self.experience_buffer) < self.experience_buffer_size:
            self.experience_buffer.append(experience)           # Append the experience if maximum not met
        else:
            self.experience_buffer[self.exp_count] = experience # Otherwise, replace previous record with new one
            self.exp_count += 1
        return
    
    # Function to update the main Q network using the target Q network
    def update_main_net(self, verbose=0):
        self.estimate_target_q_values(self.current_state)                                                                               # Estimate Q values
        self.target_q_values = np.array([self.current_reward + self.discount_factor * (1-self.done) * np.max(self.target_q_values)])    # Calculate Q values using Bellman equation
        self.main_net.fit(self.previous_state, self.target_q_values, epochs=1, verbose=verbose)                                         # Update main network
        return
    
    # Function to update the target Q network
    def update_target_net(self):
        target_weights = self.target_net.get_weights()                                          # Get current target net weights
        online_weights = self.main_net.get_weights()                                            # Get current main net weights
        # Apply soft update
        for i in range(len(target_weights)):
            target_weights[i] = (1-self.tau) * target_weights[i] + self.tau * online_weights[i]
        self.target_net.set_weights(target_weights)
        return
    
    # Function to update runtime variables
    def update_state(self, state=None, action=None, reward=None, done=None):
        self.previous_state = self.current_state    # Save previous state
        self.current_action = action                # Get new action
        self.current_reward = reward                # Get new reward
        self.current_state = state                  # Get new state
        self.done = done                            # Get done flag
        return
    
    # Function to choose an action
    def choose_action(self):
        # Epsilon-greedy policy
        if np.random.rand() < self.exploration_probability:
            self.current_action = np.random.choice(self.no_actions)
            self.choice_maker = "[RANDOM]"
        else:
            self.estimate_q_values(self.current_state)
            self.current_action = np.argmax(self.q_values)
            self.choice_maker = "[ESTIMATED]"
        return

    # Function to run the DQN algorithm
    def run(self, update_network=True, store_experience=True, verbose=0):
        self.update_state(self.agent.state, self.agent.action, self.agent.reward, self.agent.done)              # Get observation
        self.choose_action()
        self.agent.decode_action(verbose=verbose)                                                               # Choose action
        if update_network:
            time.sleep(0.1) # Wait a little bit for changes
            self.update_state(self.agent.state, self.agent.action, self.agent.reward, self.agent.done)          # Get observation
            self.update_main_net(verbose=verbose)                                                               # Update main net
            if store_experience:
                self.store_experience()                                                                         # Store experience
        return
    
    # Function to train the Q networks from the experience buffer
    def train(self, no_exp, verbose=0):
        if len(self.experience_buffer) > 0:
            # Sample experiences
            if no_exp >= len(self.experience_buffer):
                sample_experience = random.sample(self.experience_buffer, len(self.experience_buffer))
            else:
                sample_experience = random.sample(self.experience_buffer, no_exp)
            # Replay experiences
            for i in range(len(sample_experience)):
                self.current_state = sample_experience[i][0]
                self.update_state(state=sample_experience[i][3],
                                  action=sample_experience[i][1],
                                  reward=sample_experience[i][2],
                                  done=sample_experience[i][4])
                self.update_main_net(verbose=verbose)
            self.update_target_net()
        return
    
    # Function to set the hyperparameters
    def set_hyperparams(self, no_actions = 1, experience_buffer_size = 500, learning_rate=0.001, metrics=None, discount_factor=0.5, exploration_probability= 0.5, tau=0.001):
        self.no_actions = no_actions
        self.experience_buffer_size = experience_buffer_size
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.metrics = metrics
        self.discount_factor = discount_factor
        self.exploration_probability = exploration_probability
        self.tau = tau
        return
    
    def decrease_exploration_probability(self, decrease_factor):
        self.exploration_probability -= decrease_factor
        return
    
    # Function to compile networks
    def compile_networks(self):
        self.main_net.compile(optimizer=self.optimizer, loss="mse", metrics=self.metrics)
        self.target_net.compile(optimizer=self.optimizer, loss="mse", metrics=self.metrics)
        return
    
    # Function to save the main and target Q networks to a file
    def save_networks(self, main_path=None, target_path=None):
        if main_path != None:
            self.main_net.save(main_path)
            print("Main Q network saved")
        if target_path != None:
            self.target_net.save(target_path)
            print("Target Q network saved")
        return
    
    # Function to load the main and target Q network from a file
    def load_networks(self, main_path=None, target_path=None):
        if main_path != None:
            self.main_net = keras.models.load_model(main_path)
            print("Main Q network loaded")
        if target_path != None:
            self.target_net = keras.models.load_model(target_path)
            print("Target Q network loaded")
        return
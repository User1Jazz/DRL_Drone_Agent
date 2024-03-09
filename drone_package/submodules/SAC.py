import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras.backend as K
import random
import time
import matplotlib.pyplot as plt

"""
Neural Networks:
Prediction Q net (Part of the Critic)
Target State Val net (Part of the Critic)
Policy Net (Actor)

Replay buffer

Loss Functions:
Policy Loss Function (For policy net)
Q Loss Function (For prediction Q net)
Value Loss Function (For Target State Val net)
Policy Entropy (For Policy net)
"""

class SAC():
    def __init__(self, agent=None, P_net=None, Q_net=None, V_net=None):
        # Make sure the required parameters are provided
        if agent == None:
            print("Agent not set")
            exit()
        if P_net == None or Q_net == None or V_net == None:
            print("Policy net OR Q net OR V net not set\nPolicy net: ", P_net, "\nQ net: ", Q_net, "\nV net: ", V_net)
            exit()
        
        self.agent = agent          # Get the initializer
        self.P_net = P_net          # Get the policy network
        self.Q_net = Q_net          # Get the prediction Q network
        self.target_Q_net = Q_net
        self.V_net= V_net           # Get the target Q network
        self.experience_buffer = [] # Initialize experience buffer
        self.update_state()         # Initialize runtime variables
        self.set_hyperparams()      # Initialize hyperparameters
        self.compile_networks()     # Compile the networks
        self.choice_maker = "[UNKNOWN]"
        self.exp_count = 0          # Initialize experience counter
        self.target_update_count = 0# Initialize counter that counts to next update of the target q network
        self.episode_rewards = []   # Initialize the list of rewards for current episode
        self.rewards = []           # Initialize the list of rewards (for analysis)
        return
    
    # Function to update runtime variables
    def update_state(self, state=None, action=None, reward=None, done=None):
        self.previous_state = self.current_state    # Save previous state
        self.current_action = action                # Get new action
        self.current_reward = reward                # Get new reward
        self.current_state = state                  # Get new state
        self.done = done                            # Get done flag
        return
    
    # Function to set the hyperparameters
    def set_hyperparams(self, no_actions = 1, experience_buffer_size = 500, target_q_update_frequency = 5, learning_rate=0.001, metrics=None, discount_factor=0.5, exploration_probability= 0.5, tau=0.001):
        self.no_actions = no_actions
        self.experience_buffer_size = experience_buffer_size
        self.target_q_update_frequency = target_q_update_frequency
        self.optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
        self.metrics = metrics
        self.discount_factor = discount_factor
        self.exploration_probability = exploration_probability
        self.tau = tau
        return
    
    def compile_networks(self, alpha=0.01):
        self.alpha = alpha
        self.P_net.compile(optimizer=self.optimizer, loss=policy_loss(alpha=self.alpha), metrics=self.metrics)
        self.Q_net.compile(optimizer=self.optimizer, loss=q_loss(), metrics=self.metrics)
        self.target_Q_net.compile(optimizer=self.optimizer, loss=q_loss(), metrics=self.metrics)
        self.V_net.compile(optimizer=self.optimizer, loss=v_loss(alpha=self.alpha), metrics=self.metrics)
        return
    
    # Function to estimate Q values given the state
    def estimate_q_values(self, state):
        self.q_values = self.Q_net.predict(state, verbose=0)
        return
    
    # Function to estimate Q' values given the state
    def estimate_target_q_values(self, state):
        self.target_q_values = self.target_Q_net.predict(state, verbose=0)
        return
    
    # Function to estimate V value given the state
    def estimate_state_value(self, state):
        self.current_state_value = self.V_net.predict(state, verbose=0)
        return
    
    # Function to estimate Policy π values given the state
    def estimate_policy(self, state):
        self.current_policy = self.P_net.predict(state, verbose=0)
        return
    
    # Function to update Policy network
    def update_p_net(self, verbose=0):
        self.estimate_q_values(self.previous_state)                                         # Estimate Q values for the current state
        self.main_net.fit(self.previous_state, self.q_values, epochs=1, verbose=verbose)    # Update Policy network
        return
    
    # Function to update Q network
    def update_q_net(self, verbose=0):
        self.estimate_state_value(self.current_state)                                   # Estimate state value for the next state
        target = self.current_reward + self.discount_factor * self.current_state_value  # Calculate target (r + γ * V'(s';Φ))
        self.Q_net.fit(self.previous_state, target, epochs=1, verbose=verbose)          # Update Q network
        return
    
    # Function to update State Value (V) network
    def update_v_net(self, verbose=0):
        self.estimate_q_values(self.previous_state)                             # Estimate Q values for the current state
        self.estimate_policy(self.previous_state)                               # Estimate policy for the current state
        policy_entropy = self.alpha * K.log(self.current_policy)                # Calculate policy entropy
        target = self.q_values - policy_entropy                                 # Calculate target (Ea~π(.|s)[Q(s,a;θ)] - α * log(π(a|s;Φ )))
        self.V_net.fit(self.previous_state, target, epochs=1, verbose=verbose)  # Update V network
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
        # Stochastic Policy Sampling strategy
        self.estimate_policy(self.current_state)
        self.current_action = tf.random.categorical(self.current_policy, 1)
        self.choice_maker = "[ESTIMATED]"
        return

    # Function to run the DQN algorithm (do not use update_network [I just left it there for experimental purposes])
    def run(self, update_network=False, store_experience=True, verbose=0):
        self.update_state(self.agent.state, self.agent.action, self.agent.reward, self.agent.done)              # Get observation
        self.episode_rewards.append(self.current_reward)                                                        # Store reward to the episode rewards list
        self.choose_action()
        self.agent.decode_action(verbose=verbose)                                                               # Choose action
        if update_network or store_experience:
            time.sleep(0.1) # Wait a little bit for changes
            self.update_state(self.agent.state, self.agent.action, self.agent.reward, self.agent.done)          # Get observation
        if update_network:
            self.update_networks(verbose=verbose)                                                               # Update main net
        if store_experience:
            self.store_experience()                                                                             # Store experience
        return
    
    def update_networks(self, verbose=0):
        self.update_q_net(verbose=verbose)
        self.update_p_net(verbose=verbose)
        self.update_v_net(verbose=verbose)
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
                self.update_networks(verbose=verbose)
            #self.update_target_q_net()
        return
    
    # Function to set the hyperparameters
    def set_hyperparams(self, no_actions = 1, experience_buffer_size = 500, target_q_update_frequency = 5, learning_rate=0.001, metrics=None, discount_factor=0.5, exploration_probability= 0.5, tau=0.001):
        self.no_actions = no_actions
        self.experience_buffer_size = experience_buffer_size
        self.target_q_update_frequency = target_q_update_frequency
        self.optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
        self.metrics = metrics
        self.discount_factor = discount_factor
        self.exploration_probability = exploration_probability
        self.tau = tau
        return
    
    # Function to save the neural networks to a file
    def save_networks(self, policy_path=None, q_path = None, v_path = None, target_q_path=None):
        if policy_path != None:
            self.P_net.save(policy_path)
            print("Q network saved")
        if q_path != None:
            self.Q_net.save(q_path)
            print("Q network saved")
        if v_path != None:
            self.V_net.save(v_path)
            print("V network saved")
        if target_q_path != None:
            self.target_Q_net.save(target_q_path)
            print("Target Q network saved")
        return
    
    # Function to load the neural networks from a file
    def load_networks(self, policy_path = None, q_path=None, v_path = None, target_q_path=None):
        if q_path != None:
            self.P_net = keras.models.load_model(policy_path)
            print("Q network loaded")
        if q_path != None:
            self.Q_net = keras.models.load_model(q_path)
            print("Q network loaded")
        if v_path != None:
            self.V_net = keras.models.load_model(v_path)
            print("Q network loaded")
        if target_q_path != None:
            self.target_Q_net = keras.models.load_model(target_q_path)
            print("Target Q network loaded")
        return
    
    # Function to save episode rewards into the rewards list and clear the episode rewards list
    def store_episode_rewards(self):
        if len(self.episode_rewards) == 0:
            print("ALERT: Episode rewards list is empty! Skipping store operation.")
            return
        self.rewards.append(self.episode_rewards)
        self.episode_rewards = []
        print("Stored reward list of size ", len(self.rewards[-1]))
        return
    
    # Function to save rewards chart
    def save_reward_chart(self, reward_path):
        # Get number of episodes
        no_episodes = [i for i in range(len(self.rewards))]
        # Initialize lists
        average_rewards = []
        median_rewards = []
        min_rewards = []
        max_rewards = []
        # Get average reward per episode
        for reward in self.rewards:
            average_rewards.append(np.average(reward))
        # Get median reward per episode
        for reward in self.rewards:
            median_rewards.append(np.median(reward))
        # Get minimum reward per episode
        for reward in self.rewards:
            min_rewards.append(np.min(reward))
        # Get maximum reward per episode
        for reward in self.rewards:
            max_rewards.append(np.max(reward))
        # Format x axis
        xint = range(0, len(no_episodes))
        plt.xticks(xint)
        # Plot
        plt.plot(no_episodes, average_rewards, label='Average Reward', color='red')  # Plot average reward per episode with red line (data is tagged as 'Average Reward')
        plt.plot(no_episodes, median_rewards, label='Median Reward', color='blue')  # Plot median reward per episode with blue line (data is tagged as 'Median Reward')
        plt.xlabel('Episode (n-1)')                                               # Set label for X axis (episodes)
        plt.ylabel('Reward')                                                      # Set label for Y axis (reward values)
        plt.title('Rewards per Episode')                                          # Set chart title
        plt.fill_between(no_episodes, min_rewards, max_rewards, alpha=0.2)           # Show range between min and max rewards
        plt.legend()                                                              # Show chart legend (data tags)
        plt.savefig(reward_path + "DoubleDQN_Rewards.jpg")                        # Save chart
        return

# Lp(Φ) = α * log(π(a|s;Φ)) - Q(s,a;θ)
#       alpha     y_pred       y_true
def policy_loss(alpha=0.01):
    def loss(y_true, y_pred):
        policy_entropy = alpha * K.log(y_pred)
        p_loss = policy_entropy - y_true
        return p_loss
    return loss

# TD(θ)=1/2E(s,a,r,s')~D[(Q(s,a;θ) - (r + γ * V'(s';Φ)))^2]
#                         y_pred     |_____y_true_____|
def q_loss():
    def loss(y_true, y_pred):
        TD = 1/2 * (K.pow((y_pred - y_true), 2))
        return TD
    return loss

# Lv(ψ) = 1/2Es~D[(V(s,ψ) - Ea~π(.|s)[Q(s,a;θ)] - α * log(π(a|s;Φ )))^2]
#                  y_pred   |________________y_true_________________|
def v_loss():
    def loss(y_true, y_pred):
        V_loss = 1/2 * (K.pow((y_pred - y_true), 2))
        return V_loss
    return loss
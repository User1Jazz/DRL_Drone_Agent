import numpy as np
import random
import time
import tensorflow as tf
from tensorflow import keras
import keras.backend as K
import matplotlib.pyplot as plt

class DoubleDQN():
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
        self.target_update_count = 0# Initialize counter that counts to next update of the target q network
        self.episode_rewards = []   # Initialize the list of rewards for current episode
        self.rewards = []           # Initialize the list of rewards (for analysis)
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
        #self.estimate_target_q_values(self.current_state)                                                                                    # Estimate target Q values
        #self.estimate_q_values(self.previous_state)                                                                                          # Estimate Q values
        #self.target_q_values[np.arange(len(self.target_q_values)), np.argmax(self.q_values, axis=1)] = 0                                       # Update target Q values using the action selected by the main Q net
        #self.target_q_values = np.array([self.current_reward + self.discount_factor * (1-self.done) * np.max(self.target_q_values, axis=1)]) # Calculate Q values using Bellman equation
        #self.main_net.fit(self.previous_state, self.target_q_values, epochs=1, verbose=verbose)                                              # Update main network
        with tf.GradientTape() as tape:
            action_masks = tf.one_hot(self.current_action, depth=self.no_actions)                       # Compute action masks
            q_values = tf.reduce_sum(self.main_net(self.previous_state) * action_masks, axis=1)         # Get Q values for current state
            next_actions = tf.argmax(self.main_net(self.current_state), axis =1)                        # Get actions for next state
            next_actions = tf.one_hot(next_actions, self.no_actions)                                    # Get one-hot vector out of actions
            next_q_values = tf.reduce_sum(self.target_net(self.current_state) * next_actions, axis=1)   # Calculate Q values for the next state
            target_q_values = self.current_reward + (1-self.done) * next_q_values                       # Calculate target Q values
            target_q_values = tf.stop_gradient(target_q_values)                                         # Detach target Q values from the graph
            loss = tf.reduce_mean(tf.square(q_values - target_q_values))                                # Calculate loss
        gradients = tape.gradient(loss, self.main_net.trainable_variables)                              # Calculate gradients
        self.optimizer.apply_gradients(zip(gradients, self.main_net.trainable_variables))               # Update network parameters
        if verbose > 0:
            print("Loss: ", loss)
        return
    
    # Function to update the target Q network
    def update_target_net(self):
        target_weights = self.target_net.get_weights()                                          # Get current target net weights
        online_weights = self.main_net.get_weights()                                            # Get current main net weights
        # Apply soft update
        self.target_update_count += 1
        if self.target_update_count == self.target_q_update_frequency:
            print("Updating target q network...")
            for i in range(len(target_weights)):
                target_weights[i] = (1-self.tau) * target_weights[i] + self.tau * online_weights[i]
            self.target_net.set_weights(target_weights)
            self.target_update_count = 0
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
            self.update_main_net(verbose=verbose)                                                               # Update main net
        if store_experience:
            self.store_experience()                                                                             # Store experience
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
    
    def decrease_exploration_probability(self, decrease_factor):
        self.exploration_probability -= decrease_factor
        return
    
    # Function to compile networks
    def compile_networks(self):
        self.main_net.compile(optimizer=self.optimizer, loss=loss_with_entropy(alpha=0.01, temperature=1.0), metrics=self.metrics)
        self.target_net.compile(optimizer=self.optimizer, loss=loss_with_entropy(alpha=0.01, temperature=1.0), metrics=self.metrics)
        self.target_net.set_weights(self.main_net.get_weights())
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
    def save_reward_chart(self, save_path):
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
        plt.savefig(save_path)                                                    # Save chart
        return

def loss_with_entropy(alpha=0.01, temperature=1.0):
    def loss(y_true, y_pred):
        mse_loss = K.mean(K.square(y_true - y_pred))
        action_probs = K.softmax(y_pred / temperature)
        entropy = -K.sum(action_probs * K.log(action_probs + 0.000001))
        total_loss = mse_loss + alpha * entropy
        return total_loss
    return loss
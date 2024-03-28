import numpy as np
from matplotlib import pyplot as plt

class ReplayBuffer(object):
    def __init__(self, mem_size, input_shape, n_actions, discrete=False):
        self.mem_size = mem_size
        self.mem_cntr = 0
        self.input_shape = input_shape
        self.discrete = discrete
        self.state_memory = np.zeros(((self.mem_size,) + self.input_shape))
        self.new_state_memory = np.zeros(((self.mem_size,) + self.input_shape))
        dtype = np.int8 if self.discrete else np.float32
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=dtype)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)
        return
    
    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1- int(done)
        if self.discrete:
            actions = np.zeros(self.action_memory.shape[1])
            actions[action] = 1.0
            self.action_memory[index] = actions
        else:
            self.action_memory[index] = action
        self.mem_cntr += 1
        return
    
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]
        return states, actions, rewards, states_, terminal

class RewardStorage(object):
    def __init__(self):
        self.episode_rewards = []
        self.rewards = []
        return
    
    def add_reward(self, reward):
        self.episode_rewards.append(reward)
        return
    
    def store_episode_rewards(self):
        if len(self.episode_rewards) == 0:
            print("ALERT: Episode rewards list is empty! Skipping store operation.")
            return
        self.rewards.append(self.episode_rewards)
        self.episode_rewards = []
        print("Stored reward list of size ", len(self.rewards[-1]))
        return
    
    def save_reward_chart(self, save_path, color='red', alpha=0.2, reward_gap=100):
        n_episodes = [i for i in range(len(self.rewards))]
        avg_rewards = []
        min_rewards = []
        max_rewards = []
        for episode_rewards in self.rewards:
            avg_rewards.append(np.median(episode_rewards))
        for episode_rewards in self.rewards:
            min_rewards.append(np.min(episode_rewards))
        for episode_rewards in self.rewards:
            max_rewards.append(np.max(episode_rewards))
        xint = range(0, len(n_episodes))
        plt.xticks(xint)
        plt.plot(n_episodes, avg_rewards, label='Median Reward', color=color)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Rewards per Episode')
        plt.fill_between(n_episodes, min_rewards, max_rewards, alpha=alpha)
        plt.xticks(range(0, n_episodes, reward_gap))
        plt.legend()
        plt.savefig(save_path)
        return
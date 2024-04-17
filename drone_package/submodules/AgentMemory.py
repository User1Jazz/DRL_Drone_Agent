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
    
    def save_reward_chart_median(self, save_path, color='red', alpha=0.2, reward_gap=100):
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
        plt.xticks(range(0, len(n_episodes), reward_gap))
        plt.legend()
        plt.savefig(save_path)
        plt.clf()
        return
    
    def save_reward_chart_mean(self, save_path, color='red', alpha=0.2, reward_gap=100):
        n_episodes = [i for i in range(len(self.rewards))]
        avg_rewards = []
        min_rewards = []
        max_rewards = []
        for episode_rewards in self.rewards:
            avg_rewards.append(np.mean(episode_rewards))
        for episode_rewards in self.rewards:
            min_rewards.append(np.min(episode_rewards))
        for episode_rewards in self.rewards:
            max_rewards.append(np.max(episode_rewards))
        xint = range(0, len(n_episodes))
        plt.xticks(xint)
        plt.plot(n_episodes, avg_rewards, label='Mean Reward', color=color)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Rewards per Episode')
        plt.fill_between(n_episodes, min_rewards, max_rewards, alpha=alpha)
        plt.xticks(range(0, len(n_episodes), reward_gap))
        plt.legend()
        plt.savefig(save_path)
        plt.clf()
        return
    
    def save_rewards_list(self, save_path):
        array_string = "[[" + ",\n[".join([",\t".join(map(str, row))+"]" for row in self.rewards])+"]"
 
        # Write to File
        with open(save_path, 'w') as file:
            file.write(array_string)
        return
    
class LossMemory():
    def __init__(self, is_SAC=False):
        if not is_SAC:
            self.losses =[]
        else:
            self.value_losses = []
            self.actor_losses = []
            self.critic_1_losses = []
            self.critic_2_losses = []
        return
    
    def add_loss(self, loss_val):
        if loss_val == None:
            return
        self.losses.append(loss_val)
        return
    
    def add_sac_losses(self, loss_vals):
        if loss_vals == None:
            return
        self.value_losses.append(loss_vals[0])
        self.actor_losses.append(loss_vals[1])
        self.critic_1_losses.append(loss_vals[2])
        self.critic_2_losses.append(loss_vals[3])
        return
    
    def save_loss_chart(self, save_path, color="red", gap=500):
        n_records =[i for i in range(len(self.losses))] 
        plt.xticks(n_records)
        plt.plot(n_records, self.losses, label="Loss", color=color)
        plt.xlabel('Fit Call')
        plt.ylabel('Loss Value')
        plt.title('Loss Values')
        plt.xticks(range(0, len(n_records), gap))
        plt.legend()
        plt.savefig(save_path)
        plt.clf()
        return
    
    def save_sac_losses_chart(self, save_path, color=["red", "blue", "orange", "yellow"], gap=500):
        n_records_value =[i for i in range(len(self.value_losses))] 
        n_records_actor =[i for i in range(len(self.actor_losses))]
        n_records_critic_1 =[i for i in range(len(self.critic_1_losses))]
        n_records_critic_2 =[i for i in range(len(self.critic_2_losses))]
        plt.xticks(n_records_value)
        plt.plot(n_records_value, self.value_losses, label="Value Loss", color=color[0])
        plt.plot(n_records_actor, self.actor_losses, label="Actor Loss", color=color[1])
        plt.plot(n_records_critic_1, self.critic_1_losses, label="Critic 1 Loss", color=color[2])
        plt.plot(n_records_critic_2, self.critic_2_losses, label="Critic 2 Loss", color=color[3])
        plt.xlabel('Fit Call')
        plt.ylabel('Loss Value')
        plt.title('Loss Values')
        plt.xticks(range(0, len(n_records_value), gap))
        plt.legend()
        plt.savefig(save_path)
        plt.clf()
        return

    def save_loss_list(self, save_path):
        res = "[" + ", ".join(map(str, self.losses)) + "]"
        with open(save_path, 'w') as file:
            file.write(res)
        return
    
    def save_sac_loss_lists(self, save_path):
        res = "Value: [" + ", ".join(map(str, self.value_losses)) + "]"
        res += "\nActor: [" + ", ".join(map(str, self.actor_losses)) + "]"
        res += "\nCritic_1: [" + ", ".join(map(str, self.critic_1_losses)) + "]"
        res += "\nCritic_2: [" + ", ".join(map(str, self.critic_2_losses)) + "]"
        with open(save_path, 'w') as file:
            file.write(res)
        
        return

class PositionMemory(object):
    def __init__(self):
        self.pos_x = []
        self.pos_y = []
        self.pos_z = []
        return
    
    def add_position(self, x, y, z):
        self.pos_x.append(x)
        self.pos_y.append(y)
        self.pos_z.append(z)
        return
    
    def save_position_chart(self, save_path, chart_type="3d", color="red"):
        if chart_type == "3d":
            fig = plt.figure()
            ax1 = fig.add_subplot(projection='3d')
            ax1.plot3D(self.pos_x, self.pos_y, self.pos_z, c = color)
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            ax1.set_zlabel('z')
            plt.title('chart')
            #plt.legend()
            plt.savefig(save_path)
            plt.clf()
        if chart_type == "2d":
            plt.plot(self.pos_x, self.pos_y, color=color)
            plt.xticks()
            plt.yticks()
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title('Agent Navigation')
            #plt.legend()
            plt.savefig(save_path)
            plt.clf()
        return
    
    def save_position_values(self, save_path):
        pos = "Pos_x: [" + ", ".join(map(str, self.pos_x)) + "]"
        pos += "\nPos Y: [" + ", ".join(map(str, self.pos_y)) + "]"
        pos += "\nPos Z: [" + ", ".join(map(str, self.pos_z)) + "]"
        with open(save_path, 'w') as file:
            file.write(pos)
        return
from keras.layers import Conv2D, Flatten, Dense, Activation
from keras.models import Sequential, load_model
from keras.optimizers import Adam
import numpy as np
from .AgentMemory import ReplayBuffer

class Agent(object):
    def __init__(self, alpha, gamma, n_actions, epsilon, batch_size, input_dims, epsilon_dec=0.996, epsilon_end = 0.01, mem_size=1000, fname='dqn_model.keras'):
        self.choice_maker = ["UNKNOWN"]
        self.action_space = [i for i in range(n_actions)]
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.model_file = fname
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions, discrete=True)
        self.q_eval = self.build_dqn(alpha, n_actions, input_dims, 256, 256)
        for layer in self.q_eval.layers:
            print("Layer ", layer.name, " input shape: ", layer.input_shape)
        return
    
    def build_dqn(self, lr, n_actions, input_dims, fc1_dims, fc2_dims):
        model = Sequential([
            Conv2D(32, (6,6), input_shape=input_dims),
            Activation('relu'),
            Conv2D(32, (3,3), input_shape=input_dims),
            Activation('relu'),
            Flatten(),
            Dense(fc1_dims),
            Activation('relu'),
            Dense(fc2_dims),
            Activation('relu'),
            Dense(n_actions)
        ])
        model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
        return model
    
    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)
        return
    
    def choose_action(self, state):
        state = state[np.newaxis, :]
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
            self.choice_maker = "[RANDOM]"
        else:
            actions = self.q_eval.predict(state, verbose=0)
            action = np.argmax(actions)
            self.choice_maker = "[ESTIMATED]"
        return action
    
    def learn(self, verbose=0):
        if self.memory.mem_cntr < self.batch_size:
            return
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        action_values = np.array(self.action_space, dtype=np.int8)
        action_indices = np.dot(action, action_values)
        q_eval = self.q_eval.predict(state, verbose=0)
        q_next = self.q_eval.predict(new_state, verbose=0)
        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        q_target[batch_index, action_indices] = reward + self.gamma * np.max(q_next, axis=1) * done
        _ = self.q_eval.fit(state, q_target, verbose=verbose)
        self.epsilon = self.epsilon * self.epsilon_dec if self.epsilon > self.epsilon_min else self.epsilon_min
        return
    
    def save_model(self):
        self.q_eval.save(self.model_file)
        return
    
    def load_model(self):
        self.q_eval = load_model(self.model_file)
        return
import tensorflow as tf
import tensorflow_probability as tfp
import keras
from keras.layers import Conv2D, Dense, Flatten
from keras.optimizers import Adam
import os
from AgentMemory import ReplayBuffer

class CriticNetwork(keras.Model):
    def __init__(self, n_actions, input_dims, fc1_dims=256, fc2_dims=256, name='critic', chkpt_dir='tmp/sac'):
        super(CriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self. fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
        self.conv_1 = Conv2D(32, (6,6), activation='relu', input_shape=input_dims)
        self.conv_2 = Conv2D(64, (3,3), activation='relu')
        self.flatten = Flatten()
        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.q = Dense(1, activation=None)
        return
    
    def call(self, state, action):
        action_value = self.conv_1(tf.concat([state,action], axis=1))
        action_value = self.conv_2(action_value)
        action_value = self.flatten(action_value)
        action_value = self.fc1(action_value)
        action_value = self.fc2(action_value)
        q = self.q(action_value)
        return q
    
class ValueNetwork(keras.Model):
    def __init__(self, n_actions, input_dims, fc1_dims=256, fc2_dims=256, name='value', chkpt_dir='tmp/sac'):
        super(ValueNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
        self.conv_1 = Conv2D(32, (6,6), activation='relu', input_shape=input_dims)
        self.conv_2 = Conv2D(64, (3,3), activation='relu')
        self.flatten = Flatten()
        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.v = Dense(1, activation=None)
        return
    
    def call(self, state):
        state_value = self.conv_1(state)
        state_value = self.conv_2(state_value)
        state_value = self.flatten(state_value)
        state_value = self.fc1(state_value)
        state_value = self.fc2(state_value)
        v = self.v(state_value)
        return v
    
class ActorNetwork(keras.Model):
    def __init__(self, max_action, input_dims, fc1_dims=256, fc2_dims=256, name='actor', n_actions=2, chkpt_dir='tmp/sac'):
        super(ActorNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self. fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
        self.max_action = max_action
        self.noise = 1e-6

        self.conv_1 = Conv2D(32, (6,6), activation='relu', input_shape=input_dims)
        self.conv_2 = Conv2D(64, (3,3), activation='relu')
        self.flatten = Flatten()
        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.mu = Dense(self.n_actions, activation=None)
        self.sigma = Dense(self.n_actions, activation=None)
        return
    
    def call(self, state):
        prob = self.conv_1(state)
        prob = self.conv_2(prob)
        prob = self.flatten(prob)
        prob = self.fc1(prob)
        prob = self.fc2(prob)
        mu = self.mu(prob)
        sigma = self.sigma(prob)
        sigma = tf.clip_by_value(sigma, self.noise, 1)
        return mu, sigma
    
    def sample_normal(self, state):
        mu, sigma = self.call(state)
        probabilities = tfp.distributions.Normal(mu, sigma)
        actions = probabilities.sample()
        action = tf.math.tanh(actions) * self.max_action
        log_probs = probabilities.log_prob(actions)
        log_probs -= tf.math.log(1-tf.math.pow(action, 2) + self.noise)
        log_probs = tf.math.reduce_sum(log_probs, axis=1, keepdims=True)
        return action, log_probs

class Agent(object):
    def __init__(self, alpha=0.003, beta=0.003, input_dims=8, max_action=1.0, gamma=0.99, n_actions=2, mem_size=1000000, tau=0.005, layer1_size=256, layer2_size=256, batch_size=256, reward_scale=2):
        self.choice_maker = "[UNKNOWN]"
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.actor = ActorNetwork(n_actions=n_actions, name='actor', max_action=max_action)
        self.critic_1 = CriticNetwork(n_actions=n_actions, name='critic_1')
        self.critic_2 = CriticNetwork(n_actions=n_actions, name='critic_2')
        self.value = ValueNetwork(n_actions=n_actions, name='value')
        self.target_value = ValueNetwork(n_actions=n_actions, name='target_value')

        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic_1.compile(optimizer=Adam(learning_rate=beta))
        self.critic_2.compile(optimizer=Adam(learning_rate=beta))
        self.value.compile(optimizer=Adam(learning_rate=beta))
        self.target_value.compile(optimizer=Adam(learning_rate=beta))

        self.scale = reward_scale
        self.update_network_parameters(tau=1)
        return
    
    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation])
        actions, _ = self.actor.sample_normal(state)
        self.choice_maker = "[ESTIMATED]"
        return actions[0]
    
    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)
        return
    
    def update_network_parameters(self, tau=None):
        if tau == None:
            tau = self.tau
        weights = []
        targets = self.target_value.get_weights()
        for i, weight in enumerate(self.value.get_weights()):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_value.set_weights(weights)
        return
    
    def save_models(self, path):
        print("Saving models...")
        self.actor.save_weights(path+"actor_sac")
        self.critic_1.save_weights(path+"critic_2_sac")
        self.critic_2.save_weights(path+"critic_1_sac")
        self.value.save_weights(path+"value_sac")
        self.target_value.save_weights(path+"target_value_sac")
        print("Save complete")
        return
    
    def load_models(self, path):
        print("Loading models...")
        self.actor.load_weights(path+"actor_sac")
        self.critic_1.load_weights(path+"critic_1_sac")
        self.critic_2.load_weights(path+"critic_2_sac")
        self.value.load_weights(path+"value_sac")
        self.target_value.load_weights(path+"target_value_sac")
        print("Models loaded")
        return
    
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        states = tf.convert_to_tensor(state, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)

        with tf.GradientTape() as tape:
            value = tf.squeeze(self.value(states), 1)
            value_ = tf.squeeze(self.target_value(states_), 1)
            current_policy_actions, log_probs = self.actor.sample_normal(states)
            log_probs = tf.squeeze(log_probs, 1)
            q1_new_policy = self.critic_1(states, current_policy_actions)
            q2_new_policy = self.critic_2(states, current_policy_actions)
            critic_value = tf.squeeze(tf.math.minimum(q1_new_policy, q2_new_policy), 1)
            value_target = critic_value - log_probs
            value_loss = 0.5 * keras.losses.MSE(value, value_target)
        value_network_gradient = tape.gradient(value_loss, self.value.trainable_variables)
        self.value.optimizer.apply_gradients(zip(value_network_gradient, self.value.trainable_variables))

        with tf.GradientTape() as tape:
            new_policy_actions, log_probs = self.actor.sample_normal(states)
            log_probs = tf.squeeze(log_probs, 1)
            q1_new_policy = self.critic_1(states, current_policy_actions)
            q2_new_policy = self.critic_2(states, current_policy_actions)
            critic_value = tf.squeeze(tf.math.minimum(q1_new_policy, q2_new_policy), 1)
            actor_loss = log_probs - critic_value
            actor_loss = tf.math.reduce_mean(actor_loss)
        actor_network_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_network_gradient, self.actor.trainable_variables))

        with tf.GradientTape(persistent=True) as tape:
            q_hat = self.scale * reward + self.gamma * value_ * (1-done)
            q1_old_policy = tf.squeeze(self.critic_1(state, action), 1)
            q2_old_policy = tf.squeeze(self.critic_2(state, action), 1)
            critic_1_loss = 0.5 * keras.losses.MSE(q1_old_policy, q_hat)
            critic_2_loss = 0.5 * keras.losses.MSE(q2_old_policy, q_hat)
        critic_1_network_gradient = tape.gradient(critic_1_loss, self.critic_1.trainable_variables)
        critic_2_network_gradient = tape.gradient(critic_2_loss, self.critic_2.trainable_variables)
        self.critic_1.optimizer.apply_gradients(zip(critic_1_network_gradient, self.critic_1.trainable_variables))
        self.critic_2.optimizer.apply_gradients(zip(critic_2_network_gradient, self.critic_2.trainable_variables))

        self.update_network_parameters()
        return
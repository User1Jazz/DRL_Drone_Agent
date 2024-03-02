from tensorflow import GradientTape as tape
from tensorflow import reduce_mean
from keras import layers, models

class DuelingNet(models.Model):
    def __init__(self, num_actions):
        super(DuelingNet, self).__init__()
        self.shared_layers = []
        self.state_value = layers.Dense(1, name='state_value')
        self.advantage_value = layers.Dense(num_actions, name='advantage_value')
        return
    
    def add_layer(self, layer):
        self.shared_layers.append(layer)
        return
    
    def compile(self, optimizer, loss, metrics):
        super(DuelingNet,  self).compile(metrics=metrics)
        self.optimizer = optimizer
        self.loss = loss
        return
    
    def predict(self, input, verbose=0):
        x = input
        for layer in self.shared_layers:
            x = layer(x)
        state_value = self.state_value(x)
        advantage_value = self.advantage_value(x)
        q_values = state_value + (advantage_value - reduce_mean(advantage_value, axis=1, keepdims=True))
        return q_values
    
    def fit(self, input, target, verbose=0):
        predicted = self.predict(input)
        loss = self.loss(target, predicted)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return
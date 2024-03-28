import numpy as np

class Environment(object):
    def __init__(self, state_shape):
        self.state = np.zeros(state_shape)
        self.reward = 0
        self.done = False
        return
    
    def reset(self):
        return self.state
    
    def step(self):
        return self.state, self.reward, self.done
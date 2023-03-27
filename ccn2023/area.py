import numpy as np

class Area:
    def __init__(self, num_neurons=1000, cap_size=41, p=0.03, beta=0.05):
        self.num_neurons = num_neurons
        self.recurrent_p = p
        self.cap_size = cap_size
        self.beta = beta
        self.activations = np.zeros(self.num_neurons)
        self.recurrent_connections = np.empty((self.num_neurons, self.num_neurons))
    
    def initialize_recurrent_connections(self):
        self.recurrent_connections = np.random.binomial(1,self.recurrent_p,size=(self.num_neurons,self.num_neurons)).astype("float64")
        np.fill_diagonal(self.recurrent_connections, 0)

    def clear_activations(self):
        self.activations = np.zeros(self.num_neurons)
    
    def get_activations(self):
        return np.copy(self.activations)
    
    def get_recurrent_connections(self):
        return np.copy(self.recurrent_connections)

    def set_activations(self, activations):
        assert activations.shape[0] == self.num_neurons, "Dimension mismatch between Area and activity"
        self.activations = activations
    
    def set_recurrent_connections(self, W):
        assert W.shape == (self.num_neurons, self.num_neurons)
        self.recurrent_connections = np.copy(W)

    def normalize_weights(self):
        self.recurrent_connections /= self.recurrent_connections.sum(axis=1)[:,np.newaxis]
    
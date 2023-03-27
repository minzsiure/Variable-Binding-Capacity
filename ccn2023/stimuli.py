from turtle import back
import numpy as np

class StimulusGenerator:
    def __init__(self, num_neurons=1000, cap_size=41, coreset_p=0.9, background_p=0.1, coreset=None):
        self.num_neurons = num_neurons
        self.cap_size=cap_size
        self.coreset_p = coreset_p
        self.background_p = background_p
        if coreset is None:
            self.coreset = np.arange(self.cap_size)
        else: 
            self.coreset = coreset
        self.non_coreset = np.array([i for i in range(self.num_neurons) if i not in self.coreset])
        self.firing_rates = np.ones(self.num_neurons) * self.background_p
        self.firing_rates[self.coreset] = self.coreset_p

    def sample_stimulus(self):
        x = np.zeros(self.num_neurons)
        sample = np.random.rand(self.num_neurons)
        x[sample<self.firing_rates] = 1.0

        return x
        
    def sample_stimuli(self, num_samples):
        I = np.random.rand(self.num_neurons, num_samples)
        X = (I< self.firing_rates[:,np.newaxis]).astype(np.float32)
        return X
        

    def sample_stimulus_fixed(self):
        stim_idx = np.concatenate((np.random.choice(self.coreset, np.floor(self.coreset_p*self.cap_size).astype(np.int32), replace=False), 
                                np.random.choice(self.non_coreset, self.cap_size - np.floor(self.coreset_p*self.cap_size).astype(np.int32), replace=False)))
        x = np.zeros(self.num_neurons)
        x[stim_idx] = 1.0
        
        return x

    def sample_stimuli_fixed(self, num_samples):
        X = np.array([self.sample_stimulus() for _ in range(num_samples)]).T
        return X
import numpy as np
import random
import copy
from scipy.sparse import csr_matrix
import scipy.sparse as sp


class Stimuli:
    '''
    A distribution of stimuli classes defined by coreset
    '''

    def __init__(self, num_neurons=1000, nclasses=2, nsamples=50, m=None, r=0.9, q=0.01, k=100, sparse=False):
        '''
        m: size of coreset, default m=k
        r: probability of firing within coreset
        q: probability of firing outside coreset
        '''
        self.sparse = sparse
        self.num_neurons_in_core = k if m == None else m
        self.nclasses = nclasses
        self.nsamples = nsamples
        self.num_neurons = num_neurons  # n

        self.q = q
        self.r = r
        self.k = k

        self.distributions = []

        for iclass in range(self.nclasses):
            class_dist = np.full(self.num_neurons, self.q)
            class_dist[random.sample(
                range(self.num_neurons), self.num_neurons_in_core)] = self.r
            self.distributions.append(class_dist)

    def generate_stimuli_set(self, nsamples=None):
        if nsamples == None:
            nsamples = self.nsamples
        if self.sparse:
            stimuli_set = sp.lil_matrix(
                (self.nclasses, nsamples, self.num_neurons), dtype=float)
            for iclass in range(len(self.distributions)):
                for isample in range(nsamples):
                    stimuli_set[iclass, isample, :] = sp.random(
                        1, self.num_neurons, self.distributions[iclass], format='csr')

        else:
            stimuli_set = np.zeros(
                (self.nclasses, nsamples, self.num_neurons), dtype=float)

            for iclass in range(len(self.distributions)):
                for isample in range(nsamples):
                    stimuli_set[iclass, isample, :] = np.random.binomial(
                        1, self.distributions[iclass])

        return stimuli_set

    def generate_stimuli_set_recurrence(self, nsamples=None, nrecurrent_rounds=None):
        if nsamples == None:
            nsamples = self.nsamples
        if nrecurrent_rounds == None:
            nrecurrent_rounds = 1

        if self.sparse:
            stimuli_set = sp.lil_matrix(
                (self.nclasses, nsamples, nrecurrent_rounds, self.num_neurons), dtype=float)

            for iclass in range(len(self.distributions)):
                for isample in range(nsamples):
                    for iround in range(nrecurrent_rounds):
                        stimuli_set[iclass, isample, iround, :] = sp.random(
                            1, self.num_neurons, self.distributions[iclass], format='csr')
        else:
            stimuli_set = np.zeros(
                (self.nclasses, nsamples, nrecurrent_rounds, self.num_neurons), dtype=float)

            for iclass in range(len(self.distributions)):
                for isample in range(nsamples):
                    for iround in range(nrecurrent_rounds):
                        stimuli_set[iclass, isample, iround, :] = np.random.binomial(
                            1, self.distributions[iclass])

        return stimuli_set

    def get_distributions(self):
        return self.distributions

    def convert_distributions_to_binary(self):
        '''
        Given a distribution array containing q and r, convert q to 0, and r to 1
        '''
        distributions = copy.deepcopy(self.distributions)
        for dist in distributions:
            # print(dist.shape)
            dist[dist == self.r] = 1
            dist[dist == self.q] = 0
        return distributions


if __name__ == "__main__":
    x = Stimuli(nclasses=5, num_neurons=5, k=2)
    print(x.get_distributions())
    print(x.convert_distributions_to_binary())

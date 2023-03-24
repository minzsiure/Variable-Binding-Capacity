from tkinter import N
import numpy as np
from scipy.sparse import csr_matrix


class Area:
    '''
    An Area contains a group of neurons.
    They may have pairwise recurrent connections.
    '''

    def __init__(self, n=1000, p=None, k=None, sparse=True):
        self.sparse = sparse

        # init n, number of neurons
        self.n = n

        # init k, number of winners
        if k != None:
            self.k = k
        else:
            self.k = int(np.sqrt(self.n))

        # init p, sparcity
        if p != None:
            self.p = p
        else:
            self.p = 1/np.sqrt(n)

        # neurons are inhibited to do recurrent/feedforward
        # activation unless disinhibited
        self.inhibited = True

        # init recurrent connections
        self.recurrent_connections = self.sample_initial_connections()

        # init y, the assembly
        if self.sparse:
            self.activations = csr_matrix(np.zeros(self.n, dtype=float))
        else:
            self.activations = np.zeros(self.n, dtype=float)

    def winners(self):
        """
        when access,
        return index of winners in self.activations
        """
        if self.sparse:
            return self.activations.nonzero()[1]
        else:
            return np.nonzero(self.activations)[0]

    def disinhibit(self):
        self.inhibited = False

    def inhibit(self):
        self.inhibited = True

    def sample_initial_connections(self):
        '''
        draw recurrenct connections based on sparcity value p
        '''
        connections = np.random.binomial(
            1, self.p, size=(self.n, self.n)).astype("float64")
        # no self loop
        np.fill_diagonal(connections, 0)
        if self.sparse:
            return csr_matrix(connections)
        return connections

    def wipe_activations(self):
        '''
        Reset all activation values y to be 0s
        '''
        if self.sparse:
            self.activations = csr_matrix(np.zeros(self.n))
        else:
            self.activation = np.zeros(self.n)

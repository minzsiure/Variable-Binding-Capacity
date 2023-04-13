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

    def perturb_r_and_get_new_dist(self, new_r):
        """
        Change firing probability r of old_dist to new_r, 
        then return this new distribution.
        """
        new_dist = [arr.copy() for arr in self.distributions]
        # iterate over the new list and replace self.r with new_r
        for i, arr in enumerate(new_dist):
            new_dist[i] = np.where(arr == self.r, new_r, arr)
        return new_dist

    def perturb_r_and_generate_stimuli_for_new_dist(self, new_dist=None, nsamples=None):
        """
        Generate nsamples of given distribution.
        """
        if nsamples == None:
            nsamples = self.nsamples

        stimuli_set = np.zeros(
            (self.nclasses, nsamples, self.num_neurons), dtype=float)
        for iclass in range(len(new_dist)):
            for isample in range(nsamples):
                stimuli_set[iclass, isample, :] = np.random.binomial(
                    1, new_dist[iclass])

        return stimuli_set

    def get_coreset_index(self):
        """
        Get the index of coreset neurons,
        return as a list of sub-lists, where each sub-list contains coreset indices for a .
        """
        dist_coreset_indices = [[i for i, val in enumerate(
            dist) if val == self.r] for dist in self.distributions]
        return dist_coreset_indices

    def perturbate_coreset(self, ham_dis, coreset):
        """
        Given a coreset, perturb it by ham_dis.
        """
        perturbed_coreset = copy.deepcopy(coreset)
        if ham_dis == 0:
            return perturbed_coreset
        replace = list(np.random.permutation(len(coreset))[
                       :ham_dis])  # elements for replacement
        flag = len(replace)
        # while there are elements left to replace, keep drawing from range(self.n),
        # replace only if they were not a part of the original coreset
        while flag != 0:
            lucky_number = random.randrange(self.num_neurons)
            if lucky_number not in coreset:
                replace_index = replace.pop()
                perturbed_coreset[replace_index] = lucky_number
                flag -= 1
        return perturbed_coreset

    def get_perturbated_coreset_indices(self, ham_dis):
        """
        Given the coreset indices of current distribution,
        perturb each of them by ham_dis.
        """
        dist_coreset_indices = self.get_coreset_index()
        perturb_dist_coreset_indices = []
        for core in dist_coreset_indices:
            perturb_dist_coreset_indices.append(
                self.perturbate_coreset(ham_dis, core))
        return perturb_dist_coreset_indices

    def generate_dist_based_on_coreset_indices(self, coreset_indices):
        """
        Backward version of get_coreset_index.
        Given a list of coreset indices,
        convert it back to a distribution.
        """
        distributions = []
        for indices in coreset_indices:
            dist = [self.q] * self.num_neurons
            for i in indices:
                dist[i] = self.r
            distributions.append(np.array(dist))
        return distributions

    def get_perturbated_coreset_distribution(self, ham_dis):
        """
        Given ham_dis, perturb my distributions by ham_dis.
        """
        perturb_dist_coreset_indices = self.get_perturbated_coreset_indices(
            ham_dis)
        perturb_dist = self.generate_dist_based_on_coreset_indices(
            perturb_dist_coreset_indices)
        return perturb_dist


if __name__ == "__main__":
    x = Stimuli(nclasses=2, num_neurons=10, k=2)
    print(x.get_distributions())
    print(x.perturb_r_and_generate_stimuli_for_new_dist(
        x.get_perturbated_coreset_distribution(2)))

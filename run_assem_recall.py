import numpy as np
from brain import *
from utils import *
from stimuli import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
import random
import copy
import argparse
import seaborn as sns
import os
from scipy.spatial.distance import hamming


def assembly_recall_by_perturb_X(new_info, num_neurons=1000, nrounds=5, beta=0.1,
                                 nclasses=6, m=None,
                                 k=100, connection_p=0.1, r=0.9, q=0.01, nsamples=50,
                                 with_normalization=True, wipe_y=True,
                                 nrecurrent_rounds=5, with_recurrence=True, X='r'):
    """
    1. Learn parameters by projecting nrounds of stimuli per class, for all nclasses.
    2. Form assemblies Y_0 using learned parameter.
    3. Perturb stimuli distribution's by new_info, form assemblies Y using new dist.
    4. Measure dH(Y_0, Y) for each class by taking median hamming distance within a class.

    X can be 'r' or 'coreset'
    """
    brain = Brain(num_areas=2, n=num_neurons, beta=beta,
                  p=connection_p, k=k)
    if with_normalization:
        brain.normalize_connections()

    stimuli_coreset = Stimuli(
        num_neurons=num_neurons, nclasses=nclasses, nsamples=nsamples, m=m, r=r, q=q, k=k)

    distributions = stimuli_coreset.get_distributions()
    if X == 'r':
        perturb_r_distributions = stimuli_coreset.perturb_r_and_get_new_dist(
            new_info)
    elif X == 'coreset':
        perturb_r_distributions = stimuli_coreset.get_perturbated_coreset_distribution(
            new_info)

    # 1. Learning each class
    for iclass in range(nclasses):
        # print('\tcurrent class', iclass)
        if wipe_y:
            brain.wipe_all_activations()

        # generate class distributions
        class_dist = distributions[iclass]

        for iiter in range(nrounds):  # do hebbian learning using nrounds samples
            x = np.random.binomial(1, class_dist)
            _, _, y = brain.project_with_change_in_recurrence(x, from_area_index=0, to_area_index=1,
                                                              max_iterations=1, verbose=0, return_stable_rank=False,
                                                              return_weights_assembly=True,
                                                              only_once=True, with_recurrence=with_recurrence)  # keep activations un-wiped

        if with_normalization:
            brain.normalize_connections()  # normalize after each class

    # 2, 3. Form assemblies Y_0, Y using learned parameter
    # find feedforward connection with info provided inpaired_areas
    x_feedforward = find_feedforward_matrix_index(
        brain.area_combinations, 0, 1)

    original_inputs = stimuli_coreset.generate_stimuli_set()
    original_outputs = np.zeros(
        (nclasses, nsamples, num_neurons))
    perturb_inputs = stimuli_coreset.perturb_r_and_generate_stimuli_for_new_dist(
        new_dist=perturb_r_distributions)
    perturb_outputs = np.zeros(
        (nclasses, nsamples, num_neurons))

    for j in range(nclasses):
        for isample in range(nsamples):
            temp_area2_activations_train = original_outputs[j, isample]
            temp_area2_activations_test = perturb_outputs[j, isample]

            # present multiple rounds to utilize recurrent connections, as defined by `nrecurrent_rounds`
            for iround in range(nrecurrent_rounds):
                if with_recurrence:
                    temp_area2_activations_train = capk(brain.feedforward_connections[x_feedforward].dot(original_inputs[j, isample])
                                                        + brain.areas[1].recurrent_connections.dot(temp_area2_activations_train),
                                                        k)
                    temp_area2_activations_test = capk(brain.feedforward_connections[x_feedforward].dot(perturb_inputs[j, isample])
                                                       + brain.areas[1].recurrent_connections.dot(temp_area2_activations_test),
                                                       k)
                else:
                    temp_area2_activations_train = capk(brain.feedforward_connections[x_feedforward].dot(original_inputs[j, isample]),
                                                        k)
                    temp_area2_activations_test = capk(brain.feedforward_connections[x_feedforward].dot(perturb_inputs[j, isample]),
                                                       k)
            original_outputs[j, isample] = temp_area2_activations_train
            perturb_outputs[j, isample] = temp_area2_activations_test

    # stack over all assemblies for computing hamming distance
    original_outputs_stack = np.vstack(tuple([np.vstack(tuple(
        [original_outputs[j, i] for i in range(nsamples)])) for j in range(nclasses)]))
    perturb_outputs_stack = np.vstack(tuple([np.vstack(tuple(
        [perturb_outputs[j, i] for i in range(nsamples)])) for j in range(nclasses)]))

    # 4. Measure hamming distance
    # Compute the Hamming distance between corresponding rows
    def hamming_distance(x, y):
        # return np.sum(x != y)
        return np.sum(x != y) / len(x) * 100

    # Initialize array to store median hamming distance for each class
    median_hamming_dist = np.zeros((nclasses,))

    # Iterate over each class
    for j in range(nclasses):
        # Get rows corresponding to class j
        original_outputs_j = original_outputs_stack[j *
                                                    nsamples:(j+1)*nsamples, :]
        perturb_outputs_j = perturb_outputs_stack[j*nsamples:(j+1)*nsamples, :]

        # Compute hamming distance between corresponding rows
        hamming_dist = np.array([hamming_distance(original_outputs_j[i, :], perturb_outputs_j[i, :])
                                for i in range(nsamples)])

        # Compute median hamming distance within class j
        median_hamming_dist[j] = np.median(hamming_dist)
    return median_hamming_dist  # nclasses by 1 vector


def experiment_on_assembly_recall_by_perturb_r(num_neurons=1000, nrounds=5, beta=0.1,
                                               nclasses=6, m=None,
                                               k=100, connection_p=0.1, r=0.9, q=0.01, nsamples=50,
                                               with_normalization=True, wipe_y=True,
                                               nrecurrent_rounds=5, ntrials=5, X='r'):
    """
    Experiment run by calling list of perturb_r values.
    """
    if X == 'r':
        rs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    elif X == 'coreset':
        rs = np.linspace(0, 2*k, 9).astype(int)
    results_with_recurr, results_without_recurr = np.zeros(
        (nclasses, ntrials, len(rs))), np.zeros((nclasses, ntrials, len(rs)))

    for recurr in [True, False]:
        print(recurr)
        for i, new_r in enumerate(rs):
            print(new_r)
            result_for_this_r = np.zeros((nclasses, ntrials))
            for itrial in range(ntrials):
                median_hamming_dist = assembly_recall_by_perturb_X(new_r, num_neurons=num_neurons, nrounds=nrounds, beta=beta,
                                                                   nclasses=nclasses, m=m,
                                                                   k=k, connection_p=connection_p, r=r, q=q, nsamples=nsamples,
                                                                   with_normalization=with_normalization, wipe_y=wipe_y,
                                                                   nrecurrent_rounds=nrecurrent_rounds, with_recurrence=recurr, X=X)
                result_for_this_r[:, itrial] = median_hamming_dist
            if recurr:
                results_with_recurr[:, :, i] = result_for_this_r
            else:
                results_without_recurr[:, :, i] = result_for_this_r

    median_with_recurr = np.round(np.median(results_with_recurr, axis=1))
    sem_with_recurr = np.std(results_with_recurr, axis=1)/np.sqrt(ntrials)

    median_without_recurr = np.round(np.median(results_without_recurr, axis=1))
    sem_without_recurr = np.std(
        results_without_recurr, axis=1)/np.sqrt(ntrials)

    colors = cm.get_cmap('Set1', 9)
    # plot each class
    for recurr in [True, False]:
        for iclass in range(median_with_recurr.shape[0]):
            if recurr:
                label = "Class %i with Recurrence" % (iclass)
                y = median_with_recurr[iclass, :]
                y_sem = sem_with_recurr[iclass, :]
                line_shape = '-'
            else:
                label = "Class %i without Recurrence" % (iclass)
                y = median_without_recurr[iclass, :]
                y_sem = sem_without_recurr[iclass, :]
                line_shape = '--'

            plt.plot(
                rs, y, label=label, color=colors(iclass), linestyle=line_shape)
            plt.fill_between(rs, y - y_sem, y + y_sem,
                             alpha=0.25, color=colors(iclass), edgecolor='none')
    plt.legend()
    if X == 'r':
        plt.xlabel('Probability of coreset firing $(r)$ after perturbation')
    elif X == 'coreset':
        plt.xlabel('Hamming distance between coresets')
    plt.ylabel('$dH(Y_0, Y)$ in $\%$')
    plt.show()


if __name__ == "__main__":
    # print(assembly_recall_by_perturb_X(5, X='coreset'))
    experiment_on_assembly_recall_by_perturb_r(
        nclasses=2, k=100, nsamples=50, nrecurrent_rounds=5, nrounds=15, X='r', ntrials=5)

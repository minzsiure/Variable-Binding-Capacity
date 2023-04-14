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


def perturb(assembly, alpha, k):
    """
    Given an assembly, turn off alpha*k of its coreset.
    """
    perturbed_assem = np.copy(assembly)

    m = int(alpha * k)

    idx = np.where(assembly == 1)[0]
    idx_mod = np.random.choice(idx, m, replace=False)
    perturbed_assem[idx_mod] = 0
    return perturbed_assem


def measure_percentage_recovered(original_assem, recovered_assem):
    """
    Measure intersect(S_y0, S_y)/S_y0
    """
    idx_original = np.where(original_assem == 1)[0]
    idx_recovered = np.where(recovered_assem == 1)[0]

    intersect = set(idx_original) & set(idx_recovered)
    return len(intersect)/len(idx_original)


def pattern_completion(alpha, num_neurons=1000, nrounds=5, beta=0.1,
                       nclasses=6, m=None,
                       k=100, connection_p=0.1, r=0.9, q=0.01, nsamples=50,
                       with_normalization=True, wipe_y=True,
                       nrecurrent_rounds=5):
    """
    1. Learn parameters by projecting nrounds of stimuli per class, for all nclasses.
    2. Form assemblies Y_0 using learned parameter.
    3. Perturb Y_0 by turning off alpha*k of its 1s
    4. Recover by doing capk(W_cc*y) for 2 rounds
    5. Measure intersect(S_y0, S_y)/S_y0 where S means coreset
    """
    brain = Brain(num_areas=2, n=num_neurons, beta=beta,
                  p=connection_p, k=k)
    if with_normalization:
        brain.normalize_connections()

    stimuli_coreset = Stimuli(
        num_neurons=num_neurons, nclasses=nclasses, nsamples=nsamples, m=m, r=r, q=q, k=k)

    distributions = stimuli_coreset.get_distributions()

    # 1. Learning each class
    for iclass in range(nclasses):
        # print('\tcurrent class', iclass)
        if wipe_y:
            brain.wipe_all_activations()

        # generate class distributions
        class_dist = distributions[iclass]

        for iiter in range(nrounds):  # do hebbian learning using nrounds samples
            x = np.random.binomial(1, class_dist)
            _, _, y = brain.project(x, from_area_index=0, to_area_index=1,
                                    max_iterations=1, verbose=0, return_stable_rank=False,
                                    return_weights_assembly=True,
                                    only_once=True)  # keep activations un-wiped

        if with_normalization:
            brain.normalize_connections()  # normalize after each class

    x_feedforward = find_feedforward_matrix_index(
        brain.area_combinations, 0, 1)

    original_inputs = stimuli_coreset.generate_stimuli_set()
    original_outputs = np.zeros(
        (nclasses, nsamples, num_neurons))

    # 2. Form Y_0
    for j in range(nclasses):
        for isample in range(nsamples):
            temp_area2_activations_train = original_outputs[j, isample]

            # present multiple rounds to utilize recurrent connections, as defined by `nrecurrent_rounds`
            for iround in range(nrecurrent_rounds):
                temp_area2_activations_train = capk(brain.feedforward_connections[x_feedforward].dot(original_inputs[j, isample])
                                                    + brain.areas[1].recurrent_connections.dot(temp_area2_activations_train),
                                                    k)

            original_outputs[j, isample] = temp_area2_activations_train

    # stack over all assemblies for computing hamming distance
    original_outputs_stack = np.vstack(tuple([np.vstack(tuple(
        [original_outputs[j, i] for i in range(nsamples)])) for j in range(nclasses)]))

    # 3. Perturb Y_0 by turning off alpha*k of its 1s
    # 4. Recover by doing capk(W_cc*y) for 2 rounds
    recovered_outputs = np.zeros(
        (nclasses, nsamples, num_neurons))
    for j in range(nclasses):
        for isample in range(nsamples):
            current_assembly = original_outputs[j, isample]
            # perturb by turning off alpha*k of 1s
            perturbed_assembly = perturb(current_assembly, alpha, k)

            # recover for 2 rounds
            for iround in range(2):
                perturbed_assembly = capk(brain.areas[1].recurrent_connections.dot(perturbed_assembly),
                                          k)
            recovered_outputs[j, isample] = perturbed_assembly

    recovered_outputs_stack = np.vstack(tuple([np.vstack(tuple(
        [recovered_outputs[j, i] for i in range(nsamples)])) for j in range(nclasses)]))

    # Initialize array to store median hamming distance for each class
    median_recover_rate = np.zeros((nclasses,))

    # Iterate over each class
    for j in range(nclasses):
        # Get rows corresponding to class j
        original_outputs_j = original_outputs_stack[j *
                                                    nsamples:(j+1)*nsamples, :]
        recovered_outputs_j = recovered_outputs_stack[j *
                                                      nsamples:(j+1)*nsamples, :]

        # Compute hamming distance between corresponding rows
        recover_rate = np.array([measure_percentage_recovered(original_outputs_j[i, :], recovered_outputs_j[i, :])
                                for i in range(nsamples)])

        # Compute median hamming distance within class j
        median_recover_rate[j] = np.median(recover_rate)
    return median_recover_rate  # nclasses by 1 vector


def experiment_on_pattern_completion(alpha, num_neurons=1000, beta=0.1,
                                     nclasses=6, m=None,
                                     k=100, connection_p=0.1, r=0.9, q=0.01, nsamples=50,
                                     with_normalization=True, wipe_y=True,
                                     nrecurrent_rounds=5, ntrials=5):
    """
    Experiment run by calling list of perturb_r values.
    """
    nroundss = np.linspace(4, 14, 10).astype(int)
    results = np.zeros(
        (nclasses, ntrials, len(nroundss)))

    for i, nround in enumerate(nroundss):
        result_for_this_r = np.zeros((nclasses, ntrials))
        for itrial in range(ntrials):
            median_recover = pattern_completion(alpha, num_neurons=num_neurons, nrounds=nround, beta=beta,
                                                nclasses=nclasses, m=m,
                                                k=k, connection_p=connection_p, r=r, q=q, nsamples=nsamples,
                                                with_normalization=with_normalization, wipe_y=wipe_y,
                                                nrecurrent_rounds=nrecurrent_rounds)
            result_for_this_r[:, itrial] = median_recover
        results[:, :, i] = result_for_this_r

    median = np.round(np.median(results, axis=1))
    sem = np.std(results, axis=1)/np.sqrt(ntrials)

    colors = cm.get_cmap('Set1', 9)
    # plot each class
    for iclass in range(results.shape[0]):
        label = "Class %i" % (iclass)
        y = median[iclass, :]
        y_sem = sem[iclass, :]

        plt.plot(
            nroundss, y, label=label, color=colors(iclass))
        plt.fill_between(nroundss, y - y_sem, y + y_sem,
                         alpha=0.25, color=colors(iclass), edgecolor='none')
    plt.legend()
    plt.xlabel('Learning time per class')
    plt.ylabel('Percentage assembly recovered')
    plt.show()


if __name__ == "__main__":
    experiment_on_pattern_completion(
        0.5, nclasses=2)

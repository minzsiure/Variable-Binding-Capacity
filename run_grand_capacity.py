from importlib.metadata import distributions
import numpy as np
from brain import *
from utils import *
from stimuli import *
import matplotlib.pyplot as plt
import sys
import random
import copy
import argparse
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression

# setting global random seed
rng = np.random.default_rng(2022)

"""
This file contains main functions to measure capacity wrt n or k
under both projection and reciprocal projection.
"""


def multiround_test_capacity_using_projection_with_linear_classifier(num_neurons=1000, nrounds=5, beta=0.1,
                                                                     nclasses=6, m=None,
                                                                     k=100, connection_p=0.1, r=0.9, q=0.1, num_samples=50,
                                                                     with_normalization=True, wipe_y=True,
                                                                     nrecurrent_rounds=5, classifier=False, show_input_overlap=False):
    '''
    This function measures in- and out- class similaritiy of stimuli and assembly under projection.
    nrounds: number of examples from each class shown to the brain during hebbian traing.
            At the test time, connections are fixed.
    m: size of coreset, default m=k
    r: probability of firing within coreset
    q: probability of firing outside coreset
    plot_overlap: plot assembly confusion matrix, stimuli confusion matrix.
    num_samples: number of samples for testing per class.
    classifier: when set to true, we evaluate with a linear classifier as well.
    with_normalization: when set to true, we normalize weights after training per class.
    return 2 things:
        * average of within class similarity
        * average of outside class similarity
    '''
    # brain initialization
    num_neurons_per_class = k if m == None else m

    brain = Brain(num_areas=2, n=num_neurons, beta=beta,
                  p=connection_p, k=k)
    if with_normalization:
        brain.normalize_connections()

    # stimuli initialization: prepare input for later testing (NOT learning)
    n_samples = num_samples
    stimuli_coreset = Stimuli(
        num_neurons=num_neurons, nclasses=nclasses, nsamples=num_samples, m=m, r=r, q=q, k=k)

    # **Here is input for later testing**
    if classifier:
        inputs_to_train = stimuli_coreset.generate_stimuli_set()
    inputs_to_test = stimuli_coreset.generate_stimuli_set()

    distributions = stimuli_coreset.get_distributions()

    # learning in nrounds
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

    # we store the output here
    if classifier:
        outputs_from_train = np.zeros(
            (nclasses, n_samples, num_neurons))

    outputs_from_test = np.zeros(
        (nclasses, n_samples, num_neurons))

    # find feedforward connection with info provided inpaired_areas
    x_feedforward = find_feedforward_matrix_index(
        brain.area_combinations, 0, 1)

    for j in range(nclasses):
        for isample in range(n_samples):
            if classifier:
                temp_area2_activations_train = outputs_from_train[j, isample]

            temp_area2_activations_test = outputs_from_test[j, isample]

            # present multiple rounds to utilize recurrent connections, as defined by `nrecurrent_rounds`
            for iround in range(nrecurrent_rounds):
                if classifier:
                    temp_area2_activations_train = capk(brain.feedforward_connections[x_feedforward].dot(inputs_to_train[j, isample])
                                                        + brain.areas[1].recurrent_connections.dot(temp_area2_activations_train),
                                                        k)

                temp_area2_activations_test = capk(brain.feedforward_connections[x_feedforward].dot(inputs_to_test[j, isample])
                                                   + brain.areas[1].recurrent_connections.dot(temp_area2_activations_test),
                                                   k)
            if classifier:
                outputs_from_train[j, isample] = temp_area2_activations_train

            outputs_from_test[j, isample] = temp_area2_activations_test

    # stack over all assemblies
    if classifier:
        output_from_train_stack = np.vstack(tuple([np.vstack(tuple(
            [outputs_from_train[j, i] for i in range(n_samples)])) for j in range(nclasses)]))
        output_from_test_stack = np.vstack(tuple([np.vstack(tuple(
            [outputs_from_test[j, i] for i in range(n_samples)])) for j in range(nclasses)]))

    ########## compute overlap ############
    # compute input overlap
    if show_input_overlap:
        inp_overlap_mat = np.sum(np.mean(
            inputs_to_test[:, :-1][:, np.newaxis, :] * inputs_to_test[:, 1:][np.newaxis, :, :], axis=2), axis=-1)
        inp_overlap_mat = inp_overlap_mat / num_neurons_per_class * 100

    # compute assembly overlap
    assm_overlap_mat = np.sum(np.mean(
        outputs_from_test[:, :-1][:, np.newaxis, :] * outputs_from_test[:, 1:][np.newaxis, :, :], axis=2), axis=-1)
    assm_overlap_mat = assm_overlap_mat / k * 100
    ############################################

    ########## compute average ############
    # avg of inside class similarity
    avg_assm_overlap_within_class = np.trace(assm_overlap_mat)/nclasses
    # avg of outside class similarity
    avg_assm_overlap_outside_class = (np.sum(assm_overlap_mat) -
                                      np.trace(assm_overlap_mat))/(nclasses*(nclasses-1))

    if show_input_overlap:
        # avg of inside class similarity
        avg_sim_overlap_within_class = np.trace(inp_overlap_mat)/nclasses
        # avg of outside class similarity
        avg_sim_overlap_outside_class = (np.sum(inp_overlap_mat) -
                                         np.trace(inp_overlap_mat))/(nclasses*(nclasses-1))
        print('\taverage stimuli within class overlap %.2f, average stimuli OUTSIDE class overlap %.2f' % (
            avg_sim_overlap_within_class, avg_sim_overlap_outside_class))

    print('\taverage assembly within class overlap %.2f, average assembly OUTSIDE class overlap %.2f' % (
        avg_assm_overlap_within_class, avg_assm_overlap_outside_class))

    ##################classifier##########################
    if classifier:
        # generate labels
        y_label = generate_labels(nclasses, n_samples)
        # fit a classifier
        classifier = LogisticRegression(
            solver='liblinear', max_iter=80)

        classifier.fit(output_from_train_stack, y_label)
        fit_acc = classifier.score(output_from_train_stack, y_label)
        eval_acc = classifier.score(output_from_test_stack, y_label)
        print('\t%i classes. fit acc is %.4f, eval acc is %.4f, chance is %.4f' %
              (nclasses, fit_acc, eval_acc, 1/nclasses))

        # print('\t%i classes. Fit Accuracy of L1 on Y1 is %.4f (expecting 1)' %
        #       (nclasses, fit_acc))
        # print('\t%i classes. Evaluation Accuracy of L1 on Y1 prime is %.4f (expecting 1)' %
        #       (nclasses, eval_acc))

    return round(avg_assm_overlap_outside_class, 2), round(avg_assm_overlap_within_class, 2)


def compute_min_diagonal_and_max_offDiagonal(matrix):
    """
    Given a matrix, 
    compute the minimum of diagonal elements and maximum of off diagonal elements.
    """
    # get diagonal elements and compute the minimum
    diagonal = np.diag(matrix)
    min_diagonal = np.min(diagonal)

    # get off-diagonal elements and compute the maximum
    off_diagonal = matrix[~np.eye(matrix.shape[0], dtype=bool)]
    max_off_diagonal = np.max(off_diagonal)

    return round(min_diagonal, 2), round(max_off_diagonal, 2)


def multiround_test_capacity_using_reciprocal_projection_with_linear_classifier(num_neurons=1000, nrounds=5, beta=0.1,
                                                                                nclasses=6, m=None,
                                                                                k=50, connection_p=0.1, r=0.9, q=0.01, num_samples=50,
                                                                                with_normalization=True, wipe_y=True,
                                                                                nrecurrent_rounds=5, residual_reci_project=False, classifier=False,
                                                                                show_input_overlap=False, use_average=True):
    '''
    This function measures in- and out- class similaritiy of stimuli and assembly under reciprocal-projection.
    nrounds: number of examples from each class shown to the brain during hebbian traing.
            At the test time, connections are fixed.
    m: size of coreset, default m=k
    r: probability of firing within coreset
    q: probability of firing outside coreset
    plot_overlap: plot assembly confusion matrix, stimuli confusion matrix.
    num_samples: number of samples for testing per class.
    classifier: when set to true, we evaluate with a linear classifier as well.
    with_normalization: when set to true, we normalize weights after training per class.
    return 4 things:
        * average of within class similarity (area 1, 2)
        * average of outside class similarity (area 1, 2)
    '''
    # brain initialization
    num_neurons_per_class = k if m == None else m

    brain = Brain(num_areas=3, n=num_neurons, beta=beta,
                  p=connection_p, k=k)
    if with_normalization:
        brain.normalize_connections()

    # stimuli initialization: prepare input for later testing (NOT learning)
    n_samples = num_samples
    stimuli_coreset = Stimuli(
        num_neurons=num_neurons, nclasses=nclasses, nsamples=num_samples, m=m, r=r, q=q, k=k)

    # **Here is input for later testing**
    if classifier:
        inputs_to_train = stimuli_coreset.generate_stimuli_set()

    inputs_to_test = stimuli_coreset.generate_stimuli_set()

    distributions = stimuli_coreset.get_distributions()

    # learning in nrounds
    for iclass in range(nclasses):
        # print('\tcurrent class', iclass)
        if wipe_y:
            brain.wipe_all_activations()

        # generate class distributions
        class_dist = distributions[iclass]

        for iiter in range(nrounds):  # do hebbian learning using nrounds samples
            x = np.random.binomial(1, class_dist)
            if residual_reci_project:
                _, _, _, _, _, _, y2, y3 = brain.residual_reciprocal_project(x, area1_index=0, area2_index=1, area3_index=2,
                                                                             max_iterations=1, verbose=0,
                                                                             return_weights_assembly=True,
                                                                             only_once=True)  # keep activations un-wiped
            else:
                _, _, _, _, _, y2, y3 = brain.reciprocal_project(x, area1_index=0, area2_index=1, area3_index=2,
                                                                 max_iterations=1, verbose=0,
                                                                 return_weights_assembly=True,
                                                                 only_once=True)  # keep activations un-wiped

        if with_normalization:
            brain.normalize_connections()  # normalize after each class

    # for fitting classifier
    if classifier:
        outputs_y_train = np.zeros((nclasses, n_samples, num_neurons))
        outputs_z_train = np.zeros((nclasses, n_samples, num_neurons))

    # for evaluating classifier
    outputs_y_test = np.zeros((nclasses, n_samples, num_neurons))
    outputs_z_test = np.zeros((nclasses, n_samples, num_neurons))

    # find feedforward connection with info provided inpaired_areas
    iweights_1to2 = find_feedforward_matrix_index(
        brain.area_combinations, 0, 1)  # feedforward
    iweights_2to3 = find_feedforward_matrix_index(
        brain.area_combinations, 1, 2)  # feedforward
    iweights_3to2 = find_feedforward_matrix_index(
        brain.area_combinations, 2, 1)  # feedbackward

    if residual_reci_project:
        iweights_1to3 = find_feedforward_matrix_index(
            brain.area_combinations, 0, 2)  # feedbackward

    if classifier:
        for j in range(nclasses):
            for isample in range(n_samples):
                temp_area2_activations_train = outputs_y_train[j, isample]
                temp_area3_activations_train = outputs_z_train[j, isample]

                for iround in range(nrecurrent_rounds):
                    x = inputs_to_train[j, isample]
                    new_area2_activations = capk(brain.feedforward_connections[iweights_1to2].dot(x)
                                                 + brain.areas[1].recurrent_connections.dot(temp_area2_activations_train)
                                                 + brain.feedforward_connections[iweights_3to2].dot(temp_area3_activations_train),
                                                 k)
                    if residual_reci_project:
                        new_area3_activations = capk(brain.feedforward_connections[iweights_2to3].dot(temp_area2_activations_train)
                                                     + brain.areas[2].recurrent_connections.dot(temp_area3_activations_train)
                                                     + brain.feedforward_connections[iweights_1to3].dot(x),
                                                     k)
                    else:
                        new_area3_activations = capk(brain.feedforward_connections[iweights_2to3].dot(temp_area2_activations_train)
                                                     + brain.areas[2].recurrent_connections.dot(temp_area3_activations_train),
                                                     k)
                    temp_area2_activations_train = new_area2_activations
                    temp_area3_activations_train = new_area3_activations

                # only record the last round of activations, after utilizing recurrent connections
                outputs_y_train[j, isample] = temp_area2_activations_train
                outputs_z_train[j, isample] = temp_area3_activations_train

    # testing time
    for j in range(nclasses):
        for isample in range(n_samples):
            temp_area2_activations_test = outputs_y_test[j, isample]
            temp_area3_activations_test = outputs_z_test[j, isample]

            for iround in range(nrecurrent_rounds):
                x = inputs_to_test[j, isample]
                new_area2_activations = capk(brain.feedforward_connections[iweights_1to2].dot(x)
                                             + brain.areas[1].recurrent_connections.dot(temp_area2_activations_test)
                                             + brain.feedforward_connections[iweights_3to2].dot(temp_area3_activations_test),
                                             k)
                if residual_reci_project:
                    new_area3_activations = capk(brain.feedforward_connections[iweights_2to3].dot(temp_area2_activations_test)
                                                 + brain.areas[2].recurrent_connections.dot(temp_area3_activations_test)
                                                 + brain.feedforward_connections[iweights_1to3].dot(x),
                                                 k)
                else:
                    new_area3_activations = capk(brain.feedforward_connections[iweights_2to3].dot(temp_area2_activations_test)
                                                 + brain.areas[2].recurrent_connections.dot(temp_area3_activations_test),
                                                 k)
                temp_area2_activations_test = new_area2_activations
                temp_area3_activations_test = new_area3_activations

            # only record the last round of activations, after utilizing recurrent connections
            outputs_y_test[j, isample] = temp_area2_activations_test
            outputs_z_test[j, isample] = temp_area3_activations_test

    # stack
    if classifier:
        Y = np.vstack(tuple([np.vstack(tuple(
            [outputs_y_train[j, i] for i in range(n_samples)])) for j in range(nclasses)]))
        Z = np.vstack(tuple([np.vstack(tuple(
            [outputs_z_train[j, i] for i in range(n_samples)])) for j in range(nclasses)]))
        Y_prime = np.vstack(tuple([np.vstack(tuple(
            [outputs_y_test[j, i] for i in range(n_samples)])) for j in range(nclasses)]))
        Z_prime = np.vstack(tuple([np.vstack(tuple(
            [outputs_z_test[j, i] for i in range(n_samples)])) for j in range(nclasses)]))

    ########## compute overlap ############
    # compute input overlap
    if show_input_overlap:
        inp_overlap_mat = np.sum(np.mean(
            inputs_to_test[:, :-1][:, np.newaxis, :] * inputs_to_test[:, 1:][np.newaxis, :, :], axis=2), axis=-1)
        inp_overlap_mat = inp_overlap_mat / num_neurons_per_class * 100

    # compute assembly overlap `assm_overlap_mat_y`, `assm_overlap_mat_z`
    assm_overlap_mat_y = np.sum(np.mean(
        outputs_y_test[:, :-1][:, np.newaxis, :] * outputs_y_test[:, 1:][np.newaxis, :, :], axis=2), axis=-1)
    assm_overlap_mat_y = assm_overlap_mat_y / k * 100

    assm_overlap_mat_z = np.sum(np.mean(
        outputs_z_test[:, :-1][:, np.newaxis, :] * outputs_z_test[:, 1:][np.newaxis, :, :], axis=2), axis=-1)
    assm_overlap_mat_z = assm_overlap_mat_z / k * 100
    ############################################

    ########## compute average ############
    if show_input_overlap:
        # avg of inside class similarity
        avg_sim_overlap_within_class = np.trace(inp_overlap_mat)/nclasses
        # avg of outside class similarity
        avg_sim_overlap_outside_class = (np.sum(inp_overlap_mat) -
                                         np.trace(inp_overlap_mat))/(nclasses*(nclasses-1))
        print('\taverage stimuli within class overlap %.2f, average stimuli OUTSIDE class overlap %.2f' % (
            avg_sim_overlap_within_class, avg_sim_overlap_outside_class))

    if use_average:
        # avg of inside class similarity
        avg_assm_overlap_within_class_Y = np.trace(assm_overlap_mat_y)/nclasses
        # avg of outside class similarity
        avg_assm_overlap_outside_class_Y = (np.sum(assm_overlap_mat_y) -
                                            np.trace(assm_overlap_mat_y))/(nclasses*(nclasses-1))

        # avg of inside class similarity
        avg_assm_overlap_within_class_Z = np.trace(assm_overlap_mat_z)/nclasses
        # avg of outside class similarity
        avg_assm_overlap_outside_class_Z = (np.sum(assm_overlap_mat_z) -
                                            np.trace(assm_overlap_mat_z))/(nclasses*(nclasses-1))
        print('\tY. average assembly within class overlap %.2f, average assembly OUTSIDE class overlap %.2f' % (
            avg_assm_overlap_within_class_Y, avg_assm_overlap_outside_class_Y))
        print('\tZ. average assembly within class overlap %.2f, average assembly OUTSIDE class overlap %.2f' % (
            avg_assm_overlap_within_class_Z, avg_assm_overlap_outside_class_Z))
    else:
        ########## compute min of diagonal / max of off-diagonal ############
        min_diagonal_y, max_off_diagonal_y = compute_min_diagonal_and_max_offDiagonal(
            assm_overlap_mat_y)
        min_diagonal_z, max_off_diagonal_z = compute_min_diagonal_and_max_offDiagonal(
            assm_overlap_mat_z)
        print('\tY. minimum diagonal entry %.2f, maximum off-diagonal entry %.2f' % (
            min_diagonal_y, max_off_diagonal_y))
        print('\tZ. minimum diagonal entry %.2f, maximum off-diagonal entry %.2f' % (
            min_diagonal_z, max_off_diagonal_z))

    ##################classifier##########################
    if classifier:
        # generate labels
        y_label = generate_labels(nclasses, n_samples)

        # fit classifier
        L = LogisticRegression(solver='liblinear', max_iter=80)
        L.fit(Y, y_label)
        Lp = LogisticRegression(solver='liblinear', max_iter=80)
        Lp.fit(Z, y_label)

        L_tr_acc = L.score(Y, y_label)  # fit on Y, see fit accuracy on Y
        # fit on Y, see eval accuracy on Y_prime
        L_te_acc = L.score(Y_prime, y_label)

        Lp_tr_acc = Lp.score(Z, y_label)  # fit on Z, see fit accuracy on Z
        # fit on Z, see eval accuracy on Z_prime
        Lp_te_acc = Lp.score(Z_prime, y_label)

        print('\tY. %i classes. Fit Accuracy is %.4f. Eval Accuracy is %.4f. Chance is %.4f.' %
              (nclasses, L_tr_acc, L_te_acc, 1/nclasses))
        print('\tZ. %i classes. Fit Accuracy is %.4f. Eval Accuracy is %.4f. Chance is %.4f.' %
              (nclasses, Lp_tr_acc, Lp_te_acc, 1/nclasses))
        print()

    if use_average:
        return round(avg_assm_overlap_outside_class_Y, 2), round(avg_assm_overlap_within_class_Y, 2), round(avg_assm_overlap_outside_class_Z, 2), round(avg_assm_overlap_within_class_Z, 2)
    else:
        return max_off_diagonal_y, min_diagonal_y, max_off_diagonal_z, min_diagonal_z


def test_capacity_in_projection_as_a_function_of_brain_size_with_linear_classifier(nrange, nrounds=5, beta=0.1, m=None,
                                                                                   k=50, connection_p=0.1, r=0.9, q=0.01,
                                                                                   num_samples=50, with_normalization=True, wipe_y=True,
                                                                                   nrecurrent_rounds=5, num_trials=5,
                                                                                   plot_all=True, plot_name='plot1.pdf', show_input_overlap=False,
                                                                                   classifier=False, standard='average'):
    '''
    We test for capacity in assembly calculus with respect to 
    confusion between average in- and out- class similarity
    as a function of brain size.
    When the average of out-class similarity exceeds or equal to in-class similarity,
    the assembly model hits its capacity (catastrophic forgetting). 
    '''
    result = []
    global_starting_classes = 2

    concat = np.zeros(
        (nrange[1]//100, num_trials))
    count = 0
    # assume linearity, each class will start with previous
    for n in range(nrange[0], nrange[1]+100, 100):
        print('******* searching with %i neurons **********' % (n))
        avg_capacity = []
        for itrial in range(num_trials):
            try_class = global_starting_classes
            print('==================================================')
            while True:
                print('num_neurons='+str(n)+', k='+str(k)+', m=' +
                      str(k)+', nclasses='+str(try_class))
                print('current trial', itrial)
                avg_assm_overlap_outside_class, avg_assm_overlap_within_class = multiround_test_capacity_using_projection_with_linear_classifier(num_neurons=n, nrounds=nrounds, beta=beta,
                                                                                                                                                 nclasses=try_class, m=m,
                                                                                                                                                 k=k, connection_p=connection_p, r=r, q=q,
                                                                                                                                                 num_samples=num_samples,
                                                                                                                                                 with_normalization=with_normalization, wipe_y=wipe_y,
                                                                                                                                                 nrecurrent_rounds=nrecurrent_rounds,
                                                                                                                                                 show_input_overlap=show_input_overlap,
                                                                                                                                                 classifier=classifier, standard=standard)
                if avg_assm_overlap_within_class <= avg_assm_overlap_outside_class:
                    avg_capacity.append(try_class)
                    print()
                    break
                else:
                    print()
                    try_class += 1
            print('==================================================')
        result.append(round(np.median(avg_capacity)))
        concat[count] = avg_capacity

        count += 1
        global_starting_classes = round(
            np.median(avg_capacity))

    if plot_all:
        Y_mean = np.round(np.median(concat, axis=1))
        Y_sem = np.std(concat, axis=1)/np.sqrt(num_trials)

        x = [i for i in range(nrange[0], nrange[1]+100, 100)]

        m1, b1 = np.polyfit(x, Y_mean, 1)

        if round(b1, 4) < 0:
            plt.plot(
                x, Y_mean, label=f"y = {round(m1,4)} * x {round(b1,4)}")
        else:
            plt.plot(
                x, Y_mean, label=f"y = {round(m1,4)} * x + {round(b1,4)}")
        plt.fill_between(x, Y_mean - Y_sem, Y_mean +
                         Y_sem, alpha=0.5)
        plt.legend()

        res_Y_mean = {key: value for key, value in zip(x, Y_mean)}
        res_Y_sem = {key: value for key, value in zip(x, Y_sem)}
        print(concat)

        plt.title(
            'Projection. \nk=%i, beta=%.1f, p=%.1f, nrecurrent=%i, q=%.2f, ntrials=%i' % (k, beta, connection_p, nrecurrent_rounds, q, num_trials))
        plt.xlabel('Number of neurons')
        plt.ylabel('Capacity (number of classes)')

        # save figure
        output_directory = 'figures/project_capacity/capacity_wrt_n/'
        output_filepath = output_directory + plot_name
        plt.savefig(output_filepath, format='pdf')

        plt.show()

    return res_Y_mean, res_Y_sem


def test_capacity_in_projection_as_a_function_of_capk_size_with_linear_classifier(krange, num_neurons=250, nrounds=5, beta=0.1, m=50,
                                                                                  connection_p=0.1, r=0.9, q=0.01,
                                                                                  num_samples=50, with_normalization=True, wipe_y=True,
                                                                                  nrecurrent_rounds=5, num_trials=5,
                                                                                  plot_all=True, plot_name='plot1.pdf'):
    '''
    We test for capacity in assembly calculus with respect to 
    confusion between average in- and out- class similarity
    as a function of capK size.
    When the average of out-class similarity exceeds or equal to in-class similarity,
    the assembly model hits its capacity (catastrophic forgetting). 
    '''
    result = []
    global_starting_classes = 2

    concat = np.zeros(
        ((krange[1]-krange[0])//20 + 1, num_trials))
    count = 0
    # assume linearity, each class will start with previous
    for k in range(krange[0], krange[1]+1, 20):
        print('******* searching with %i for capk **********' % (k))
        avg_capacity = []

        for itrial in range(num_trials):
            try_class = global_starting_classes
            print('==================================================')
            while True:
                print('num_neurons='+str(num_neurons)+', k='+str(k)+', m=' +
                      str(m)+', nclasses='+str(try_class))
                print('current trial', itrial)
                avg_assm_overlap_outside_class, avg_assm_overlap_within_class = multiround_test_capacity_using_projection_with_linear_classifier(num_neurons=num_neurons, nrounds=nrounds, beta=beta,
                                                                                                                                                 nclasses=try_class, m=m,
                                                                                                                                                 k=k, connection_p=connection_p, r=r, q=q,
                                                                                                                                                 num_samples=num_samples,
                                                                                                                                                 with_normalization=with_normalization, wipe_y=wipe_y,
                                                                                                                                                 nrecurrent_rounds=nrecurrent_rounds)
                if avg_assm_overlap_within_class <= avg_assm_overlap_outside_class:
                    avg_capacity.append(try_class)
                    print()
                    break
                else:
                    try_class += 1
                    print()
            print('==================================================')
        result.append(round(np.median(avg_capacity)))
        concat[count] = avg_capacity
        count += 1

    if plot_all:
        Y_mean = np.round(np.median(concat, axis=1))
        # print('compare should be equal', Y_mean, result_Y)
        Y_sem = np.std(concat, axis=1)/np.sqrt(num_trials)

        x = [i for i in range(krange[0], krange[1]+1, 20)]

        plt.plot(
            x, Y_mean)
        plt.fill_between(x, Y_mean - Y_sem, Y_mean +
                         Y_sem, alpha=0.5)
        # plt.legend()
        res_Y_mean = {key: value for key, value in zip(x, Y_mean)}
        res_Y_sem = {key: value for key, value in zip(x, Y_sem)}
        print(concat)

        plt.title(
            'Projection. \nn=%i, beta=%.1f, p=%.1f, nrecurrent=%i, q=%.2f, ntrials=%i, m=%i' % (num_neurons, beta, connection_p, nrecurrent_rounds, q, num_trials, m))
        plt.xlabel('Capk size')
        plt.ylabel('Capacity (number of classes)')

        # save figure
        output_directory = 'figures/project_capacity/capacity_wrt_k/'
        output_filepath = output_directory + plot_name
        plt.savefig(output_filepath, format='pdf')

        plt.show()
    return res_Y_mean, res_Y_sem


def test_capacity_in_projection_as_a_function_of_beta_with_linear_classifier(betarange, k=50, num_neurons=250, nrounds=5, m=50,
                                                                             connection_p=0.1, r=0.9, q=0.01,
                                                                             num_samples=50, with_normalization=True, wipe_y=True,
                                                                             nrecurrent_rounds=5, num_trials=5,
                                                                             plot_all=True, plot_name='plot1.pdf'):
    '''
    We test for capacity in assembly calculus with respect to 
    confusion between average in- and out- class similarity
    as a function of capK size.
    When the average of out-class similarity exceeds or equal to in-class similarity,
    the assembly model hits its capacity (catastrophic forgetting). 
    try [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    '''
    result = []
    global_starting_classes = 10

    concat = np.zeros((len(betarange), num_trials))
    count = 0
    # assume linearity, each class will start with previous
    for beta in betarange:
        print('******* searching with %.4f for beta **********' % (beta))
        avg_capacity = []

        for itrial in range(num_trials):
            try_class = global_starting_classes
            print('==================================================')
            while True:
                print('beta='+str(beta)+', nclasses='+str(try_class))
                print('current trial', itrial)
                avg_assm_overlap_outside_class, avg_assm_overlap_within_class = multiround_test_capacity_using_projection_with_linear_classifier(num_neurons=num_neurons, nrounds=nrounds, beta=beta,
                                                                                                                                                 nclasses=try_class, m=m,
                                                                                                                                                 k=k, connection_p=connection_p, r=r, q=q,
                                                                                                                                                 num_samples=num_samples,
                                                                                                                                                 with_normalization=with_normalization, wipe_y=wipe_y,
                                                                                                                                                 nrecurrent_rounds=nrecurrent_rounds)
                if avg_assm_overlap_within_class <= avg_assm_overlap_outside_class:
                    avg_capacity.append(try_class)
                    print()
                    break
                else:
                    try_class += 1
                    print()
            print('==================================================')
        result.append(round(np.median(avg_capacity)))
        concat[count] = avg_capacity

        count += 1
        # global_starting_classes = round(np.median(avg_capacity))

    if plot_all:
        Y_mean = np.round(np.median(concat, axis=1))
        # print('compare should be equal', Y_mean, result_Y)
        Y_sem = np.std(concat, axis=1)/np.sqrt(num_trials)

        x = betarange

        # m1, b1 = np.polyfit(x, Y_mean, 1)

        plt.plot(
            x, Y_mean)
        plt.fill_between(x, Y_mean - Y_sem, Y_mean +
                         Y_sem, alpha=0.5)
        # plt.legend()
        res_Y_mean = {key: value for key, value in zip(x, Y_mean)}
        res_Y_sem = {key: value for key, value in zip(x, Y_sem)}
        print(concat)
        print('mean', res_Y_mean)
        print('sem', res_Y_sem)

        plt.title(
            'Projection. \nn=%i, k=m=%i, p=%.1f, nrecurrent=%i, q=%.2f, ntrials=%i' % (num_neurons, k, connection_p, nrecurrent_rounds, q, num_trials))
        plt.xlabel('beta')
        plt.ylabel('Capacity (number of classes)')

        # save figure
        output_directory = 'figures/project_capacity/capacity_wrt_beta/'
        output_filepath = output_directory + plot_name
        plt.savefig(output_filepath, format='pdf')

        plt.show()
    return res_Y_mean, res_Y_sem


def test_capacity_in_reciprocal_projection_as_a_function_of_brain_size_with_linear_classifier(nrange, nrounds=5, beta=0.1, m=None,
                                                                                              k=50, connection_p=0.1, r=0.9, q=0.01,
                                                                                              num_samples=50, with_normalization=True, wipe_y=True,
                                                                                              nrecurrent_rounds=5, num_trials=5,
                                                                                              plot_all=True, plot_name='plot1.pdf', residual_reci_project=False,
                                                                                              classifier=False, show_input_overlap=False, use_average=True):
    '''
    We test for capacity in assembly calculus with respect to 
    confusion between average in- and out- class similarity
    as a function of brain size.
    When the average of out-class similarity exceeds or equal to in-class similarity,
    the assembly model hits its capacity (catastrophic forgetting). 
    '''
    # result_Y, result_Z = [], []
    global_starting_classes = 2

    concat_Y = np.zeros(
        (nrange[1]//100, num_trials))
    concat_Z = np.zeros(
        (nrange[1]//100, num_trials))

    count = 0
    # assume linearity, each class will start with previous
    for n in range(nrange[0], nrange[1]+100, 100):
        print('******* searching with %i neurons **********' % (n))
        avg_capacity_Y = []
        avg_capacity_Z = []

        for itrial in range(num_trials):
            try_class = global_starting_classes
            obtain_capacity_of_Y, obtain_capacity_of_Z = False, False
            print('==================================================')
            while True:
                print('num_neurons='+str(n)+', k='+str(k)+', m=' +
                      str(k)+', nclasses='+str(try_class), 'current trial', itrial)
                overlap_outside_class_Y, overlap_within_class_Y, overlap_outside_class_Z,  overlap_within_class_Z = multiround_test_capacity_using_reciprocal_projection_with_linear_classifier(num_neurons=n, nrounds=nrounds, beta=beta,
                                                                                                                                                                                                nclasses=try_class, m=m,
                                                                                                                                                                                                k=k, connection_p=connection_p, r=r, q=q,
                                                                                                                                                                                                num_samples=num_samples,
                                                                                                                                                                                                with_normalization=with_normalization, wipe_y=wipe_y,
                                                                                                                                                                                                nrecurrent_rounds=nrecurrent_rounds, residual_reci_project=residual_reci_project,
                                                                                                                                                                                                classifier=classifier, show_input_overlap=show_input_overlap, use_average=use_average)
                if not obtain_capacity_of_Y and overlap_within_class_Y <= overlap_outside_class_Y:
                    avg_capacity_Y.append(try_class)
                    obtain_capacity_of_Y = True
                    # break
                if not obtain_capacity_of_Z and overlap_within_class_Z - 0.05 <= overlap_outside_class_Z:
                    avg_capacity_Z.append(try_class)
                    obtain_capacity_of_Z = True

                if obtain_capacity_of_Y and obtain_capacity_of_Z:
                    print('Y found: ', obtain_capacity_of_Y,
                          '.Z found: ', obtain_capacity_of_Z)
                    print()
                    break
                else:
                    print('Y found: ', obtain_capacity_of_Y,
                          '. Z found: ', obtain_capacity_of_Z)
                    print()
                    try_class += 1
            print('==================================================')
        concat_Y[count] = avg_capacity_Y
        concat_Z[count] = avg_capacity_Z
        count += 1

        global_starting_classes = min(
            round(min(avg_capacity_Z)), round(min(avg_capacity_Y)))

    if plot_all:
        Y_mean = np.round(np.median(concat_Y, axis=1))
        # print('compare should be equal', Y_mean, result_Y)
        Y_sem = np.std(concat_Y, axis=1)/np.sqrt(num_trials)

        Z_mean = np.round(np.median(concat_Z, axis=1))
        # print('compare should be equal', Z_mean, result_Z)
        Z_sem = np.std(concat_Z, axis=1)/np.sqrt(num_trials)

        x = [i for i in range(nrange[0], nrange[1]+100, 100)]
        # compute capacity
        m1, b1 = np.polyfit(x, Y_mean, 1)
        m2, b2 = np.polyfit(x, Z_mean, 1)

        res_Y_mean = {key: value for key, value in zip(x, Y_mean)}
        res_Y_sem = {key: value for key, value in zip(x, Y_sem)}
        res_Z_mean = {key: value for key, value in zip(x, Z_mean)}
        res_Z_sem = {key: value for key, value in zip(x, Z_sem)}
        print('Y.', concat_Y)
        print('Z.', concat_Z)

        plt.plot(
            x, Y_mean, label=f"Area 1. y = {round(m1,4)} * x + {round(b1,4)}")
        plt.fill_between(x, Y_mean - Y_sem, Y_mean +
                         Y_sem, alpha=0.5)
        plt.plot(
            x, Z_mean, label=f"Area 2. y = {round(m2,4)} * x + {round(b2,4)}")
        plt.fill_between(x, Z_mean - Z_sem, Z_mean +
                         Z_sem, alpha=0.5)
        plt.legend()

        plt.title(
            'Reciprocal Projection. \nk=%i, beta=%.1f, p=%.1f, nrecurrent=%i, ntrials=%i' % (k, beta, connection_p, nrecurrent_rounds, num_trials))
        plt.xlabel('Number of neurons')
        plt.ylabel('Capacity (number of classes)')

        # save figure
        output_directory = 'figures/reciprocal_project_capacity/capacity_wrt_n/'
        output_filepath = output_directory + plot_name
        plt.savefig(output_filepath, format='pdf')
        plt.show()

    return res_Y_mean, res_Y_sem, res_Z_mean, res_Z_sem


def test_capacity_in_reciprocal_projection_as_a_function_of_capk_size_with_linear_classifier(krange, num_neurons=250, nrounds=5, beta=0.1, m=50,
                                                                                             connection_p=0.1, r=0.9, q=0.01,
                                                                                             num_samples=50, with_normalization=True, wipe_y=True,
                                                                                             nrecurrent_rounds=5, num_trials=5,
                                                                                             plot_all=True, plot_name='plot1.pdf',  residual_reci_project=False,
                                                                                             classifier=False, show_input_overlap=False, use_average=True):
    '''
    We test for capacity in assembly calculus with respect to 
    confusion between average in- and out- class similarity
    as a function of capK size.
    When the average of out-class similarity exceeds or equal to in-class similarity,
    the assembly model hits its capacity (catastrophic forgetting). 
    '''
    global_starting_classes = 2

    concat_Y = np.zeros(
        ((krange[1]-krange[0])//20 + 1, num_trials))
    concat_Z = np.zeros(
        ((krange[1]-krange[0])//20 + 1, num_trials))

    count = 0
    # assume linearity, each class will start with previous
    for k in range(krange[0], krange[1]+1, 20):
        print('******* searching with %i for capk **********' % (k))
        avg_capacity_Y = []
        avg_capacity_Z = []

        for itrial in range(num_trials):
            try_class = global_starting_classes
            obtain_capacity_of_Y, obtain_capacity_of_Z = False, False
            print('==================================================')
            while True:
                print('k='+str(k)+', m=' +
                      str(m)+', nclasses='+str(try_class))
                print('current trial', itrial)
                overlap_outside_class_Y, overlap_within_class_Y, overlap_outside_class_Z,  overlap_within_class_Z = multiround_test_capacity_using_reciprocal_projection_with_linear_classifier(num_neurons=num_neurons, nrounds=nrounds, beta=beta,
                                                                                                                                                                                                nclasses=try_class, m=m,
                                                                                                                                                                                                k=k, connection_p=connection_p, r=r, q=q,
                                                                                                                                                                                                num_samples=num_samples,
                                                                                                                                                                                                with_normalization=with_normalization, wipe_y=wipe_y,
                                                                                                                                                                                                nrecurrent_rounds=nrecurrent_rounds, residual_reci_project=residual_reci_project,
                                                                                                                                                                                                classifier=classifier, show_input_overlap=show_input_overlap, use_average=use_average)
                if not obtain_capacity_of_Y and overlap_within_class_Y <= overlap_outside_class_Y:
                    avg_capacity_Y.append(try_class)
                    obtain_capacity_of_Y = True
                    # break
                if not obtain_capacity_of_Z and overlap_within_class_Z-0.05 <= overlap_outside_class_Z:
                    avg_capacity_Z.append(try_class)
                    obtain_capacity_of_Z = True
                if obtain_capacity_of_Y and obtain_capacity_of_Z:
                    print('Y found: ', obtain_capacity_of_Y,
                          '.Z found: ', obtain_capacity_of_Z)
                    print()
                    break
                else:
                    print('Y found: ', obtain_capacity_of_Y,
                          '. Z found: ', obtain_capacity_of_Z)
                    print()
                    try_class += 1
            print('==================================================')
        concat_Y[count] = avg_capacity_Y
        concat_Z[count] = avg_capacity_Z
        count += 1

    if plot_all:
        Y_mean = np.round(np.median(concat_Y, axis=1))
        Y_sem = np.std(concat_Y, axis=1)/np.sqrt(num_trials)
        Z_mean = np.round(np.median(concat_Z, axis=1))
        Z_sem = np.std(concat_Z, axis=1)/np.sqrt(num_trials)

        x = [i for i in range(krange[0], krange[1]+1, 20)]

        plt.plot(
            x, Y_mean, label="Area 1")
        plt.fill_between(x, Y_mean - Y_sem, Y_mean +
                         Y_sem, alpha=0.5)
        plt.plot(
            x, Z_mean, label="Area 2")
        plt.fill_between(x, Z_mean - Z_sem, Z_mean +
                         Z_sem, alpha=0.5)
        plt.legend()

        res_Y_mean = {key: value for key, value in zip(x, Y_mean)}
        res_Y_sem = {key: value for key, value in zip(x, Y_sem)}
        res_Z_mean = {key: value for key, value in zip(x, Z_mean)}
        res_Z_sem = {key: value for key, value in zip(x, Z_sem)}
        print('Y.', concat_Y)
        print('Z.', concat_Z)

        plt.title(
            'Reciprocal Projection. \nn=%i, beta=%.1f, p=%.1f, nrecurrent=%i, q=%.2f, ntrials=%i, m=%i' % (num_neurons, beta, connection_p, nrecurrent_rounds, q, num_trials, m))
        plt.xlabel('Capk size')
        plt.ylabel('Capacity (number of classes)')

        # save figure
        output_directory = 'figures/reciprocal_project_capacity/capacity_wrt_k/'
        output_filepath = output_directory + plot_name
        plt.savefig(output_filepath, format='pdf')

        plt.show()
    return res_Y_mean, res_Y_sem, res_Z_mean, res_Z_sem


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find capacity of A.C. wrt n or k")
    parser.add_argument("--operation", type=str, required=True,
                        help="Operations names (project or reci-project)")
    parser.add_argument("--parameter", type=str,
                        required=True, help="Parameter names (n or k)")
    parser.add_argument("--ntrials", type=int,
                        required=True, help="number of trials")
    parser.add_argument("--plot", type=str, required=True,
                        help="name of the plot")
    parser.add_argument("--skipConnection", type=bool,
                        required=True, help="whether to use skip connection")

    args = parser.parse_args()
    # Access the variables
    operation = args.operation
    parameter = args.parameter
    ntrials = args.ntrials
    plot = args.plot
    skipConnection = args.skipConnection

    if operation == "project":
        # project wrt k
        if parameter == 'k':
            test_capacity_in_projection_as_a_function_of_capk_size_with_linear_classifier([
                20, 240], plot_name='%s.pdf' % (plot), num_trials=ntrials)
        # project wrt n
        if parameter == 'n':
            test_capacity_in_projection_as_a_function_of_brain_size_with_linear_classifier(
                [100, 800], num_trials=ntrials, q=0.01, plot_name='%s.pdf' % (plot), show_input_overlap=False, classifier=False)

    # # reciprocal-project
    elif operation == 'reci-project':
        if parameter == 'k':
            test_capacity_in_reciprocal_projection_as_a_function_of_capk_size_with_linear_classifier([
                20, 240], plot_name='%s.pdf' % (plot), num_trials=ntrials, residual_reci_project=skipConnection, classifier=False, use_average=True)
        if parameter == 'n':
            test_capacity_in_reciprocal_projection_as_a_function_of_brain_size_with_linear_classifier(
                [100, 800], num_trials=ntrials, nrecurrent_rounds=5, plot_name='%s.pdf' % (plot), residual_reci_project=skipConnection, classifier=False, use_average=True)

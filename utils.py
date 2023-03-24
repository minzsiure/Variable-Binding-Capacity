import numpy as np
import copy
from sklearn.linear_model import LogisticRegression
from brain import *
from utils import *
from stimuli import *

# setting global random seed
rng = np.random.default_rng(2022)


def overlap(a, b):
    return len(set(a) & set(b))


def capk(input, cap_size):
    output = np.zeros_like(input)
    if len(input.shape) == 1:
        idx = np.argsort(input)[-cap_size:]
        output[idx] = 1
    else:
        idx = np.argsort(input, axis=-1)[:, -cap_size:]
        np.put_along_axis(output, idx, 1, axis=-1)
    return output


def hebbian_update(in_vec, out_vec, W, beta):
    for i in np.where(out_vec != 0.)[0]:
        for j in np.where(in_vec != 0.)[0]:
            W[i, j] *= 1. + beta
    return W


def find_feedforward_matrix_index(area_combinations, from_index, to_index):
    '''
    return the index of pair [from_index, to_index] in the list Brain.area_combinations
    '''
    for i, acomb in enumerate(area_combinations):
        if all(acomb == [from_index, to_index]):
            return i


def generate_labels(num_classes, n):
    '''
    Generate labels based on number of classes, each class has n labels
    '''
    labels = []
    for i in range(num_classes):
        labels += [i] * n
    return np.array(labels)


def hamming_distance(vec1, vec2):
    '''
    calculate hamming distance between the two vectors.
    vec1, vec2 both have to be binary vectors
    '''
    return np.count_nonzero(vec1 != vec2)


def use_stimuli_form_assembly_stack_and_classify_in_projection(stimuli_coreset, brain_initial,
                                                               nclasses, paired_areas, n_samples=50,
                                                               nrecurrent_rounds=5, num_neurons=1000, k=100,
                                                               return_accuracy='test', verbose=True):
    """
    Input:  stimuli_coreset (Stimuli object)
            brain (Brain object)
            nclasses (int): number of classes 
            n_samples: number of samples per class
            nrecurrent_rounds: number of rounds for recurrence
            paired_areas: tuples of mapping (from,to). We assume there are only two tuples.
            num_neurons: number of neurons
            k: number of winners
    Output: * brain (a copy of Brain object after updating, without modify input)
            * stack_of_assemblies (list): [outputs_from_left_train, outputs_from_left_test, 
                                        outputs_from_right_train, outputs_from_right_test]
            * accuracy (list): [left_tr_acc, left_te_acc, right_tr_acc, right_te_acc]
    Procedure:
    1. initalize 4 set of inputs from stimuli_coreset: 
    inputs_to_left_train = stimuli_coreset.generate_stimuli_set()
    inputs_to_left_test = stimuli_coreset.generate_stimuli_set()
    inputs_to_right_train = stimuli_coreset.generate_stimuli_set()
    inputs_to_right_test = stimuli_coreset.generate_stimuli_set()
    2. initialize 4 set of assembly outputs (all zeros):
    outputs_from_left_train = np.zeros((nclasses, n_samples, num_neurons))
    outputs_from_left_test = np.zeros((nclasses, n_samples, num_neurons))
    outputs_from_right_train = np.zeros((nclasses, n_samples, num_neurons))
    outputs_from_right_test = np.zeros((nclasses, n_samples, num_neurons))
    3. find feedforward connection with info provided inpaired_areas
    4. for nclasses, for nsamples, for nrecurrent_rounds, form assemblies
    5. stack all assemblies    
    output_from_left_train_stack = np.vstack(tuple([np.vstack(tuple(
        [outputs_from_left_train[j, i] for i in range(n_samples)])) for j in range(nclasses)]))
    output_from_left_test_stack = np.vstack(tuple([np.vstack(tuple(
        [outputs_from_left_test[j, i] for i in range(n_samples)])) for j in range(nclasses)]))
    output_from_right_train_stack = np.vstack(tuple([np.vstack(tuple(
        [outputs_from_right_train[j, i] for i in range(n_samples)])) for j in range(nclasses)]))
    output_from_right_test_stack = np.vstack(tuple([np.vstack(tuple(
        [outputs_from_right_test[j, i] for i in range(n_samples)])) for j in range(nclasses)]))
    6. generate labels, and fit classifiers.
    7. measure accuracies and return outputs. 
    """
    brain = copy.deepcopy(brain_initial)

    # 1. initalize 4 set of inputs from stimuli_coreset:
    inputs_to_left_train = stimuli_coreset.generate_stimuli_set()
    inputs_to_left_test = stimuli_coreset.generate_stimuli_set()
    inputs_to_right_train = stimuli_coreset.generate_stimuli_set()
    inputs_to_right_test = stimuli_coreset.generate_stimuli_set()

    # 2. initialize 4 set of assembly outputs (all zeros):
    outputs_from_left_train = np.zeros((nclasses, n_samples, num_neurons))
    outputs_from_left_test = np.zeros((nclasses, n_samples, num_neurons))
    outputs_from_right_train = np.zeros((nclasses, n_samples, num_neurons))
    outputs_from_right_test = np.zeros((nclasses, n_samples, num_neurons))

    # 3. find feedforward connection with info provided inpaired_areas
    x_to_left_feedforward = find_feedforward_matrix_index(
        brain.area_combinations, paired_areas[0][0], paired_areas[0][1])
    x_to_right_feedforward = find_feedforward_matrix_index(
        brain.area_combinations, paired_areas[1][0], paired_areas[1][1])

    # 4. for nclasses, for nsamples, for nrecurrent_rounds, form assemblies
    for j in range(nclasses):
        for isample in range(n_samples):
            temp_area1_activations_train = outputs_from_left_train[j, isample]
            temp_area1_activations_test = outputs_from_left_test[j, isample]

            temp_area2_activations_train = outputs_from_right_train[j, isample]
            temp_area2_activations_test = outputs_from_right_test[j, isample]

            for iround in range(nrecurrent_rounds):
                # left hemifield
                temp_area1_activations_train = capk(brain.feedforward_connections[x_to_left_feedforward].dot(inputs_to_left_train[j, isample])
                                                    + brain.areas[paired_areas[0][1]].recurrent_connections.dot(temp_area1_activations_train),
                                                    k)
                temp_area1_activations_test = capk(brain.feedforward_connections[x_to_left_feedforward].dot(inputs_to_left_test[j, isample])
                                                   + brain.areas[paired_areas[0][1]].recurrent_connections.dot(temp_area1_activations_test),
                                                   k)
                # right hemifield
                temp_area2_activations_train = capk(brain.feedforward_connections[x_to_right_feedforward].dot(inputs_to_right_train[j, isample])
                                                    + brain.areas[paired_areas[1][1]].recurrent_connections.dot(temp_area2_activations_train), k)
                temp_area2_activations_test = capk(brain.feedforward_connections[x_to_right_feedforward].dot(inputs_to_right_test[j, isample])
                                                   + brain.areas[paired_areas[1][1]].recurrent_connections.dot(temp_area2_activations_test),
                                                   k)
            outputs_from_left_train[j, isample] = temp_area1_activations_train
            outputs_from_left_test[j, isample] = temp_area1_activations_test
            outputs_from_right_train[j, isample] = temp_area2_activations_train
            outputs_from_right_test[j, isample] = temp_area2_activations_test

     # 5. stack all assemblies
    output_from_left_train_stack = np.vstack(tuple([np.vstack(tuple(
        [outputs_from_left_train[j, i] for i in range(n_samples)])) for j in range(nclasses)]))
    output_from_left_test_stack = np.vstack(tuple([np.vstack(tuple(
        [outputs_from_left_test[j, i] for i in range(n_samples)])) for j in range(nclasses)]))
    output_from_right_train_stack = np.vstack(tuple([np.vstack(tuple(
        [outputs_from_right_train[j, i] for i in range(n_samples)])) for j in range(nclasses)]))
    output_from_right_test_stack = np.vstack(tuple([np.vstack(tuple(
        [outputs_from_right_test[j, i] for i in range(n_samples)])) for j in range(nclasses)]))

    # 6. generate labels, and fit classifiers.
    # generate labels
    y_label = generate_labels(nclasses, n_samples)
    # fit a classifier
    classifier_LEFT = LogisticRegression(
        solver='liblinear', max_iter=80)
    classifier_RIGHT = LogisticRegression(
        solver='liblinear', max_iter=80)

    classifier_LEFT.fit(output_from_left_train_stack, y_label)
    left_fit_acc = classifier_LEFT.score(output_from_left_train_stack, y_label)
    left_tr_acc = classifier_LEFT.score(output_from_left_test_stack, y_label)
    left_te_acc = classifier_LEFT.score(output_from_right_test_stack, y_label)
    if verbose:
        print('--------------', nclasses, ' classes --------------')
        print('Fit Accuracy of L1 on Y1 is %.3f (expecting 1)' % (left_fit_acc))
        print('Test Accuracy of L1 on Y1 prime is %.3f (expecting 1)' %
              (left_tr_acc))
        print('Test Accuracy of L1 on Y2 prime %.3f (expecting %.3f)' %
              (left_te_acc, 1/nclasses))

    classifier_RIGHT.fit(output_from_right_train_stack, y_label)
    right_fit_acc = classifier_RIGHT.score(
        output_from_right_train_stack, y_label)
    right_tr_acc = classifier_RIGHT.score(
        output_from_right_test_stack, y_label)
    right_te_acc = classifier_RIGHT.score(
        output_from_left_test_stack, y_label)
    if verbose:
        print('----------------------------')
        print('Fit Accuracy of L2 on Y2 is %.3f (expecting 1)' % (right_fit_acc))
        print('Test Accuracy of L2 on Y2 prime is %.3f (expecting 1)' %
              (right_tr_acc))
        print('Test Accuracy of L2 on Y1 prime %.3f (expecting %.3f)' %
              (right_te_acc, 1/nclasses))

    output_assemblies = [outputs_from_left_train, outputs_from_left_test,
                         outputs_from_right_train, outputs_from_right_test]
    # 7. measure accuracies and return outputs.
    # return brain, output_assemblies, [left_tr_acc, left_te_acc, right_tr_acc, right_te_acc]
    if return_accuracy == 'test':
        return brain, output_assemblies, [left_te_acc, right_te_acc]
    if return_accuracy == 'train':
        return brain, output_assemblies, [left_tr_acc, right_tr_acc]
    if return_accuracy == 'fit':
        return brain, output_assemblies, [left_fit_acc, right_fit_acc]
    if return_accuracy == 'assembly_stack':
        return [output_from_left_train_stack, output_from_left_test_stack,
                output_from_right_train_stack, output_from_right_test_stack]
    if return_accuracy == 'assemblies':
        return output_assemblies


def use_assemblies_form_assembly_stack_and_classify_in_projection(output_assemblies, brain_initial,
                                                                  nclasses, paired_areas, n_samples=50,
                                                                  nrecurrent_rounds=5, num_neurons=1000,
                                                                  k=100, remove_recurrence=False,
                                                                  return_accuracy='everything'):
    """
    Input:  output_assemblies: [outputs_from_left_train, outputs_from_left_test,
                         outputs_from_right_train, outputs_from_right_test]
            brain (Brain object)
            nclasses (int): number of classes 
            n_samples: number of samples per class
            nrecurrent_rounds: number of rounds for recurrence
            paired_areas: tuples of mapping (from,to). We assume there are only two tuples.
            num_neurons: number of neurons
            k: number of winners
            remove_recurrence (Bool): whether or not to use W_cc.
    Output: * brain (a copy of Brain object after updating, without modify input)
            * stack_of_assemblies (list): [outputs_from_left_train, outputs_from_left_test, 
                                        outputs_from_right_train, outputs_from_right_test]
            * accuracy (list): [left_tr_acc, left_te_acc, right_tr_acc, right_te_acc]
    Procedure:
    1. initalize 4 set of inputs from stimuli_coreset: 
    inputs_to_left_train = stimuli_coreset.generate_stimuli_set()
    inputs_to_left_test = stimuli_coreset.generate_stimuli_set()
    inputs_to_right_train = stimuli_coreset.generate_stimuli_set()
    inputs_to_right_test = stimuli_coreset.generate_stimuli_set()
    2. initialize 4 set of assembly outputs (all zeros):
    outputs_from_left_train = np.zeros((nclasses, n_samples, num_neurons))
    outputs_from_left_test = np.zeros((nclasses, n_samples, num_neurons))
    outputs_from_right_train = np.zeros((nclasses, n_samples, num_neurons))
    outputs_from_right_test = np.zeros((nclasses, n_samples, num_neurons))
    3. find feedforward connection with info provided inpaired_areas
    4. for nclasses, for nsamples, for nrecurrent_rounds, form assemblies
    5. stack all assemblies    
    output_from_left_train_stack = np.vstack(tuple([np.vstack(tuple(
        [outputs_from_left_train[j, i] for i in range(n_samples)])) for j in range(nclasses)]))
    output_from_left_test_stack = np.vstack(tuple([np.vstack(tuple(
        [outputs_from_left_test[j, i] for i in range(n_samples)])) for j in range(nclasses)]))
    output_from_right_train_stack = np.vstack(tuple([np.vstack(tuple(
        [outputs_from_right_train[j, i] for i in range(n_samples)])) for j in range(nclasses)]))
    output_from_right_test_stack = np.vstack(tuple([np.vstack(tuple(
        [outputs_from_right_test[j, i] for i in range(n_samples)])) for j in range(nclasses)]))
    6. generate labels, and fit classifiers.
    7. measure accuracies and return outputs. 
    """
    brain = copy.deepcopy(brain_initial)

    # 1. initalize 4 set of inputs from stimuli_coreset (this step is replaced by drawing assemblies from area 1, 2)

    # 2. initialize 4 set of assembly outputs (all zeros):
    outputs_from_left_train = np.zeros((nclasses, n_samples, num_neurons))
    outputs_from_left_test = np.zeros((nclasses, n_samples, num_neurons))
    outputs_from_right_train = np.zeros((nclasses, n_samples, num_neurons))
    outputs_from_right_test = np.zeros((nclasses, n_samples, num_neurons))

    # 3. find feedforward connection with info provided inpaired_areas
    x_to_left_feedforward = find_feedforward_matrix_index(
        brain.area_combinations, paired_areas[0][0], paired_areas[0][1])
    x_to_right_feedforward = find_feedforward_matrix_index(
        brain.area_combinations, paired_areas[1][0], paired_areas[1][1])

    # 4. for nclasses, for nsamples, for nrecurrent_rounds, form assemblies
    for j in range(nclasses):
        for isample in range(n_samples):
            temp_area1_activations_train = outputs_from_left_train[j, isample]
            temp_area1_activations_test = outputs_from_left_test[j, isample]

            temp_area2_activations_train = outputs_from_right_train[j, isample]
            temp_area2_activations_test = outputs_from_right_test[j, isample]

            # get assembly as input
            left_set_index = random.randint(0, 1)
            left_index = random.randint(0, n_samples-1)
            x_to_left = output_assemblies[left_set_index][j, left_index]

            right_set_index = random.randint(2, 3)
            right_index = random.randint(0, n_samples-1)
            x_to_right = output_assemblies[right_set_index][j, right_index]

            # project with rcurrence
            for iround in range(nrecurrent_rounds):
                if not remove_recurrence:
                    # left hemifield
                    temp_area1_activations_train = capk(brain.feedforward_connections[x_to_left_feedforward].dot(x_to_left)
                                                        + brain.areas[paired_areas[0][1]].recurrent_connections.dot(temp_area1_activations_train),
                                                        k)
                    temp_area1_activations_test = capk(brain.feedforward_connections[x_to_left_feedforward].dot(x_to_left)
                                                       + brain.areas[paired_areas[0][1]].recurrent_connections.dot(temp_area1_activations_test),
                                                       k)
                    # right hemifield
                    temp_area2_activations_train = capk(brain.feedforward_connections[x_to_right_feedforward].dot(x_to_right)
                                                        + brain.areas[paired_areas[1][1]].recurrent_connections.dot(temp_area2_activations_train), k)
                    temp_area2_activations_test = capk(brain.feedforward_connections[x_to_right_feedforward].dot(x_to_right)
                                                       + brain.areas[paired_areas[1][1]].recurrent_connections.dot(temp_area2_activations_test),
                                                       k)
                else:
                    # left hemifield
                    temp_area1_activations_train = capk(
                        brain.feedforward_connections[x_to_left_feedforward].dot(x_to_left), k)
                    temp_area1_activations_test = capk(
                        brain.feedforward_connections[x_to_left_feedforward].dot(x_to_left), k)
                    # right hemifield
                    temp_area2_activations_train = capk(
                        brain.feedforward_connections[x_to_right_feedforward].dot(x_to_right), k)
                    temp_area2_activations_test = capk(
                        brain.feedforward_connections[x_to_right_feedforward].dot(x_to_right), k)

            outputs_from_left_train[j, isample] = temp_area1_activations_train
            outputs_from_left_test[j, isample] = temp_area1_activations_test
            outputs_from_right_train[j, isample] = temp_area2_activations_train
            outputs_from_right_test[j, isample] = temp_area2_activations_test

     # 5. stack all assemblies
    output_from_left_train_stack = np.vstack(tuple([np.vstack(tuple(
        [outputs_from_left_train[j, i] for i in range(n_samples)])) for j in range(nclasses)]))
    output_from_left_test_stack = np.vstack(tuple([np.vstack(tuple(
        [outputs_from_left_test[j, i] for i in range(n_samples)])) for j in range(nclasses)]))
    output_from_right_train_stack = np.vstack(tuple([np.vstack(tuple(
        [outputs_from_right_train[j, i] for i in range(n_samples)])) for j in range(nclasses)]))
    output_from_right_test_stack = np.vstack(tuple([np.vstack(tuple(
        [outputs_from_right_test[j, i] for i in range(n_samples)])) for j in range(nclasses)]))

    # 6. generate labels, and fit classifiers.
    # generate labels
    y_label = generate_labels(nclasses, n_samples)
    # fit a classifier
    classifier_LEFT = LogisticRegression(
        solver='liblinear', max_iter=80)
    classifier_RIGHT = LogisticRegression(
        solver='liblinear', max_iter=80)

    classifier_LEFT.fit(output_from_left_train_stack, y_label)
    left_fit_acc = classifier_LEFT.score(output_from_left_train_stack, y_label)
    left_tr_acc = classifier_LEFT.score(output_from_left_test_stack, y_label)
    left_te_acc = classifier_LEFT.score(output_from_right_test_stack, y_label)
    # print('--------------', nclasses, ' classes --------------')
    # print('Fit Accuracy of L1 on Y1 is %.3f (expecting 1)' % (left_fit_acc))
    # print('Test Accuracy of L1 on Y1 prime is %.3f (expecting 1)' %
    #       (left_tr_acc))
    # print('Test Accuracy of L1 on Y2 prime %.3f (expecting %.3f)' %
    #       (left_te_acc, 1/nclasses))

    classifier_RIGHT.fit(output_from_right_train_stack, y_label)
    right_fit_acc = classifier_RIGHT.score(
        output_from_right_train_stack, y_label)
    right_tr_acc = classifier_RIGHT.score(
        output_from_right_test_stack, y_label)
    right_te_acc = classifier_RIGHT.score(
        output_from_left_test_stack, y_label)
    # print('----------------------------')
    # print('Fit Accuracy of L2 on Y2 is %.3f (expecting 1)' % (right_fit_acc))
    # print('Test Accuracy of L2 on Y2 prime is %.3f (expecting 1)' %
    #       (right_tr_acc))
    # print('Test Accuracy of L2 on Y1 prime %.3f (expecting %.3f)' %
    #       (right_te_acc, 1/nclasses))
    # print()

    # 7. measure accuracies and return outputs.
    # return brain, output_assemblies, [left_tr_acc, left_te_acc, right_tr_acc, right_te_acc]
    output_assemblies = [outputs_from_left_train, outputs_from_left_test,
                         outputs_from_right_train, outputs_from_right_test]
    output_assemblies_stack = [output_from_left_train_stack, output_from_left_test_stack,
                               output_from_right_train_stack, output_from_right_test_stack]
    if return_accuracy == 'everything':
        return brain, output_assemblies, output_assemblies_stack, [classifier_LEFT, classifier_RIGHT]
    if return_accuracy == 'test':
        return brain, output_assemblies, [left_te_acc, right_te_acc]
    if return_accuracy == 'train':
        return brain, output_assemblies, [left_tr_acc, right_tr_acc]
    if return_accuracy == 'fit':
        return brain, output_assemblies, [left_fit_acc, right_fit_acc]
    if return_accuracy == 'classifier':
        return [classifier_LEFT, classifier_RIGHT]


def use_stimuli_form_assembly_stack_and_classify_in_reciprocal_projection(stimuli_coreset, brain_initial,
                                                                          nclasses, n_samples=50, nrecurrent_rounds=5,
                                                                          num_neurons=1000, k=100,
                                                                          return_accuracy='test',
                                                                          area0=0, area1=1, area2=2):
    """
    Input:  stimuli_coreset (Stimuli object)
            brain (Brain object)
            nclasses (int): number of classes 
            n_samples: number of samples per class
            nrecurrent_rounds: number of rounds for recurrence
            paired_areas: tuples of mapping (from,to). We assume there are only two tuples.
            num_neurons: number of neurons
            k: number of winners
    Output: * brain (a copy of Brain object after updating, without modify input)
            * stack_of_assemblies (list): [outputs_from_left_train, outputs_from_left_test, 
                                        outputs_from_right_train, outputs_from_right_test]
            * accuracy (list): [left_tr_acc, left_te_acc, right_tr_acc, right_te_acc]
    Procedure:
    1. initalize 2 ses of inputs from stimuli_coreset: 
    inputs_to_train = stimuli_coreset.generate_stimuli_set()
    inputs_to_test = stimuli_coreset.generate_stimuli_set()
    2. initialize 4 set of assembly outputs (all zeros):
    outputs_y_train = np.zeros((nclasses, n_samples, num_neurons))
    outputs_y_test = np.zeros((nclasses, n_samples, num_neurons))
    outputs_z_train = np.zeros((nclasses, n_samples, num_neurons))
    outputs_z_test = np.zeros((nclasses, n_samples, num_neurons))
    3. find feedforward connection 0to1, 1to2, 2to1
    4. for nclasses, for nsamples, for nrecurrent_rounds, form assemblies Y, Z in area 1, area 2
        repeat again for Y', Z'.
    5. stack all assemblies    
    Y = np.vstack(tuple([np.vstack(tuple(
        [outputs_y_train[j, i] for i in range(n_samples)])) for j in range(nclasses)]))
    Z = np.vstack(tuple([np.vstack(tuple(
        [outputs_z_train[j, i] for i in range(n_samples)])) for j in range(nclasses)]))
    Y_prime = np.vstack(tuple([np.vstack(tuple(
        [outputs_y_test[j, i] for i in range(n_samples)])) for j in range(nclasses)]))
    Z_prime = np.vstack(tuple([np.vstack(tuple(
        [outputs_z_test[j, i] for i in range(n_samples)])) for j in range(nclasses)]))
    6. generate labels, and fit classifiers. L1 on Y, L2 on Z.
    7. measure accuracies and return outputs. 
    """
    brain = copy.deepcopy(brain_initial)

    # 1. initalize 4 set of inputs from stimuli_coreset:
    inputs_to_train = stimuli_coreset.generate_stimuli_set()
    inputs_to_test = stimuli_coreset.generate_stimuli_set()

    # 2. initialize 4 set of assembly outputs (all zeros):
    outputs_y_train = np.zeros((nclasses, n_samples, num_neurons))
    outputs_z_train = np.zeros((nclasses, n_samples, num_neurons))
    outputs_y_test = np.zeros((nclasses, n_samples, num_neurons))
    outputs_z_test = np.zeros((nclasses, n_samples, num_neurons))

    # 3. find feedforward connection with info provided inpaired_areas
    iweights_1to2 = find_feedforward_matrix_index(
        brain.area_combinations, area0, area1)  # feedforward
    iweights_2to3 = find_feedforward_matrix_index(
        brain.area_combinations, area1, area2)  # feedforward
    iweights_3to2 = find_feedforward_matrix_index(
        brain.area_combinations, area2, area1)  # feedbackward

    # 4. for nclasses, for nsamples, for nrecurrent_rounds, form assemblies
    # train set
    for j in range(nclasses):
        for isample in range(n_samples):
            temp_area2_activations_train = outputs_y_train[j, isample]
            temp_area3_activations_train = outputs_z_train[j, isample]

            for iround in range(nrecurrent_rounds):
                x = inputs_to_train[j, isample]
                new_area2_activations = capk(brain.feedforward_connections[iweights_1to2].dot(x)
                                             + brain.areas[area1].recurrent_connections.dot(temp_area2_activations_train)
                                             + brain.feedforward_connections[iweights_3to2].dot(temp_area3_activations_train),
                                             k)
                new_area3_activations = capk(brain.feedforward_connections[iweights_2to3].dot(temp_area2_activations_train)
                                             + brain.areas[area2].recurrent_connections.dot(temp_area3_activations_train),
                                             k)
                temp_area2_activations_train = new_area2_activations
                temp_area3_activations_train = new_area3_activations

            # only record the last round of activations, after utilizing recurrent connections
            outputs_y_train[j, isample] = temp_area2_activations_train
            outputs_z_train[j, isample] = temp_area3_activations_train

    # test set
    for j in range(nclasses):
        for isample in range(n_samples):
            temp_area2_activations_test = outputs_y_test[j, isample]
            temp_area3_activations_test = outputs_z_test[j, isample]

            for iround in range(nrecurrent_rounds):
                x = inputs_to_test[j, isample]
                new_area2_activations = capk(brain.feedforward_connections[iweights_1to2].dot(x)
                                             + brain.areas[area1].recurrent_connections.dot(temp_area2_activations_test)
                                             + brain.feedforward_connections[iweights_3to2].dot(temp_area3_activations_test),
                                             k)
                new_area3_activations = capk(brain.feedforward_connections[iweights_2to3].dot(temp_area2_activations_test)
                                             + brain.areas[area2].recurrent_connections.dot(temp_area3_activations_test),
                                             k)
                temp_area2_activations_test = new_area2_activations
                temp_area3_activations_test = new_area3_activations

            # only record the last round of activations, after utilizing recurrent connections
            outputs_y_test[j, isample] = temp_area2_activations_test
            outputs_z_test[j, isample] = temp_area3_activations_test

     # 5. stack all assemblies
    output_y_train_stack = np.vstack(tuple([np.vstack(tuple(
        [outputs_y_train[j, i] for i in range(n_samples)])) for j in range(nclasses)]))
    output_y_test_stack = np.vstack(tuple([np.vstack(tuple(
        [outputs_y_test[j, i] for i in range(n_samples)])) for j in range(nclasses)]))
    output_z_train_stack = np.vstack(tuple([np.vstack(tuple(
        [outputs_z_train[j, i] for i in range(n_samples)])) for j in range(nclasses)]))
    output_z_test_stack = np.vstack(tuple([np.vstack(tuple(
        [outputs_z_test[j, i] for i in range(n_samples)])) for j in range(nclasses)]))

    # 6. generate labels, and fit classifiers.
    # generate labels
    y_label = generate_labels(nclasses, n_samples)
    # fit a classifier
    classifier_LEFT = LogisticRegression(
        solver='liblinear', max_iter=80)
    classifier_RIGHT = LogisticRegression(
        solver='liblinear', max_iter=80)

    classifier_LEFT.fit(output_y_train_stack, y_label)
    left_fit_acc = classifier_LEFT.score(output_y_train_stack, y_label)
    left_tr_acc = classifier_LEFT.score(output_y_test_stack, y_label)
    left_te_acc = classifier_LEFT.score(output_z_test_stack, y_label)
    print('--------------', nclasses, ' classes --------------')
    print('Fit Accuracy of L1 on Y is %.3f (expecting 1)' % (left_fit_acc))
    print('Test Accuracy of L1 on Y prime is %.3f (expecting 1)' %
          (left_tr_acc))
    print('Test Accuracy of L1 on Z prime %.3f (expecting %.3f)' %
          (left_te_acc, 1/nclasses))

    classifier_RIGHT.fit(output_z_train_stack, y_label)
    right_fit_acc = classifier_RIGHT.score(
        output_z_train_stack, y_label)
    right_tr_acc = classifier_RIGHT.score(
        output_z_test_stack, y_label)
    right_te_acc = classifier_RIGHT.score(
        output_y_test_stack, y_label)
    print('----------------------------')
    print('Fit Accuracy of L2 on Z is %.3f (expecting 1)' % (right_fit_acc))
    print('Test Accuracy of L2 on Z prime is %.3f (expecting 1)' %
          (right_tr_acc))
    print('Test Accuracy of L2 on Y prime %.3f (expecting %.3f)' %
          (right_te_acc, 1/nclasses))

    output_assemblies = [outputs_y_train, outputs_y_test,
                         outputs_z_train, outputs_z_test]
    # 7. measure accuracies and return outputs.
    # return brain, output_assemblies, [left_tr_acc, left_te_acc, right_tr_acc, right_te_acc]
    # return brain, output_assemblies, [left_tr_acc, left_te_acc, right_tr_acc, right_te_acc]
    if return_accuracy == 'test':
        return brain, output_assemblies, [left_te_acc, right_te_acc]
    if return_accuracy == 'train':
        return brain, output_assemblies, [left_tr_acc, right_tr_acc]
    if return_accuracy == 'fit':
        return brain, output_assemblies, [left_fit_acc, right_fit_acc]
    if return_accuracy == 'classifier':
        return [classifier_LEFT, classifier_RIGHT]
    if return_accuracy == 'z_test_assembly_stack':
        return output_z_test_stack
    if return_accuracy == 'assembly_stack':
        return [output_y_train_stack, output_y_test_stack, outputs_z_train, outputs_z_test]
    if return_accuracy == 'assembly':
        return output_assemblies

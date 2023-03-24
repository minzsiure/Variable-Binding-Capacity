import numpy as np
from utils import *
from area import *
from scipy.sparse import csr_matrix


class Brain:
    '''
    A Brain contains multiple Areas.
    There may exist feedforward connections between Areas.
    '''

    def __init__(self, num_areas=2, n=1000, beta=0.1, p=None, k=None, sparse=False):
        self.num_areas = num_areas
        self.n = n
        self.sparse = sparse

        # init p
        if p != None:
            self.p = p
        else:
            self.p = 1/np.sqrt(self.n)

        # init beta
        self.beta = beta

        # a list of brain areas
        self.areas = [Area(n=self.n, p=self.p, k=k, sparse=self.sparse)
                      for i in range(self.num_areas)]

        # store all pairs of Area indices where
        # there could be a feedforward connection between them
        self.area_combinations = np.array(
            np.where(~np.eye(self.num_areas, dtype=bool))).T

        # init feedforward connections for all pairs
        self.feedforward_connections = [
            self.sample_initial_connections() for i in range(len(self.area_combinations))]

    def sample_initial_connections(self):
        '''
        Draw feedforward connections (binary) based on p
        '''
        connections = np.random.binomial(
            1, self.p, size=(self.n, self.n)).astype("float64")

        if self.sparse:
            return csr_matrix(connections)
        else:
            return connections

    def normalize_connections(self):
        '''
        Normalize connection weights by dividing by sum along axis 0
        '''
        # normalize feedforward connection
        for i in range(len(self.feedforward_connections)):
            self.feedforward_connections[i] /= self.feedforward_connections[i].sum(
                axis=0)

        # normalize recurrent connection
        for j in range(self.num_areas):
            self.areas[j].recurrent_connections /= self.areas[j].recurrent_connections.sum(
                axis=0)

    def wipe_all_activations(self):
        '''
        Reset all activations to be 0
        '''
        for i in range(self.num_areas):
            if self.sparse:
                self.areas[i].activations = csr_matrix(
                    np.zeros(
                        self.areas[i].n, dtype=float))
            else:
                self.areas[i].activations = np.zeros(
                    self.areas[i].n, dtype=float)

    def project(self, input_activation, from_area_index, to_area_index,
                max_iterations=50, verbose=0,
                return_stable_rank=False,
                return_weights_assembly=False,
                only_once=False, track_neuron=False):
        '''
        Fire a fix set of neurons in Area from_area_index onto Area to_area_index.
        Input_activation is a set of activated neurons (binary) in from_area
        '''
        if self.sparse:
            input_activation = csr_matrix(input_activation)

        # disinhibit the to-area Area so it does recurrent/feedforward activation
        self.areas[to_area_index].disinhibit()

        # find the index storing feedforward connection matrix
        iweights = find_feedforward_matrix_index(
            self.area_combinations, from_area_index, to_area_index)

        if not only_once:
            # randomly sample initial feedforward connections
            self.feedforward_connections[iweights] = self.sample_initial_connections(
            )

            # randomly sample initial self-recurrent connections
            self.areas[to_area_index].recurrent_connections = self.areas[to_area_index].sample_initial_connections()

        # track indices and total numbers of neurons activated so far in each iteration
        touched_neurons = set()
        touched_neurons_size = []

        # record winners for the previous iteration
        prev_winners = self.areas[to_area_index].activations

        # initialize stable rank ratio
        stable_rank_ratio = []

        # project from from-area to to-area repeatedly, until convergence or max_iterations reached
        for t in range(max_iterations):
            if verbose and (t != 0) and (t % 50 == 0):
                print("\titeration", t)
            if only_once:
                assert t < 1

            # send input activations from from-area to to-area through opened connections
            feedforward_activations = self.feedforward_connections[iweights].dot(
                input_activation)

            # do self-recurrent activation in to-area (the first self-recurrent does not have any effect because activations are 0s)
            recurrent_activations = self.areas[to_area_index].recurrent_connections.dot(
                prev_winners)

            # and add the activations to the feedforward activations as W_yy@y_{t-1} + W_yx@x
            self.areas[to_area_index].activations = feedforward_activations + \
                recurrent_activations

            # Inhibition (capK)
            # thresholding the activations by allowing only topk neurons
            winners = capk(
                self.areas[to_area_index].activations, self.areas[to_area_index].k)
            self.areas[to_area_index].activations = winners

            # Hebbian Updates
            # feedforward connection from from-area to to-area
            self.feedforward_connections[iweights] = hebbian_update(input_activation,
                                                                    self.areas[to_area_index].activations,
                                                                    self.feedforward_connections[iweights],
                                                                    self.beta)
            # recurrent connection in to-area
            self.areas[to_area_index].recurrent_connections = hebbian_update(prev_winners,
                                                                             self.areas[to_area_index].activations,
                                                                             self.areas[to_area_index].recurrent_connections,
                                                                             self.beta)

            # if asked, do SVD and compute top-k singular value ratio
            if return_stable_rank:
                u, s, vh = np.linalg.svd(
                    self.feedforward_connections[iweights])
                # calculate the sum of top-k singular values, and its ratio wrt sum of all singular values
                stable_rank_ratio.append(
                    np.sum(s[:self.areas[to_area_index].k])/np.sum(s))

            # update neurons touched
            if track_neuron:
                touched_neurons = touched_neurons.union(
                    np.where(winners != 0)[0])
                touched_neurons_size.append(len(touched_neurons))

            # compare current winners with previous winners, and store current winners
            prev_winners = np.copy(winners)
            if verbose and (t == max_iterations - 1):
                print("\tExhausted all iterations.")

        # return connection weights (feedforward & recurrent) and assembly activations
        if return_weights_assembly:
            return self.feedforward_connections[iweights], \
                self.areas[to_area_index].recurrent_connections,\
                self.areas[to_area_index].activations

        if not only_once:
            # wipe activations
            self.areas[from_area_index].wipe_activations()
            self.areas[to_area_index].wipe_activations()

        # inhibit to-area neurons
        self.areas[to_area_index].inhibit()

        if return_stable_rank:  # return ratio of topk s.v. over all s.v. in each iteration
            return stable_rank_ratio

        return touched_neurons_size if track_neuron else None

    def associate(self):
        """
        If two assemblies in two different areas have been independently `projected`
        in a third area to form assemblies x and y, and, subsequently, 
        the two parent assemblies fire simultaneously, then each of x,y 
        will respond by having some of its neurons migrate to the other assembly; 
        this is called association of x and y.
        """
        pass

    def project_with_change_in_recurrence(self, input_activation, from_area_index, to_area_index,
                                          max_iterations=50, verbose=0,
                                          return_stable_rank=False,
                                          return_weights_assembly=False,
                                          only_once=False, with_recurrence=True):
        '''
        Fire a fix set of neurons in Area from_area_index onto Area to_area_index.
        Input_activation is a set of activated neurons (binary) in from_area
        '''
        # disinhibit the to-area Area so it does recurrent/feedforward activation
        self.areas[to_area_index].disinhibit()

        # find the index storing feedforward connection matrix
        iweights = find_feedforward_matrix_index(
            self.area_combinations, from_area_index, to_area_index)

        if not only_once:
            # randomly sample initial feedforward connections
            self.feedforward_connections[iweights] = self.sample_initial_connections(
            )

            # randomly sample initial self-recurrent connections
            if with_recurrence:
                self.areas[to_area_index].recurrent_connections = self.areas[to_area_index].sample_initial_connections()

        # track indices and total numbers of neurons activated so far in each iteration
        touched_neurons = set()
        touched_neurons_size = []

        # record winners for the previous iteration
        prev_winners = self.areas[to_area_index].activations

        # initialize stable rank ratio
        stable_rank_ratio = []

        # project from from-area to to-area repeatedly, until convergence or max_iterations reached
        for t in range(max_iterations):
            if verbose and (t != 0) and (t % 50 == 0):
                print("\titeration", t)
            if only_once:
                assert t < 1

            # send input activations from from-area to to-area through opened connections
            feedforward_activations = self.feedforward_connections[iweights].dot(
                input_activation)

            # do self-recurrent activation in to-area (the first self-recurrent does not have any effect because activations are 0s)
            if with_recurrence:
                recurrent_activations = self.areas[to_area_index].recurrent_connections.dot(
                    prev_winners)

            # and add the activations to the feedforward activations as W_yy@y_{t-1} + W_yx@x
            if with_recurrence:
                self.areas[to_area_index].activations = feedforward_activations + \
                    recurrent_activations
            else:
                self.areas[to_area_index].activations = feedforward_activations

            # Inhibition (capK)
            # thresholding the activations by allowing only topk neurons
            winners = capk(
                self.areas[to_area_index].activations, self.areas[to_area_index].k)
            self.areas[to_area_index].activations = winners

            # Hebbian Updates
            # feedforward connection from from-area to to-area
            self.feedforward_connections[iweights] = hebbian_update(input_activation,
                                                                    self.areas[to_area_index].activations,
                                                                    self.feedforward_connections[iweights],
                                                                    self.beta)
            # recurrent connection in to-area
            if with_recurrence:
                self.areas[to_area_index].recurrent_connections = hebbian_update(prev_winners,
                                                                                 self.areas[to_area_index].activations,
                                                                                 self.areas[to_area_index].recurrent_connections,
                                                                                 self.beta)

            # if asked, do SVD and compute top-k singular value ratio
            if return_stable_rank:
                u, s, vh = np.linalg.svd(
                    self.feedforward_connections[iweights])
                # calculate the sum of top-k singular values, and its ratio wrt sum of all singular values
                stable_rank_ratio.append(
                    np.sum(s[:self.areas[to_area_index].k])/np.sum(s))

            # update neurons touched
            touched_neurons = touched_neurons.union(np.where(winners != 0)[0])
            touched_neurons_size.append(len(touched_neurons))

            # compare current winners with previous winners, and store current winners
            prev_winners = np.copy(winners)
            if verbose and (t == max_iterations - 1):
                print("\tExhausted all iterations.")

        # return connection weights (feedforward & recurrent) and assembly activations
        if return_weights_assembly and with_recurrence:
            return self.feedforward_connections[iweights], \
                self.areas[to_area_index].recurrent_connections,\
                self.areas[to_area_index].activations

        if return_weights_assembly and not with_recurrence:
            return self.feedforward_connections[iweights], \
                None, \
                self.areas[to_area_index].activations

        if not only_once:
            # wipe activations
            self.areas[from_area_index].wipe_activations()
            self.areas[to_area_index].wipe_activations()

        # inhibit to-area neurons
        self.areas[to_area_index].inhibit()

        if return_stable_rank:  # return ratio of topk s.v. over all s.v. in each iteration
            return stable_rank_ratio

        return touched_neurons_size

    def reciprocal_project(self, input_activation, area1_index, area2_index, area3_index,
                           max_iterations=100, verbose=0,
                           return_stable_rank=False,
                           return_weights_assembly=False,
                           only_once=False, new_winner=False):
        '''
        We perform Reciprocal Projection: 
            Area1 (stimulus x) -> Area2 <--> Area3
        Project a fix set of neurons in area 1,
        area2 receives both feedforward activations from area1 and area3,
        area3 receives feedforward activations from area2,
        area2 and area3 both have recurrence connections.
        area2(t+1) = W1to2 * area1(t) + W2to2 * area2(t) + W3to2 * area3(t)
            update W1to2, W2to2, W3to2
        area3(t+1) = W2to3 * area2(t) + W3to3 * area3(t)
            update W2to3, W3to3
        '''
        # disinhibit areas
        self.areas[area2_index].disinhibit()
        self.areas[area3_index].disinhibit()

        # find the index storing feedforward and feedback connection matrix
        iweights_1to2 = find_feedforward_matrix_index(
            self.area_combinations, area1_index, area2_index)  # feedforward
        iweights_2to3 = find_feedforward_matrix_index(
            self.area_combinations, area2_index, area3_index)  # feedforward
        iweights_3to2 = find_feedforward_matrix_index(
            self.area_combinations, area3_index, area2_index)  # feedbackward

        if not only_once:
            # randomly initialize feedforward and recurrent connections
            self.feedforward_connections[iweights_1to2] = self.sample_initial_connections(
            )
            self.areas[area2_index].recurrent_connections = self.areas[area2_index].sample_initial_connections()
            self.feedforward_connections[iweights_2to3] = self.sample_initial_connections(
            )
            self.areas[area3_index].recurrent_connections = self.areas[area3_index].sample_initial_connections()
            self.feedforward_connections[iweights_3to2] = self.sample_initial_connections(
            )

        # track indices and total numbers of neurons activated so far in each iteration
        touched_neurons_2 = set()
        touched_neurons_3 = set()
        touched_neurons_size_2 = []
        touched_neurons_size_3 = []

        # record winner neurons for the previous iteration
        prev_winners_area2 = self.areas[area2_index].activations
        prev_winners_area3 = self.areas[area3_index].activations

        # store ratio of stable rank
        stable_rank_ratio_1to2 = []
        stable_rank_ratio_2to3 = []
        stable_rank_ratio_3to2 = []

        for t in range(max_iterations):
            if verbose and (t != 0) and (t % 50 == 0):
                print("\titeration", t)
            if only_once:
                assert t < 1

            # forward pass: area1 --> area2 <-- area3 (feedforward from 1 and from 3 + recurrent in 2)
            # feedforward 1 to 2
            feedforward_activations_1to2 = self.feedforward_connections[iweights_1to2].dot(
                input_activation)
            # self-recurrent activation in area2 (the first self-recurrent does not have any effect because activations are 0s)
            recurrent_activations_2 = self.areas[area2_index].recurrent_connections.dot(
                prev_winners_area2)
            # feedforward 3 to 2
            feedforward_activations_3to2 = self.feedforward_connections[iweights_3to2].dot(
                prev_winners_area3)
            # add the activations to the feedforward activations
            self.areas[area2_index].activations = feedforward_activations_1to2 + \
                recurrent_activations_2 + feedforward_activations_3to2

            # thresholding the activations by allowing only topk neurons
            winners_area2 = capk(
                self.areas[area2_index].activations, self.areas[area2_index].k)
            self.areas[area2_index].activations = winners_area2

            # hebbian update: feedforward connection area1 to 2
            self.feedforward_connections[iweights_1to2] = hebbian_update(input_activation,
                                                                         self.areas[area2_index].activations,
                                                                         self.feedforward_connections[iweights_1to2],
                                                                         self.beta)
            # hebbian update: feedforward connection area3 to 2
            self.feedforward_connections[iweights_3to2] = hebbian_update(prev_winners_area3,
                                                                         self.areas[area2_index].activations,
                                                                         self.feedforward_connections[iweights_3to2],
                                                                         self.beta)
            # hebbian update: recurrent connection in area2
            self.areas[area2_index].recurrent_connections = hebbian_update(prev_winners_area2,
                                                                           self.areas[area2_index].activations,
                                                                           self.areas[area2_index].recurrent_connections,
                                                                           self.beta)
            # update neurons touched
            touched_neurons_2 = touched_neurons_2.union(
                np.where(winners_area2 != 0)[0])
            touched_neurons_size_2.append(len(touched_neurons_2))

            # forward pass: area2 --> area3 (feedforward from 2 to 3 + recurrent in 3)
            if not new_winner:
                # use prev winner in area2
                feedforward_activations_2to3 = self.feedforward_connections[iweights_2to3].dot(
                    prev_winners_area2)
            else:
                # use updated winner in area2
                feedforward_activations_2to3 = self.feedforward_connections[iweights_2to3].dot(
                    winners_area2)

            # self-recurrent activation in area3 (the first self-recurrent does not have any effect because activations are 0s)
            recurrent_activations_3 = self.areas[area3_index].recurrent_connections.dot(
                prev_winners_area3)
            # add the activations to the feedforward activations
            self.areas[area3_index].activations = feedforward_activations_2to3 + \
                recurrent_activations_3
            # thresholding the activations by allowing only topk neurons
            winners_area3 = capk(
                self.areas[area3_index].activations, self.areas[area3_index].k)

            self.areas[area3_index].activations = winners_area3
            # hebbian update: feedforward weight area2 to 3
            self.feedforward_connections[iweights_2to3] = hebbian_update(prev_winners_area2,
                                                                         self.areas[area3_index].activations,
                                                                         self.feedforward_connections[iweights_2to3],
                                                                         self.beta)
            # hebbian update: recurrent connection in 3
            self.areas[area3_index].recurrent_connections = hebbian_update(prev_winners_area3,
                                                                           self.areas[area3_index].activations,
                                                                           self.areas[area3_index].recurrent_connections,
                                                                           self.beta)
            # update neurons touched
            touched_neurons_3 = touched_neurons_3.union(
                np.where(winners_area3 != 0)[0])
            touched_neurons_size_3.append(len(touched_neurons_3))

            # if asked, do SVD and calculate top-k singular value ratio
            if return_stable_rank:
                # single value decomposition
                u1to2, s1to2, v1to2 = np.linalg.svd(
                    self.feedforward_connections[iweights_1to2])
                u2to3, s2to3, v2to3 = np.linalg.svd(
                    self.feedforward_connections[iweights_2to3])
                u3to2, s3to2, v3to2 = np.linalg.svd(
                    self.feedforward_connections[iweights_3to2])
                # calculate the sum of top-k singular values, and its ratio wrt sum of all singular values
                stable_rank_ratio_1to2.append(
                    np.sum(s1to2[:self.areas[area2_index].topk])/np.sum(s1to2))
                stable_rank_ratio_2to3.append(
                    np.sum(s2to3[:self.areas[area3_index].topk])/np.sum(s2to3))
                stable_rank_ratio_3to2.append(
                    np.sum(s3to2[:self.areas[area2_index].topk])/np.sum(s3to2))

            # update prev winners
            prev_winners_area2 = np.copy(winners_area2)
            prev_winners_area3 = np.copy(winners_area3)

            if verbose and (t == max_iterations - 1):
                print("\tExhausted all iterations.")

        if return_weights_assembly:  # return connection weights and assembly activations
            return self.feedforward_connections[iweights_1to2], \
                self.feedforward_connections[iweights_2to3], \
                self.feedforward_connections[iweights_3to2], \
                self.areas[area2_index].recurrent_connections,\
                self.areas[area3_index].recurrent_connections,\
                self.areas[area2_index].activations,\
                self.areas[area3_index].activations,

        if not only_once:
            # wipe activations
            self.wipe_all_activations()

        # inhibit neurons
        self.areas[area2_index].inhibit()
        self.areas[area3_index].inhibit()

        if return_stable_rank:
            return stable_rank_ratio_1to2, stable_rank_ratio_2to3, stable_rank_ratio_3to2
        return touched_neurons_size_2, touched_neurons_size_3

    def residual_reciprocal_project(self, input_activation, area1_index, area2_index, area3_index,
                                    max_iterations=100, verbose=0,
                                    return_stable_rank=False,
                                    return_weights_assembly=False,
                                    only_once=False, new_winner=False):
        '''
        We perform Reciprocal Projection: 
            Area1 (stimulus x) -> Area2 <--> Area3
        Project a fix set of neurons in area 1,
        area2 receives both feedforward activations from area1 and area3,
        area3 receives feedforward activations from area2,
        area2 and area3 both have recurrence connections.
        area2(t+1) = W1to2 * area1(t) + W2to2 * area2(t) + W3to2 * area3(t)
            update W1to2, W2to2, W3to2
        area3(t+1) = W2to3 * area2(t) + W3to3 * area3(t)
            update W2to3, W3to3
        '''
        # disinhibit areas
        self.areas[area2_index].disinhibit()
        self.areas[area3_index].disinhibit()

        # find the index storing feedforward and feedback connection matrix
        iweights_1to2 = find_feedforward_matrix_index(
            self.area_combinations, area1_index, area2_index)  # feedforward
        iweights_2to3 = find_feedforward_matrix_index(
            self.area_combinations, area2_index, area3_index)  # feedforward
        iweights_3to2 = find_feedforward_matrix_index(
            self.area_combinations, area3_index, area2_index)  # feedbackward

        # residual connection
        iweights_1to3 = find_feedforward_matrix_index(
            self.area_combinations, area1_index, area2_index)  # feedforward

        if not only_once:
            # randomly initialize feedforward and recurrent connections
            # area 1 (stimuli)
            self.feedforward_connections[iweights_1to2] = self.sample_initial_connections(
            )
            self.feedforward_connections[iweights_1to3] = self.sample_initial_connections(
            )
            # area 2
            self.areas[area2_index].recurrent_connections = self.areas[area2_index].sample_initial_connections()
            self.feedforward_connections[iweights_2to3] = self.sample_initial_connections(
            )
            # area 3
            self.areas[area3_index].recurrent_connections = self.areas[area3_index].sample_initial_connections()
            self.feedforward_connections[iweights_3to2] = self.sample_initial_connections(
            )

        # track indices and total numbers of neurons activated so far in each iteration
        touched_neurons_2 = set()
        touched_neurons_3 = set()
        touched_neurons_size_2 = []
        touched_neurons_size_3 = []

        # record winner neurons for the previous iteration
        prev_winners_area2 = self.areas[area2_index].activations
        prev_winners_area3 = self.areas[area3_index].activations

        # store ratio of stable rank
        stable_rank_ratio_1to2 = []
        stable_rank_ratio_2to3 = []
        stable_rank_ratio_3to2 = []

        for t in range(max_iterations):
            if verbose and (t != 0) and (t % 50 == 0):
                print("\titeration", t)
            if only_once:
                assert t < 1

            # forward pass: area1 --> area2 <--> area3 (feedforward from 1 and from 3 + recurrent in 2)
            # feedforward 1 to 2
            feedforward_activations_1to2 = self.feedforward_connections[iweights_1to2].dot(
                input_activation)
            # self-recurrent activation in area2 (the first self-recurrent does not have any effect because activations are 0s)
            recurrent_activations_2 = self.areas[area2_index].recurrent_connections.dot(
                prev_winners_area2)
            # feedforward 3 to 2
            feedforward_activations_3to2 = self.feedforward_connections[iweights_3to2].dot(
                prev_winners_area3)
            # add the activations to the feedforward activations
            self.areas[area2_index].activations = feedforward_activations_1to2 + \
                recurrent_activations_2 + feedforward_activations_3to2

            # thresholding the activations by allowing only topk neurons
            winners_area2 = capk(
                self.areas[area2_index].activations, self.areas[area2_index].k)
            self.areas[area2_index].activations = winners_area2

            # hebbian update: feedforward connection area1 to 2
            self.feedforward_connections[iweights_1to2] = hebbian_update(input_activation,
                                                                         self.areas[area2_index].activations,
                                                                         self.feedforward_connections[iweights_1to2],
                                                                         self.beta)
            # hebbian update: feedforward connection area3 to 2
            self.feedforward_connections[iweights_3to2] = hebbian_update(prev_winners_area3,
                                                                         self.areas[area2_index].activations,
                                                                         self.feedforward_connections[iweights_3to2],
                                                                         self.beta)
            # hebbian update: recurrent connection in area2
            self.areas[area2_index].recurrent_connections = hebbian_update(prev_winners_area2,
                                                                           self.areas[area2_index].activations,
                                                                           self.areas[area2_index].recurrent_connections,
                                                                           self.beta)
            # update neurons touched
            touched_neurons_2 = touched_neurons_2.union(
                np.where(winners_area2 != 0)[0])
            touched_neurons_size_2.append(len(touched_neurons_2))

            # forward pass: area2 --> area3 (feedforward from 2 to 3 + recurrent in 3)
            if not new_winner:
                # use prev winner in area2
                feedforward_activations_2to3 = self.feedforward_connections[iweights_2to3].dot(
                    prev_winners_area2)
            else:
                # use updated winner in area2
                feedforward_activations_2to3 = self.feedforward_connections[iweights_2to3].dot(
                    winners_area2)

            # **redidual activation**
            feedforward_activations_1to3 = self.feedforward_connections[iweights_1to3].dot(
                input_activation)

            # self-recurrent activation in area3 (the first self-recurrent does not have any effect because activations are 0s)
            recurrent_activations_3 = self.areas[area3_index].recurrent_connections.dot(
                prev_winners_area3)
            # **add the activations (and residual activation) to the feedforward activations**
            self.areas[area3_index].activations = feedforward_activations_1to3 + feedforward_activations_2to3 + \
                recurrent_activations_3
            # thresholding the activations by allowing only topk neurons
            winners_area3 = capk(
                self.areas[area3_index].activations, self.areas[area3_index].k)

            self.areas[area3_index].activations = winners_area3
            # **hebbian update: feedforward weight area1 to 3
            self.feedforward_connections[iweights_1to3] = hebbian_update(input_activation,
                                                                         self.areas[area3_index].activations,
                                                                         self.feedforward_connections[iweights_1to3],
                                                                         self.beta)
            # hebbian update: feedforward weight area2 to 3
            self.feedforward_connections[iweights_2to3] = hebbian_update(prev_winners_area2,
                                                                         self.areas[area3_index].activations,
                                                                         self.feedforward_connections[iweights_2to3],
                                                                         self.beta)
            # hebbian update: recurrent connection in 3
            self.areas[area3_index].recurrent_connections = hebbian_update(prev_winners_area3,
                                                                           self.areas[area3_index].activations,
                                                                           self.areas[area3_index].recurrent_connections,
                                                                           self.beta)
            # update neurons touched
            touched_neurons_3 = touched_neurons_3.union(
                np.where(winners_area3 != 0)[0])
            touched_neurons_size_3.append(len(touched_neurons_3))

            # if asked, do SVD and calculate top-k singular value ratio
            if return_stable_rank:
                # single value decomposition
                u1to2, s1to2, v1to2 = np.linalg.svd(
                    self.feedforward_connections[iweights_1to2])
                u2to3, s2to3, v2to3 = np.linalg.svd(
                    self.feedforward_connections[iweights_2to3])
                u3to2, s3to2, v3to2 = np.linalg.svd(
                    self.feedforward_connections[iweights_3to2])
                # calculate the sum of top-k singular values, and its ratio wrt sum of all singular values
                stable_rank_ratio_1to2.append(
                    np.sum(s1to2[:self.areas[area2_index].topk])/np.sum(s1to2))
                stable_rank_ratio_2to3.append(
                    np.sum(s2to3[:self.areas[area3_index].topk])/np.sum(s2to3))
                stable_rank_ratio_3to2.append(
                    np.sum(s3to2[:self.areas[area2_index].topk])/np.sum(s3to2))

            # update prev winners
            prev_winners_area2 = np.copy(winners_area2)
            prev_winners_area3 = np.copy(winners_area3)

            if verbose and (t == max_iterations - 1):
                print("\tExhausted all iterations.")

        if return_weights_assembly:  # **return connection weights and assembly activations
            return self.feedforward_connections[iweights_1to2], \
                self.feedforward_connections[iweights_2to3], \
                self.feedforward_connections[iweights_3to2], \
                self.feedforward_connections[iweights_1to3], \
                self.areas[area2_index].recurrent_connections,\
                self.areas[area3_index].recurrent_connections,\
                self.areas[area2_index].activations,\
                self.areas[area3_index].activations,

        if not only_once:
            # wipe activations
            self.wipe_all_activations()

        # inhibit neurons
        self.areas[area2_index].inhibit()
        self.areas[area3_index].inhibit()

        if return_stable_rank:
            return stable_rank_ratio_1to2, stable_rank_ratio_2to3, stable_rank_ratio_3to2
        return touched_neurons_size_2, touched_neurons_size_3

    def modified_reciprocal_project(self, input_activation, area1_index, area2_index, area3_index,
                                    max_iterations=100, verbose=0,
                                    return_stable_rank=False,
                                    return_weights_assembly=False,
                                    only_once=False):
        '''
        We perform Reciprocal Projection: 
            Area2 <--> Area3
        Project a fix set of neurons in area 1,
        area2 receives both feedforward activations from area1 and area3,
        area3 receives feedforward activations from area2,
        area2 and area3 both have recurrence connections.
        area2(t+1) = W1to2 * area1(t) + W2to2 * area2(t) + W3to2 * area3(t)
            update W1to2, W2to2, W3to2
        area3(t+1) = W2to3 * area2(t) + W3to3 * area3(t)
            update W2to3, W3to3
        '''
        # disinhibit areas
        self.areas[area2_index].disinhibit()
        self.areas[area3_index].disinhibit()

        # find the index storing feedforward and feedback connection matrix
        iweights_1to2 = find_feedforward_matrix_index(
            self.area_combinations, area1_index, area2_index)  # feedforward
        iweights_2to3 = find_feedforward_matrix_index(
            self.area_combinations, area2_index, area3_index)  # feedforward
        iweights_3to2 = find_feedforward_matrix_index(
            self.area_combinations, area3_index, area2_index)  # feedbackward

        if not only_once:
            # randomly initialize feedforward and recurrent connections
            self.feedforward_connections[iweights_1to2] = self.sample_initial_connections(
            )
            self.areas[area2_index].recurrent_connections = self.areas[area2_index].sample_initial_connections()
            self.feedforward_connections[iweights_2to3] = self.sample_initial_connections(
            )
            self.areas[area3_index].recurrent_connections = self.areas[area3_index].sample_initial_connections()
            self.feedforward_connections[iweights_3to2] = self.sample_initial_connections(
            )

        # track indices and total numbers of neurons activated so far in each iteration
        touched_neurons_2 = set()
        touched_neurons_3 = set()
        touched_neurons_size_2 = []
        touched_neurons_size_3 = []

        # record winner neurons for the previous iteration
        prev_winners_area2 = self.areas[area2_index].activations
        prev_winners_area3 = self.areas[area3_index].activations

        # store ratio of stable rank
        stable_rank_ratio_1to2 = []
        stable_rank_ratio_2to3 = []
        stable_rank_ratio_3to2 = []

        for t in range(max_iterations):
            if verbose and (t != 0) and (t % 50 == 0):
                print("\titeration", t)
            if only_once:
                assert t < 1

            # forward pass: area1 --> area2 <-- area3 (feedforward from 1 and from 3 + recurrent in 2)
            # feedforward 1 to 2
            # feedforward_activations_1to2 = self.feedforward_connections[iweights_1to2].dot(
            #     input_activation)
            # self-recurrent activation in area2 (the first self-recurrent does not have any effect because activations are 0s)
            recurrent_activations_2 = self.areas[area2_index].recurrent_connections.dot(
                prev_winners_area2)
            # feedforward 3 to 2
            feedforward_activations_3to2 = self.feedforward_connections[iweights_3to2].dot(
                prev_winners_area3)
            # add the activations to the feedforward activations
            self.areas[area2_index].activations = recurrent_activations_2 + \
                feedforward_activations_3to2

            # thresholding the activations by allowing only topk neurons
            winners_area2 = capk(
                self.areas[area2_index].activations, self.areas[area2_index].k)
            self.areas[area2_index].activations = winners_area2

            # hebbian update: feedforward connection area1 to 2
            # self.feedforward_connections[iweights_1to2] = hebbian_update(input_activation,
            #                                                              self.areas[area2_index].activations,
            #                                                              self.feedforward_connections[iweights_1to2],
            #                                                              self.beta)
            # hebbian update: feedforward connection area2 to 3
            self.feedforward_connections[iweights_3to2] = hebbian_update(prev_winners_area3,
                                                                         self.areas[area2_index].activations,
                                                                         self.feedforward_connections[iweights_3to2],
                                                                         self.beta)
            # hebbian update: recurrent connection in area2
            self.areas[area2_index].recurrent_connections = hebbian_update(prev_winners_area2,
                                                                           self.areas[area2_index].activations,
                                                                           self.areas[area2_index].recurrent_connections,
                                                                           self.beta)
            # update neurons touched
            touched_neurons_2 = touched_neurons_2.union(
                np.where(winners_area2 != 0)[0])
            touched_neurons_size_2.append(len(touched_neurons_2))

            # forward pass: area2 --> area3 (feedforward from 2 to 3 + recurrent in 3)
            feedforward_activations_2to3 = self.feedforward_connections[iweights_2to3].dot(
                prev_winners_area2)
            # self-recurrent activation in area3 (the first self-recurrent does not have any effect because activations are 0s)
            recurrent_activations_3 = self.areas[area3_index].recurrent_connections.dot(
                prev_winners_area3)
            # add the activations to the feedforward activations
            self.areas[area3_index].activations = feedforward_activations_2to3 + \
                recurrent_activations_3
            # thresholding the activations by allowing only topk neurons
            winners_area3 = capk(
                self.areas[area3_index].activations, self.areas[area3_index].k)

            self.areas[area3_index].activations = winners_area3
            # hebbian update: feedforward weight area2 to 3
            self.feedforward_connections[iweights_2to3] = hebbian_update(prev_winners_area2,
                                                                         self.areas[area3_index].activations,
                                                                         self.feedforward_connections[iweights_2to3],
                                                                         self.beta)
            # hebbian update: recurrent connection in 3
            self.areas[area3_index].recurrent_connections = hebbian_update(prev_winners_area3,
                                                                           self.areas[area3_index].activations,
                                                                           self.areas[area3_index].recurrent_connections,
                                                                           self.beta)
            # update neurons touched
            touched_neurons_3 = touched_neurons_3.union(
                np.where(winners_area3 != 0)[0])
            touched_neurons_size_3.append(len(touched_neurons_3))

            # if asked, do SVD and calculate top-k singular value ratio
            if return_stable_rank:
                # single value decomposition
                u1to2, s1to2, v1to2 = np.linalg.svd(
                    self.feedforward_connections[iweights_1to2])
                u2to3, s2to3, v2to3 = np.linalg.svd(
                    self.feedforward_connections[iweights_2to3])
                u3to2, s3to2, v3to2 = np.linalg.svd(
                    self.feedforward_connections[iweights_3to2])
                # calculate the sum of top-k singular values, and its ratio wrt sum of all singular values
                stable_rank_ratio_1to2.append(
                    np.sum(s1to2[:self.areas[area2_index].topk])/np.sum(s1to2))
                stable_rank_ratio_2to3.append(
                    np.sum(s2to3[:self.areas[area3_index].topk])/np.sum(s2to3))
                stable_rank_ratio_3to2.append(
                    np.sum(s3to2[:self.areas[area2_index].topk])/np.sum(s3to2))

            # update prev winners
            prev_winners_area2 = np.copy(winners_area2)
            prev_winners_area3 = np.copy(winners_area3)

            if verbose and (t == max_iterations - 1):
                print("\tExhausted all iterations.")

        if return_weights_assembly:  # return connection weights and assembly activations
            return self.feedforward_connections[iweights_1to2], \
                self.feedforward_connections[iweights_2to3], \
                self.feedforward_connections[iweights_3to2], \
                self.areas[area2_index].recurrent_connections,\
                self.areas[area3_index].recurrent_connections,\
                self.areas[area2_index].activations,\
                self.areas[area3_index].activations,

        if not only_once:
            # wipe activations
            self.wipe_all_activations()

        # inhibit neurons
        self.areas[area2_index].inhibit()
        self.areas[area3_index].inhibit()

        if return_stable_rank:
            return stable_rank_ratio_1to2, stable_rank_ratio_2to3, stable_rank_ratio_3to2
        return touched_neurons_size_2, touched_neurons_size_3


if __name__ == "__main__":
    # brain = Brain(num_areas=10)
    # print(brain.area_combinations.shape)
    # print(len(brain.feedforward_connections))
    print(np.random.binomial(10, 0.1, size=100) * 1.0)

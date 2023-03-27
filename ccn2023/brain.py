import numpy as np
from area import *
from collections import defaultdict

from utils import *

#TODO: Write a method to obtain assemblies from a set of stimuli

class Brain:
    def __init__(self, num_areas=2, num_neurons=1000, cap_size=41, p=0.03, beta=0.05):
        self.num_areas = num_areas
        self.num_neurons = num_neurons
        self.afferent_p = p
        self.cap_size = cap_size
        self.beta = beta

        self.areas = [Area(num_neurons=num_neurons, cap_size=cap_size, p=p, beta=beta) for _ in range(self.num_areas)]
        self.area_combinations = np.array(np.where(np.eye(self.num_areas, dtype=bool))).T
        self.afferent_connections = defaultdict(lambda : np.empty((self.num_neurons, self.num_neurons)))

    def initialize_afferent_connections(self, area_combination):
        self.afferent_connections[area_combination] = np.random.binomial(1,self.afferent_p,size=(self.num_neurons,self.num_neurons)).astype("float64")

    def normalize_afferent_connections(self, area_combination):
        self.afferent_connections[area_combination] /= self.afferent_connections[area_combination].sum(axis=1)[:,np.newaxis]

    def get_afferent_connections(self, area_combinations):
        out = {}
        for k in area_combinations:
            if k in self.afferent_connections:
                out[k] = np.copy(self.afferent_connections[k])
            else:
                out[k] = -1
        return out

    def set_afferent_connections(self, connections_dict):
        for k in connections_dict.keys():
            if connections_dict[k].shape == (self.num_neurons, self.num_neurons):
                self.afferent_connections[k] = np.copy(connections_dict[k])
            else:
                print("matrix provided for area combination {} is of shape {}, expected shape {}".format(k, connections_dict[k].shape, (self.num_neurons, self.num_neurons)))

    def project(self, input_activation, from_area_idx, to_area_idx, max_iterations=50, use_recurrent=True, normalize=True, normalize_interval=5, initialize=True, return_weights_assembly=False):
        if initialize:
            self.initialize_afferent_connections((from_area_idx, to_area_idx))
        if use_recurrent and initialize:
            self.areas[to_area_idx].initialize_recurrent_connections()

        self.areas[to_area_idx].clear_activations()
        self.areas[from_area_idx].set_activations(input_activation)
        

        total_support = set()
        total_support_size = []

        for t in range(max_iterations):
            total_input = self.afferent_connections[(from_area_idx, to_area_idx)].dot(self.areas[from_area_idx].activations)
            if use_recurrent:
                recurrent_input = self.areas[to_area_idx].recurrent_connections.dot(self.areas[to_area_idx].activations)
                total_input += recurrent_input

            new_activations = capk(total_input, self.cap_size)

            hebbian_update(self.afferent_connections[(from_area_idx, to_area_idx)], new_activations, self.areas[from_area_idx].activations, self.beta)
            if use_recurrent:
                hebbian_update(self.areas[to_area_idx].recurrent_connections, new_activations, self.areas[to_area_idx].activations, self.beta)
            
            if normalize and (t+1)%normalize_interval == 0:
                self.afferent_connections[(from_area_idx, to_area_idx)] /= self.afferent_connections[(from_area_idx, to_area_idx)].sum(axis=1)[:,np.newaxis]
                if use_recurrent:
                    self.areas[to_area_idx].normalize_weights()
            
            support = np.where(new_activations !=0)[0]
            total_support = total_support.union(support)
            total_support_size.append(len(total_support))

            self.areas[to_area_idx].set_activations(new_activations)

        if return_weights_assembly:
            if use_recurrent:
                return self.areas[to_area_idx].get_activations(), self.areas[to_area_idx].get_recurrent_connections(), self.get_afferent_connections([(from_area_idx, to_area_idx)])[(from_area_idx, to_area_idx)]
            else:
                return self.areas[to_area_idx].get_activations(), self.get_afferent_connections([(from_area_idx, to_area_idx)])[(from_area_idx, to_area_idx)]

        return np.array(total_support_size)

    def project_stream(self, stimulus_generator, from_area_idx, to_area_idx, max_iterations=50, use_recurrent=True, normalize=True, normalize_interval=5, initialize=True, return_weights_assembly=False):
        if initialize:
            self.initialize_afferent_connections((from_area_idx, to_area_idx))
        if use_recurrent and initialize:
            self.areas[to_area_idx].initialize_recurrent_connections()

        self.areas[to_area_idx].clear_activations()

        total_support = set()
        total_support_size = []

        for t in range(max_iterations):
            self.areas[from_area_idx].set_activations(stimulus_generator.sample_stimulus())
            total_input = self.afferent_connections[(from_area_idx, to_area_idx)].dot(self.areas[from_area_idx].activations)
            if use_recurrent:
                recurrent_input = self.areas[to_area_idx].recurrent_connections.dot(self.areas[to_area_idx].activations)
                total_input += recurrent_input

            new_activations = capk(total_input, self.cap_size)

            hebbian_update(self.afferent_connections[(from_area_idx, to_area_idx)], new_activations, self.areas[from_area_idx].activations, self.beta)
            if use_recurrent:
                hebbian_update(self.areas[to_area_idx].recurrent_connections, new_activations, self.areas[to_area_idx].activations, self.beta)

            if normalize and (t+1)%normalize_interval == 0:
                self.afferent_connections[(from_area_idx, to_area_idx)] /= self.afferent_connections[(from_area_idx, to_area_idx)].sum(axis=1)[:,np.newaxis]
                if use_recurrent:
                    self.areas[to_area_idx].normalize_weights()

            support = np.where(new_activations !=0)[0]
            total_support = total_support.union(support)
            total_support_size.append(len(total_support))

            self.areas[to_area_idx].set_activations(new_activations)

        if return_weights_assembly:
            if use_recurrent:
                return self.areas[to_area_idx].get_activations(), self.areas[to_area_idx].get_recurrent_connections(), self.get_afferent_connections([(from_area_idx, to_area_idx)])[(from_area_idx, to_area_idx)]
            else:
                return self.areas[to_area_idx].get_activations(), self.get_afferent_connections([(from_area_idx, to_area_idx)])[(from_area_idx, to_area_idx)]

        return np.array(total_support_size)

    def reciprocal_project_stream(self, stimulus_generator, area1_idx, area2_idx, area3_idx, max_iterations=50, use_recurrent=True, normalize=True, normalize_interval=5, initialize=True, return_weights_assembly=False):
        '''
        Yichen edited in March, 2023
        Original reciprocal project
        X --> Y <--> Z
        '''
        if initialize:
            self.initialize_afferent_connections((area1_idx, area2_idx))
            self.initialize_afferent_connections((area2_idx, area3_idx))
            self.initialize_afferent_connections((area3_idx, area2_idx))
        if use_recurrent and initialize:
            self.areas[area2_idx].initialize_recurrent_connections()
            self.areas[area3_idx].initialize_recurrent_connections()

        self.areas[area2_idx].clear_activations()
        self.areas[area3_idx].clear_activations()

        total_support_area2 = set()
        total_support_size_area2 = []
        total_support_area3 = set()
        total_support_size_area3 = []

        for t in range(max_iterations):
            self.areas[area1_idx].set_activations(stimulus_generator.sample_stimulus())
            area2_inputs = self.afferent_connections[(area1_idx, area2_idx)].dot(self.areas[area1_idx].activations)\
                            + self.afferent_connections[(area3_idx, area2_idx)].dot(self.areas[area3_idx].activations)
            area3_inputs = self.afferent_connections[(area2_idx, area3_idx)].dot(self.areas[area2_idx].activations)
            if use_recurrent:
                area2_inputs += self.areas[area2_idx].recurrent_connections.dot(self.areas[area2_idx].activations)
                area3_inputs += self.areas[area3_idx].recurrent_connections.dot(self.areas[area3_idx].activations)
                
            new_activations_area2 = capk(area2_inputs, self.cap_size)
            new_activations_area3 = capk(area3_inputs, self.cap_size)

            hebbian_update(self.afferent_connections[(area1_idx, area2_idx)], new_activations_area2, self.areas[area1_idx].activations, self.beta)
            hebbian_update(self.afferent_connections[(area3_idx, area2_idx)], new_activations_area2, self.areas[area3_idx].activations, self.beta)
            hebbian_update(self.afferent_connections[(area2_idx, area3_idx)], new_activations_area3, self.areas[area2_idx].activations, self.beta)

            if use_recurrent:
                hebbian_update(self.areas[area2_idx].recurrent_connections, new_activations_area2, self.areas[area2_idx].activations, self.beta)
                hebbian_update(self.areas[area3_idx].recurrent_connections, new_activations_area3, self.areas[area3_idx].activations, self.beta)

            if normalize and (t+1)%normalize_interval == 0:
                self.afferent_connections[(area1_idx, area2_idx)] /= self.afferent_connections[(area1_idx, area2_idx)].sum(axis=1)[:,np.newaxis]
                self.afferent_connections[(area3_idx, area2_idx)] /= self.afferent_connections[(area3_idx, area2_idx)].sum(axis=1)[:,np.newaxis]
                self.afferent_connections[(area2_idx, area3_idx)] /= self.afferent_connections[(area2_idx, area3_idx)].sum(axis=1)[:,np.newaxis]
                if use_recurrent:
                    self.areas[area2_idx].normalize_weights()
                    self.areas[area3_idx].normalize_weights()

            support_area2 = np.where(new_activations_area2 !=0)[0]
            total_support_area2 = total_support_area2.union(support_area2)
            total_support_size_area2.append(len(total_support_area2))
            support_area3 = np.where(new_activations_area3 !=0)[0]
            total_support_area3 = total_support_area3.union(support_area3)
            total_support_size_area3.append(len(total_support_area3))

            self.areas[area2_idx].set_activations(new_activations_area2)
            self.areas[area3_idx].set_activations(new_activations_area3)

        if return_weights_assembly:
            if use_recurrent:
                return self.areas[area2_idx].get_activations(), \
                        self.areas[area3_idx].get_activations(),\
                        self.areas[area2_idx].get_recurrent_connections(), \
                        self.areas[area3_idx].get_recurrent_connections(), \
                        self.get_afferent_connections([(area1_idx, area2_idx)])[(area1_idx, area2_idx)], \
                        self.get_afferent_connections([(area3_idx, area2_idx)])[(area3_idx, area2_idx)],\
                        self.get_afferent_connections([(area2_idx, area3_idx)])[(area2_idx, area3_idx)]

            else:
                return self.areas[area2_idx].get_activations(), \
                        self.areas[area3_idx].get_activations(), \
                        self.get_afferent_connections([(area1_idx, area2_idx)])[(area1_idx, area2_idx)],\
                        self.get_afferent_connections([(area3_idx, area2_idx)])[(area3_idx, area2_idx)],\
                        self.get_afferent_connections([(area2_idx, area3_idx)])[(area2_idx, area3_idx)]

        return np.array(total_support_size_area2), np.array(total_support_size_area3)


    def skip_reciprocal_project_stream(self, stimulus_generator, area1_idx, area2_idx, area3_idx, max_iterations=50, use_recurrent=True, normalize=True, normalize_interval=5, initialize=True, return_weights_assembly=False):
        '''
        Yichen edited in March, 2023
        New design for reciprocal project with skip connection
        X --> Y <--> Z, and X --> Z
        '''
        if initialize:
            self.initialize_afferent_connections((area1_idx, area2_idx))
            self.initialize_afferent_connections((area1_idx, area3_idx))
            self.initialize_afferent_connections((area2_idx, area3_idx))
            self.initialize_afferent_connections((area3_idx, area2_idx))
        if use_recurrent and initialize:
            self.areas[area2_idx].initialize_recurrent_connections()
            self.areas[area3_idx].initialize_recurrent_connections()

        self.areas[area2_idx].clear_activations()
        self.areas[area3_idx].clear_activations()

        total_support_area2 = set()
        total_support_size_area2 = []
        total_support_area3 = set()
        total_support_size_area3 = []

        for t in range(max_iterations):
            self.areas[area1_idx].set_activations(stimulus_generator.sample_stimulus())
            area2_inputs = self.afferent_connections[(area1_idx, area2_idx)].dot(self.areas[area1_idx].activations)\
                            + self.afferent_connections[(area3_idx, area2_idx)].dot(self.areas[area3_idx].activations)
            area3_inputs = self.afferent_connections[(area1_idx, area3_idx)].dot(self.areas[area1_idx].activations)\
                            + self.afferent_connections[(area2_idx, area3_idx)].dot(self.areas[area2_idx].activations)
            if use_recurrent:
                area2_inputs += self.areas[area2_idx].recurrent_connections.dot(self.areas[area2_idx].activations)
                area3_inputs += self.areas[area3_idx].recurrent_connections.dot(self.areas[area3_idx].activations)
                
            new_activations_area2 = capk(area2_inputs, self.cap_size)
            new_activations_area3 = capk(area3_inputs, self.cap_size)

            hebbian_update(self.afferent_connections[(area1_idx, area2_idx)], new_activations_area2, self.areas[area1_idx].activations, self.beta)
            hebbian_update(self.afferent_connections[(area3_idx, area2_idx)], new_activations_area2, self.areas[area3_idx].activations, self.beta)
            hebbian_update(self.afferent_connections[(area1_idx, area3_idx)], new_activations_area3, self.areas[area1_idx].activations, self.beta)
            hebbian_update(self.afferent_connections[(area2_idx, area3_idx)], new_activations_area3, self.areas[area2_idx].activations, self.beta)

            if use_recurrent:
                hebbian_update(self.areas[area2_idx].recurrent_connections, new_activations_area2, self.areas[area2_idx].activations, self.beta)
                hebbian_update(self.areas[area3_idx].recurrent_connections, new_activations_area3, self.areas[area3_idx].activations, self.beta)

            if normalize and (t+1)%normalize_interval == 0:
                self.afferent_connections[(area1_idx, area2_idx)] /= self.afferent_connections[(area1_idx, area2_idx)].sum(axis=1)[:,np.newaxis]
                self.afferent_connections[(area3_idx, area2_idx)] /= self.afferent_connections[(area3_idx, area2_idx)].sum(axis=1)[:,np.newaxis]
                self.afferent_connections[(area1_idx, area3_idx)] /= self.afferent_connections[(area1_idx, area3_idx)].sum(axis=1)[:,np.newaxis]
                self.afferent_connections[(area2_idx, area3_idx)] /= self.afferent_connections[(area2_idx, area3_idx)].sum(axis=1)[:,np.newaxis]
                if use_recurrent:
                    self.areas[area2_idx].normalize_weights()
                    self.areas[area3_idx].normalize_weights()

            support_area2 = np.where(new_activations_area2 !=0)[0]
            total_support_area2 = total_support_area2.union(support_area2)
            total_support_size_area2.append(len(total_support_area2))
            support_area3 = np.where(new_activations_area3 !=0)[0]
            total_support_area3 = total_support_area3.union(support_area3)
            total_support_size_area3.append(len(total_support_area3))

            self.areas[area2_idx].set_activations(new_activations_area2)
            self.areas[area3_idx].set_activations(new_activations_area3)

        if return_weights_assembly:
            if use_recurrent:
                return self.areas[area2_idx].get_activations(), \
                        self.areas[area3_idx].get_activations(),\
                        self.areas[area2_idx].get_recurrent_connections(), \
                        self.areas[area3_idx].get_recurrent_connections(), \
                        self.get_afferent_connections([(area1_idx, area2_idx)])[(area1_idx, area2_idx)], \
                        self.get_afferent_connections([(area3_idx, area2_idx)])[(area3_idx, area2_idx)],\
                        self.get_afferent_connections([(area1_idx, area3_idx)])[(area1_idx, area3_idx)],\
                        self.get_afferent_connections([(area2_idx, area3_idx)])[(area2_idx, area3_idx)]

            else:
                return self.areas[area2_idx].get_activations(), \
                        self.areas[area3_idx].get_activations(), \
                        self.get_afferent_connections([(area1_idx, area2_idx)])[(area1_idx, area2_idx)],\
                        self.get_afferent_connections([(area3_idx, area2_idx)])[(area3_idx, area2_idx)],\
                        self.get_afferent_connections([(area1_idx, area3_idx)])[(area1_idx, area3_idx)],\
                        self.get_afferent_connections([(area2_idx, area3_idx)])[(area2_idx, area3_idx)]

        return np.array(total_support_size_area2), np.array(total_support_size_area3)

    def create_assemblies(self, stimuli, from_area_idx, to_area_idx, max_iterations=5, use_recurrent=True):
        self.areas[to_area_idx].set_activations(np.zeros_like(stimuli))

        for t in range(max_iterations):
            self.areas[from_area_idx].set_activations(stimuli)
            total_input = self.afferent_connections[(from_area_idx, to_area_idx)].dot(self.areas[from_area_idx].activations)
            if use_recurrent:
                recurrent_input = self.areas[to_area_idx].recurrent_connections.dot(self.areas[to_area_idx].activations)
                total_input += recurrent_input

            new_activations = capk(total_input, self.cap_size)
            
            self.areas[to_area_idx].set_activations(new_activations)

        return self.areas[to_area_idx].get_activations()


    def create_assemblies_reciprocal_project(self, stimuli, area1_idx, area2_idx, area3_idx, max_iterations=5, use_recurrent=True):
        '''
        Yichen edited in March, 2023
        Get the assembly activations for original reciprocal project
        X --> Y <--> Z
        '''
        self.areas[area2_idx].set_activations(np.zeros_like(stimuli))
        self.areas[area3_idx].set_activations(np.zeros_like(stimuli))

        for t in range(max_iterations):
            self.areas[area1_idx].set_activations(stimuli)
            area2_inputs = self.afferent_connections[(area1_idx, area2_idx)].dot(self.areas[area1_idx].activations)\
                            + self.afferent_connections[(area3_idx, area2_idx)].dot(self.areas[area3_idx].activations)
            area3_inputs = self.afferent_connections[(area2_idx, area3_idx)].dot(self.areas[area2_idx].activations)
            if use_recurrent:
                area2_inputs += self.areas[area2_idx].recurrent_connections.dot(self.areas[area2_idx].activations)
                area3_inputs += self.areas[area3_idx].recurrent_connections.dot(self.areas[area3_idx].activations)

            new_activations_area2 = capk(area2_inputs, self.cap_size)
            new_activations_area3 = capk(area3_inputs, self.cap_size)

            self.areas[area2_idx].set_activations(new_activations_area2)
            self.areas[area3_idx].set_activations(new_activations_area3)

        return self.areas[area2_idx].get_activations(), self.areas[area3_idx].get_activations()


    def create_assemblies_skip_reciprocal_project(self, stimuli, area1_idx, area2_idx, area3_idx, max_iterations=5, use_recurrent=True):
        '''
        Yichen edited in March, 2023
        Get the assembly activations for skip reciprocal project
        X --> Y <--> Z, and X --> Z
        '''
        self.areas[area2_idx].set_activations(np.zeros_like(stimuli))
        self.areas[area3_idx].set_activations(np.zeros_like(stimuli))

        for t in range(max_iterations):
            self.areas[area1_idx].set_activations(stimuli)
            area2_inputs = self.afferent_connections[(area1_idx, area2_idx)].dot(self.areas[area1_idx].activations)\
                            + self.afferent_connections[(area3_idx, area2_idx)].dot(self.areas[area3_idx].activations)
            area3_inputs = self.afferent_connections[(area1_idx, area3_idx)].dot(self.areas[area1_idx].activations)\
                            + self.afferent_connections[(area2_idx, area3_idx)].dot(self.areas[area2_idx].activations)
            if use_recurrent:
                area2_inputs += self.areas[area2_idx].recurrent_connections.dot(self.areas[area2_idx].activations)
                area3_inputs += self.areas[area3_idx].recurrent_connections.dot(self.areas[area3_idx].activations)

            new_activations_area2 = capk(area2_inputs, self.cap_size)
            new_activations_area3 = capk(area3_inputs, self.cap_size)

            self.areas[area2_idx].set_activations(new_activations_area2)
            self.areas[area3_idx].set_activations(new_activations_area3)

        return self.areas[area2_idx].get_activations(), self.areas[area3_idx].get_activations()


if __name__ == "__main__":
    B = Brain()
    x = np.zeros(B.num_neurons)
    stim_idx = np.random.permutation(B.num_neurons)[:B.cap_size]
    x[stim_idx] = 1.

    # print(B.project(x,0,1, use_recurrent=False))
    # print(B.project(x,0,1))

    from stimuli import StimulusGenerator
    S = StimulusGenerator(B.num_neurons, B.cap_size, coreset=stim_idx)
    # print(B.project_stream(S,0,1, use_recurrent=False))
    print(B.project_stream(S,0,1))
    X = S.sample_stimuli(100)
    Y = B.create_assemblies(X,0,1)
    print(Y.shape)
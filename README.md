# Variable-Binding-Capacity
The emergence of Associative Memories in Assembly Calculus and the evaluation of its capacity with respect to model parameters, brain size ($n$) and cap size ($k$), along with a newly introduced model: skip connections with variable binding.

brain.py 
---
Define Brain (class), which consists of several Area (class). 
Many important operations in Assembly Calculus, such as `projection` and `reciprocal_projection` are defined.

area.py
---
Define Area (class), which is where assemblies are formed.

stimuli.py
---
Given number of classes and number of samples per class, generate a stimuli matrix with dimension (self.nclasses, self.nsamples, self.num_neurons)

utils.py
---
Define commonly used functions such as `capk`, `hebbian_update`, `find_feedforward_matrix_index`, `generate_labels`.

run_grand_capacity.py
---
Capacity of Assembly Calculus in terms of `n` and `k`.
To run, use the following command:

Run projection wrt to n
```
python run_grand_capacity.py --operation project --parameter n --ntrials 5 --plot plot1 --skipConnection True
```

Run projection wrt to k
```
python run_grand_capacity.py --operation project --parameter k --ntrials 5 --plot plot1 --skipConnection True
```

Note that `skipConnection` is a `bool`, indicating whether to do direct decoder (skip connection) from area 0 to area 2!

Run reci-project wrt to n
```
python run_grand_capacity.py --operation reci-project --parameter n --ntrials 5 --plot plot1 --skipConnection True
```

Run reci-project wrt to k
```
python run_grand_capacity.py --operation reci-project --parameter k --ntrials 5 --plot plot1 --skipConnection True
```


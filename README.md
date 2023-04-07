This repository contains the code for reproducing experiments conducted in our paper:

**Through the Lens of Emergent Associative Memories: Skip Connections Increase the Capacity of Variable Binding Mechanisms**
Yi Xie*, Yichen Li*, Akshay Rangamani, Tomaso Poggio

The emergence of Associative Memories in Assembly Calculus and the evaluation of its capacity with respect to model parameters, brain size ($n$) and cap size ($k$), along with a newly introduced model: *variable binding* (**VB**) with *skip connection*.

## A) Documentation on files for basic setup

### brain.py 
Define `Brain` (class), which consists of several `Area` (class). 
Many important operations in Assembly Calculus, such as `projection` and `reciprocal_projection` are defined.

### area.py
Define `Area` (class), which is where assemblies are formed.

### stimuli.py
Given number of classes and number of samples per class, generate a stimuli matrix with dimension (`self.nclasses`, `self.nsamples`, `self.num_neurons`)

### utils.py
Define commonly used functions such as `capk`, `hebbian_update`, `find_feedforward_matrix_index`, `generate_labels`.

## B) To reproduce Assembly Recall experiments:
Run `Assembly_Recall.ipynb`.

## C) To reproduce the model capacity:
`run_grand_capacity.py` measures the capacity of Assembly Calculus in terms of `n` and `k`.
To run, use the following command:

**Run projection wrt to n**
```
python run_grand_capacity.py --operation project --parameter n --ntrials 5 --plot plot1 --skipConnection True --transposeWeight False
```

**Run projection wrt to k**
```
python run_grand_capacity.py --operation project --parameter k --ntrials 5 --plot plot1 --skipConnection True --transposeWeight False
```

Note that `skipConnection` is a `bool` flag, indicating whether to do direct decoder (skip connection) from area 0 to area 2.
Including `--skipConnection` makes it True, and excluding it makes it False.

We also note that reci-project is noisier than project in general, so one can consider to relax the threadhold when searching capacity. Here, we only include the non-relaxed threadhold.

**Run reci-project wrt to n**
```
python run_grand_capacity.py --operation reci-project --parameter n --ntrials 5 --plot plot1 --skipConnection
```

**Run reci-project wrt to k**
```
python run_grand_capacity.py --operation reci-project --parameter k --ntrials 5 --plot plot1 --skipConnection
```

## D) To reproduce the SVD distance plot:
Run `python svd_plot/cns_fig1.py`

This repository contains the code for reproducing experiments conducted in our paper:

**Through the Lens of Emergent Associative Memories: <br>
Skip Connections Increase the Capacity of Variable Binding Mechanisms**

Yi Xie*, Yichen Li*, Akshay Rangamani, Tomaso Poggio

**In brief:** we study the two classic operations in Assembly Calculus, $\texttt{project}$ and \texttt{reciprocal-project}. In particular, \texttt{project} is used as a baseline for \texttt{reciprocal-project}. We establish that Associative Memories emerge in Assembly Calculus through Hebbian Learning with a stream of stimuli drawn from the same distribution; further, this phenomenon generalizes to multiple classes of stimuli (**see D**). Additionally, we discuss the role of homeostasis (normalization) within the spectral structure and that recurrent connections make the assembly model significantly more robust (**see B**). In light of this establishment, we measure the capacity of Associative Memories in Assembly Calculus as a function of model parameter $N$, number of neurons in each brain area, and $K$, the maximum number of active neurons in an area at any time (**see C**). We discuss the phenomenon of cascading capacity of assemblies over hierarchical brain areas. Lastly, we further leverage our knowledge of \texttt{reciprocal-project}'s hypothesized biological role in variable binding mechanisms to propose an addition of \textit{skip connection}. This addition allows a direct access to the pointer variable by the sensory input, which increases the capacity by an order of magnitude as a way to tackle the challenges caused by the aforementioned phenomenon. 

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

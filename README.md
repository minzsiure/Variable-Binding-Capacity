This repository contains the code for reproducing experiments conducted in our paper:

**Skip Connections Increase the Capacity of <br>
Associative Memories in Variable Binding Mechanisms**

Yi Xie $^{1*}$, Yichen Li $^{2*}$, Akshay Rangamani $^{1}$, Tomaso Poggio $^{1}$

$^1$ Center for Brains, Minds, and Machines,  Massachusetts Institute of Technology

$^2$ Department of Psychology, Harvard University

**Abstract:** 
The flexibility of intelligent behavior is fundamentally attributed to the ability to separate and assign structural information from content in sensory inputs. Variable binding is the atomic computation that underlies this ability. In this work, we investigate the implementation of variable binding via pointers of assemblies of neurons, which are sets of excitatory neurons that fire together. The Assembly Calculus is a framework that describes a set of operations to create and modify assemblies of neurons. We focus on the $\texttt{project}$ (which creates assemblies) and $\texttt{reciprocal-project}$ (which performs variable binding) operations and study the capacity of networks in terms of the number of assemblies that can be reliably created and retrieved. We find that assembly calculus networks implemented through Hebbian plasticity resemble associative memories in their structure and behavior. However, for networks with $N$ neurons per brain area, the capacity of variable binding networks ($0.01N$) is an order of magnitude lower than the capacity of assembly creation networks ($0.22N$). To alleviate this drop in capacity, we propose a $\textit{skip connection}$ between the input and variable assembly, which boosts the capacity to a similar order of magnitude ($0.1N$) as the $\texttt{Project}$ operation, while maintain its biological plausibility. 

## Citation
```
@article {5360,
	title = {Skip Connections Increase the Capacity of Associative Memories in Variable Binding Mechanisms},
	number = {142},
	year = {2023},
	month = {06/2023},
	abstract = {<p>The flexibility of intelligent behavior is fundamentally attributed to the ability to separate and assign structural information from content in sensory inputs. Variable binding is the atomic computation that underlies this ability. In this work, we investigate the implementation of variable binding via pointers of assemblies of neurons, which are sets of excitatory neurons that fire together. The Assembly Calculus is a framework that describes a set of operations to create and modify assemblies of neurons. We focus on the project (which creates assemblies) and reciprocal-project (which performs vari- able binding) operations and study the capacity of networks in terms of the number of assemblies that can be reliably created and retrieved. We find that assembly calculus networks implemented through Hebbian plasticity resemble associative memories in their structure and behavior. However, for net- works with N neurons per brain area, the capacity of variable binding networks (0.01N) is an order of magnitude lower than the capacity of assembly creation networks (0.22N). To alleviate this drop in capacity, we propose a skip connection between the input and variable assembly, which boosts the capacity to a similar order of magnitude (0.1N ) as the Project operation, while maintain its biological plausibility.</p>
},
	author = {Yi Xie and Yichen Li and Akshay Rangamani}
}
```

## Dependencies
```
pandas numpy matplotlib seaborn scipy scikit-learn
```

## Structure of basic setup

### brain.py 
Define `Brain` (class), which consists of several `Area` (class). 
Many important operations in Assembly Calculus, such as $\texttt{project}$ and $\texttt{reciprocal-project}$ are defined.

### area.py
Define `Area` (class), which is where assemblies are formed.

### stimuli.py
Given number of classes and number of samples per class, generate a stimuli matrix with dimension (`self.nclasses`, `self.nsamples`, `self.num_neurons`)

### utils.py
Define commonly used functions such as `capk`, `hebbian_update`, `find_feedforward_matrix_index`, `generate_labels`.

## To reproduce Pattern Completion experiments:
Run  `run_pattern_completion.py`.

## To reproduce Assembly Recall experiments:
Run `Assembly_Recall.ipynb` for singular class implementation.
Run `run_assem_recall.py` for multiclass implementations.

## To reproduce the model capacity:
`run_grand_capacity.py` measures the capacity of Assembly Calculus in terms of `n` and `k`.
To run, use the following command:

**Run $\texttt{project}$ wrt to $n$**
```
python run_grand_capacity.py --operation project --parameter n --ntrials 5 --plot plot1  
```

**Run $\texttt{project}$ wrt to $k$**
```
python run_grand_capacity.py --operation project --parameter k --ntrials 5 --plot plot1 
```

**Run $\texttt{project}$ wrt to $p$**
```
python run_grand_capacity.py --operation project --parameter p --ntrials 5 --plot plot1 --num_neurons 250
```

> Note that `skipConnection` is a `bool` flag, indicating whether to do direct decoder (skip connection) from area 0 to area 2.
Including `--skipConnection` makes it True, and excluding it makes it False.

> We also note that $\texttt{reciprocal-project}$ is noisier than $\texttt{project}$ in general, so one can consider to relax the threadhold when searching capacity. Here, we only include the non-relaxed threadhold.

**Run $\texttt{reciprocal-project}$ wrt to $n$**
```
python run_grand_capacity.py --operation reci-project --parameter n --ntrials 5 --plot plot1 --skipConnection
```

**Run $\texttt{reciprocal-project}$ wrt to $k$**
```
python run_grand_capacity.py --operation reci-project --parameter k --ntrials 5 --plot plot1 --skipConnection
```

**Run $\texttt{reciprocal-project}$ wrt to $p$**
```
python run_grand_capacity.py --operation reci-project --parameter p --ntrials 5 --plot plot1 --skipConnection
```

## To reproduce the SVD distance plot:
Run `python svd_plot/cns_fig1.py`



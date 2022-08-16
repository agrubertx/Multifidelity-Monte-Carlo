# Multifidelity-Monte-Carlo
A Python implementation of multifidelity Monte Carlo estimation, with experiments reproducing the results of the papers:

1. A. Gruber, M. Gunzburger, L. Ju, and Z. Wang.  [A Multifidelity Monte Carlo Method for Realistic Computational Budgets](https://arxiv.org/abs/2206.07572#),

1. A. Gruber, M. Gunzburger, L. Ju, R. Lan, and Z. Wang.  [Multifidelity Monte Carlo Estimation for Efficient Uncertainty Quantification in Climate-Related Modeling](toBeUploaded).


## Description

The file "mfmc.py" contains all functions necessary for using the method.  Each subfolder "xxx_experiment/" corresponds to a simulation from one of the above papers, which can be reproduced by running all cells (at once) in the associated iPython notebook.  The folders "toy_experiment" and "burgers_experiment" correspond to paper (1), while the rest correspond to paper (2).  Note that the experiments in paper (1) generate their own data, while the experiments in paper (2) use the precollected data available in their "data/" subfolder.


## Data
To perform MFMC on your own dataset, you must simply provide
- an (N_samples x N_models) array of model evaluations.
- an (N_models)-vector of weights (usually computational costs of each model)
See the source code in "mfmc.py" and all the experiments recorded here for examples.


## Installation
The code was written and tested using Python 3.8 on Mac OS 12.1.  The required dependencies are:
* [Numpy](https://numpy.org/)
* [Matplotlib](https://matplotlib.org/)
* [Scipy](https://www.scipy.org/) (for interpolation and Burgers ROM)


## Citation
Please cite our paper(s) if you find this code useful in your own work:
```
@misc{gruber2022a,
      title={A Multifidelity Monte Carlo Method for Realistic Computational Budgets}, 
      author={Anthony Gruber and Max Gunzburger and Lili Ju and Zhu Wang},
      year={2022},
      eprint={2206.07572},
      archivePrefix={arXiv},
      primaryClass={math.NA}
}

@misc{gruber2022multifidelity,
      title={Multifidelity Monte Carlo Estimation for Efficient Uncertainty Quantification in Climate-Related Modeling}, 
      author={Anthony Gruber and Max Gunzburger and Lili Ju and Rihui Lan and Zhu Wang},
      year={2022},
      eprint={to_fill_in},
      archivePrefix={EGUsphere},
}
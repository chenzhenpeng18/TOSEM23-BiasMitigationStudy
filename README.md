# Revisiting Bias Mitigation Methods in Software Fairness

Welcome to visit the homepage of our TSE submission entitled "Revisiting Bias Mitigation Methods in Software Fairness". The homepage contains scripts and data used in this paper.

## Experimental environment

We use Python 3.7.11 for our experiments. We use the IBM AI Fairness 360 (AIF360) toolkit for implementing bias mitigation methods and computing fairness metrics. This toolkit can be installed as follows:

```
pip install aif360
```

More information on AIF360 can be found on https://github.com/Trusted-AI/AIF360.

In addition, we require the following Python packages:
```
pip install sklearn
pip install numpy
pip install shapely
pip install matplotlib
pip install "tensorflow >= 1.13.1, < 2"
```

## Dataset

We use the five default datasets supported by the AIF 360 toolkit. Please refer to https://github.com/Trusted-AI/AIF360/tree/master/aif360/data for the raw data files.

## Scripts and results
The repository contains four folders:

* ```Fair360/``` contains code for implementing the 15 bias mitigation methods in AIF360.

* ```Fairway/``` contains code for implementing Fairway, a bias mitigation method proposed by [Chakraborty et al.](https://doi.org/10.1145/3368089.3409697) in ESEC/FSE 2020.

* ```Fair-SMOTE/``` contains code for implementing Fair-SMOTE, a bias mitigation method proposed by [Chakraborty et al.](https://doi.org/10.1145/3468264.3468537) in ESEC/FSE 2021.

* ```Fairea/``` contains code for measuring fairness-performance trade-offs. It is implemented based on Fairea, a trade-off benchmark method proposed by [Hort et al.](https://doi.org/10.1145/3468264.3468565) in ESEC/FSE 2021.

* ```Result/``` contains the raw results of the original models and the models after applying 17 bias mitigation methods in eight tasks. Each file in this folder has 53 columns, with the first column indicating the metric, the next 50 columns the metric values of 50 runs, and the last two columns the mean and std values of the 50 runs.

## Declaration

Thanks to the authors of existing bias mitigation methods for open source, to facilitate our implementation of this paper. Therefore, when using our code or data for your work, please also consider citing their papers, including [AIF360](https://arxiv.org/abs/1810.01943), [Fairway](https://doi.org/10.1145/3368089.3409697), [Fair-SMOTE](https://doi.org/10.1145/3468264.3468537), and [Fairea](https://doi.org/10.1145/3468264.3468565).  


# MuheCC
This is a repliacation package for Paper `Boosting Commit Classification Based on Multivariate Mixed Features and Heterogeneous Classifier Selection`.<br>

## Content
1. [Get Started](#1-Get-Started)<br>
&ensp;&ensp;[1.1 Requirements](#11-Requirements)<br>
&ensp;&ensp;[1.2 Dataset](#12-Dataset)<br>
2. [Run](#2-Run)<br>


## 1 GetStarted
### 1.1 Requirements
* Packages:
  * bayesian_optimization==1.4.2
  * hyperopt==0.2.7
  * lightgbm==3.2.1
  * matplotlib==3.5.2
  * numpy==1.21.5
  * optuna==3.2.0
  * pandas==1.3.5
  * scikit_learn==1.2.2
  * torch==1.12.1
  * transformers==4.18.0

  You can see these in ```EnsembleLearning/requirements.txt```.
### 1.2 Dataset
The dataset can be found in ```EnsembleLearning/Commit_dataset_final.xlsx```.

## 2 Run
1. Train and test three baselines
```
python EnsembleLearning/BERT+lightGBM.py
```
2. Train and test MuheCC
```
python EnsembleLearning/.stacking_model_choose.py
```

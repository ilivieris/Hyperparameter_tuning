# Hyperparameter tuning

This is the implementation code for hyperparameter optimization framework for tuning sklearn models using Optuna. The dataset is splitted in a training set and a hold-out test. The tuning is performed on the training set (using k-fold stratified cross-validation) while the evaluation on the hold-out set.

Currently, the available models are
- XGBoost
- Random-Forest
but the code can be easily modified to include any sklearn model.

All the results, figures and models are logged in MLflow.

<br/>

## Optuna

**Optuna** is a hyperparameter optimization framework for tuning models. It lets you understand how hyperparameters affect your model and improves your model performance.

There are many samplers available to tune your models. It still contains the standard grid search and random search models. But, in addition, you can also choose :

- Tree-structured Parzen Estimator 
- A Quasi-Monte Carlo Sampler
- An Intersection Search Space Sampler

More information can be fould in
- https://towardsdatascience.com/mlops-with-optuna-b7c52d931b4b
- https://towardsdatascience.com/hyper-parameter-optimization-with-optuna-4920d5732edf




<br/>


## Data
---

The toy-dataset used in this repository concern a multi-class classification problem contained in ``Data/dataset.csv``

<br/>

## Notebooks
---

- ``01. EDA.ipynb``: performs a Exploratory Data Analysis on the given dataset using sweetviz package
- ``02. Random-Forest.ipynb`` hyperparameter tuning of Random-Forest model
- ``03. XGBoost.ipynb`` hyperparameter tuning of XGBoost model
- ``04. Explainability.ipynb`` loads a model from MLflow and provides local and global explainability information using LIME and SHAP methods.

<br/>

## How to run
--- 

1. Create a virtual environment 
```
    conda create -n myEnv python=3.8
```

2. Activate the virtual environment 
```
    conda activate myEnv
```
3. Install requirements 
```
    pip install -r requirements.txt
```
4. Run jupyter notebooks
```
    01. EDA.ipynb
    02. XGBoost.ipynb
    03. Random-Forest.ipynb
    04. Explainability.ipynb
```


<br/>

## Requirements

- python==3.8
- lightgbm==3.3.5
- optuna==3.1.0
- numpy==1.23.5
- pandas==1.5.2
- scikit-learn==1.2.0
- xgboost==1.7.3
- lightgbm==3.3.5


<br/>

## :mailbox: Contact
---
Ioannis E. Livieris (livieris@novelcore.eu)

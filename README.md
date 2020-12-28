# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
In this task we're going to  look at the UCI Bank Marketing dataset where we are going to predict which of their clients are going to open a term deposit with the bank. So it's at binary classification problem. -Is the client going to open an account? -yes or no?

We will try two diffenrent approches to solving this task by using:
..*Logistic Regression (Scikit-Learn) with a hyperparameter search using Azure's Hyperdrive.
..*Azure's Auto Machine Learning (AML) 
Eventhough the AML has different algorithms at disposal it actually isn't much better than a logisitc regression which got an accuracy of 91.55% compared to the AML at 91.74%

## Scikit-learn Pipeline
The Scikit-learn pipeline is written in the script "train.py" where we will handle all the preprocession and training.
First we download the dataset from the url given, using the AzureDataFactory class.
Then the data is cleaned in various steps embedded in a method ('clean_data').
Data is cleaned and one-hot encoded, eg. by changing yes/no strings to binary 1/0 data.
Then data is split into 80% train set and 20% test set before each set is split into features and targets ('y').
As algorithm we will use a SKlearn logistic regression estimator.

At this point the dataset in handed over to hyperparametertuning using Hyperdrive. I chose to search over three different features:
  * "Solver" - specific algorithm to use for optimization
  * "C" - inverse of regularzation strength
  * "Max iterations" - Maximum number of iterations taken for the solvers to converge.
  
These are part of the parameters used in SKlearns Logistic Regression.
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

![alt text](https://github.com/MichaelSoegaard/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/img/hyperdrive.png "Hyperdrive runs")

For earlystopping we are using Bandit policy. This is a policy based on slack factor. In this case, if accuracy of the current run isn't within the slack amount from the best run, then the current run is terminated. This saves us some compute usage.

For the hyperparameter search there are three different approaches: Baysian, Grid- and Random sampling. In this case we are using Random sampling as it allows the use of early-stopping like the Bandit policy we use.
After training of each sampling the model is tested on test set, accuracy noted and model saved. At the end of the Hyperdrive run the model with the best accuracy is selected.

## AutoML
The best performing model was a Soft Voting Ensemble found using AutoML. It uses XGBoost Classifier with a standard scaler wrapper

## Pipeline comparison
Preprocessing and split was the same for both Hyperdrive and AML models. The main difference is that AML has many different algorithms at its disposal, thus it should have a better chance of getting a the best accuracy. In contrary, when using Hyperdrive we are actually fine tuning our Logistic regression model. The two models are almost equally good in this case. I would have thought AML would get an inferior accuracy. But it might not be the case becuase of the nature of the data.

## Future work
A relevant next step could be to make a Hyperdrive run on the XGBoost Ensemble model AML choose. This way we can fione tune the model and probably increase accuracy a bit. Another option would be to try the two pipelines on different data to get a more realistic (weighted) comparison.

## Proof of cluster clean up
![alt text](https://github.com/MichaelSoegaard/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/img/delete_cluster.png "Delete cluster")

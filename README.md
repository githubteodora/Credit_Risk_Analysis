# Credit Risk Analysis, Logistic Regression
Goal: To solve an inherently unbalanced classification problem.

## Tools Used for Prediction Modelling
 - Python 3 
 - Libraries: imbalanced-learn, scikit-learn; (Naive Random Oversampling, SMOTE Oversampling, Undersampling with ClusterCentroids, SMOTEENN, Balanced Random Forest, Easy Ensemble AdaBoost)
 - Jupyter Notebook

## Overview of the Loan Prediction Risk Analysis:
Using the credit card credit dataset from LendingClub, a peer-to-peer lending services company, this project aims to oversample the data using the RandomOverSampler and SMOTE algorithms, and undersample the data using the ClusterCentroids algorithm. Then, a combinatorial approach of over- and undersampling using the SMOTEENN algorithm is ised. Next, the project will compare two machine learning models that reduce bias - BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk. Fianlly, the author will offer an evaluation of the performance of these models and make a recommendation on whether they should be used to predict credit risk.

## Results:

### 1. Naive Random Oversampling
#### Balanced Accuracy Score:
```
balanced_accuracy_score(y_test, y_pred)
0.6663237827524566
```
#### Precision and Recall Scores:
```
                   pre       rec       spe        f1       geo       iba       sup
  high_risk       0.01      0.70      0.63      0.02      0.67      0.45       101
   low_risk       1.00      0.63      0.70      0.77      0.67      0.44     17104
avg / total       0.99      0.63      0.70      0.77      0.67      0.44     17205
```

### 2. SMOTE Oversampling
#### Balanced Accuracy Score:
```
balanced_accuracy_score(y_test, y_pred)
0.6623064259185507
```
#### Precision and Recall Scores:
```
                   pre       rec       spe        f1       geo       iba       sup
  high_risk       0.01      0.63      0.69      0.02      0.66      0.44       101
   low_risk       1.00      0.69      0.63      0.82      0.66      0.44     17104
avg / total       0.99      0.69      0.63      0.81      0.66      0.44     17205

```

### 3. Undersampling with ClusterCentroids
#### Balanced Accuracy Score:
```
balanced_accuracy_score(y_test, y_pred)
0.6623064259185507
```
#### Precision and Recall Scores:
```
                   pre       rec       spe        f1       geo       iba       sup
  high_risk       0.01      0.63      0.69      0.02      0.66      0.44       101
   low_risk       1.00      0.69      0.63      0.82      0.66      0.44     17104
avg / total       0.99      0.69      0.63      0.81      0.66      0.44     17205
```

### 4. SMOTEENN (Combination: Over and Under Sampling)
#### Balanced Accuracy Score:
```
balanced_accuracy_score(y_test, y_pred)
0.6447892450610824
```
#### Precision and Recall Scores:
```
                   pre       rec       spe        f1       geo       iba       sup
  high_risk       0.01      0.71      0.58      0.02      0.64      0.42       101
   low_risk       1.00      0.58      0.71      0.73      0.64      0.41     17104
avg / total       0.99      0.58      0.71      0.73      0.64      0.41     17205
```

### 5. Balanced Random Forest
#### Balanced Accuracy Score:
```
accuracy_score(y_test, y_pred)
0.9076431269979657
```
#### Precision and Recall Score:
```
                   pre       rec       spe        f1       geo       iba       sup
  high_risk       0.04      0.67      0.91      0.07      0.78      0.59        87
   low_risk       1.00      0.91      0.67      0.95      0.78      0.62     17118
avg / total       0.99      0.91      0.67      0.95      0.78      0.62     17205
```

### Easy Ensemble AdaBoost
#### Balanced Accuracy Score:
```
accuracy_score(y_test, y_pred)
0.9426329555361813
```
#### Precision and Recall Score:
```
                   pre       rec       spe        f1       geo       iba       sup
  high_risk       0.07      0.91      0.94      0.14      0.93      0.85        87
   low_risk       1.00      0.94      0.91      0.97      0.93      0.86     17118
avg / total       0.99      0.94      0.91      0.97      0.93      0.86     17205
```

## Summary of the Results:

### Definitions:
Precision and recall are:<br>
Precision = # True positives / # predicted positive = TP/(TP+FP)<br>
Recall = # True positives / # positives = TP / (TP+FN)<br>

Recall (aka Sensitivity or True Positive Rate): the measure of our model correctly identifying True Positives.<br>
Precision:  the ratio between the True Positives and all the Positives.<br>
Precision and recall are highly used for imbalanced dataset because in an highly imbalanced dataset. <br>

For this dataset, we can consider that achieving a high recall is more important than getting a high precision – we would like to detect as many high risk individuals as possible.

### Recommended Model:
Having evaluatied 6 models for detecting high credit risk indivuduals, at this stage it looks like the Easy Ensemble AdaBoost model is the best performing one, as it yields recall results closest to 1 both for high and low credit risk individuals.


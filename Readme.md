# Customer Churn Prediction using Ensemble Model
## Introduction
This tutorial demonstrates how to build an ensemble model for predicting customer churn using Python and various machine learning libraries. The code combines logistic regression, decision tree, and neural network models into a voting classifier to create a robust prediction model.

Prerequisites
Before starting, ensure you have the following libraries installed:

 1. **pandas**
 2. **numpy**
 3. **scikit-learn**
 4. **imbalanced-learn**
 5. **matplotlib**

You can install these libraries using pip:

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib
```
## Step 1: Importing Libraries
The first step is to import all necessary libraries:
``` python
import pandas as pd
import numpy as np
from os import path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, classification_report, ConfusionMatrixDisplay
from sklearn.ensemble import VotingClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
```
## Step 2: Data Preprocessing
Loading the Dataset
The code attempts to load a CSV file. If the file doesn't exist, it generates a random dataset:

``` python
dataset_path = 'path//to//csv_file[containing:]'
if path.exists(dataset_path):
    data = pd.read_csv(dataset_path)
else:
    # Generate random data
    nsample = 1000
    npredictor = 10
    number_of_class = 5
    sample_data = np.random.random([nsample,npredictor])
    target_class = np.random.randint(0,number_of_class,nsample)
    var_names = ['Variable'+str(i+1) for i in range(npredictor)]
    data = pd.DataFrame(data=sample_data,columns=var_names)
    data.insert(npredictor,'Churn',target_class)
```
### Data Cleaning and Feature Engineering
The code then performs some basic data cleaning:

```python
data.dropna(inplace=True)
```
### Splitting Features and Target
The features (X) and target variable (Y) are separated:

``` python
X = data.drop(columns=['Churn'])  # Independent features
Y = data['Churn']  # Target variable
```
### Train-Test Split
The data is split into training and testing sets:

``` python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=10)
```
### Handling Class Imbalance
SMOTE (Synthetic Minority Over-sampling Technique) is used to address class imbalance:

```python
smote = SMOTE(random_state=10)
X_train_resampled, Y_train_resampled = smote.fit_resample(X_train, Y_train)
```
### Feature Scaling
The features are standardized using StandardScaler:

``` python
scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)
```
## Step 3: Building the Model
Three different models are created:

- **Logistic Regression**
- **Decision Tree**
- **Neural Network (MLPClassifier)**

These models are then combined into an ensemble using VotingClassifier:
```python

log_reg = LogisticRegression(random_state=10)
decision_tree = DecisionTreeClassifier(random_state=10)
neural_net = MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=300, random_state=10)

ensemble_model = VotingClassifier(estimators=[
    ('LogReg', log_reg),
    ('DecisionTree', decision_tree),
    ('NeuralNet', neural_net)
], voting='soft')
```
## Step 4: Model Training
The ensemble model is trained on the resampled training data:
```python
ensemble_model.fit(X_train_resampled, Y_train_resampled)
```
## Step 5: Model Evaluation (Not shown in the provided code)
After training the model, you would typically evaluate its performance. This could include:

### Making predictions on the test set:
``` python
y_pred = ensemble_model.predict(X_test)
```
### Calculating the ROC AUC score:
```python
roc_auc = roc_auc_score(Y_test, ensemble_model.predict_proba(X_test), multi_class='ovr')
print(f"ROC AUC Score: {roc_auc}")
```
### Generating a classification report:
```python
print(classification_report(Y_test, y_pred))
```
### Visualizing the confusion matrix:
``` python
ConfusionMatrixDisplay.from_estimator(ensemble_model, X_test, Y_test)
plt.show()
```
## Conclusion
This notebook demonstrates a complete workflow for building an ensemble model to predict customer churn. It covers data preprocessing, handling class imbalance, feature scaling, and combining multiple models into an ensemble. The model can be further fine-tuned by adjusting hyperparameters or trying different combinations of base models.

Remember to replace the placeholder dataset path with your actual data file for real-world applications. Also, consider adding more extensive error handling and data validation steps when working with production data.

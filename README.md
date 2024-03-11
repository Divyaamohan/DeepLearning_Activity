# DeepLearning_Activity
Housing Dataset Analysis and Model Training
This repository contains code for analyzing the California housing dataset and training predictive models using TensorFlow and Keras.

# Steps:
## Data Loading and Preprocessing:

Loaded the housing dataset from a Parquet file.
Checked for missing values and handled them appropriately.
Calculated target variable (median house value) and performed data exploration.

## Data Exploration:

Visualized data distributions and relationships using seaborn and matplotlib.
Checked for outliers and treated them using filtering techniques.
Explored correlations between variables using correlation matrices and heatmaps.

## Feature Scaling:

Applied Min-Max Scaling and Standardization to the features for normalization.

## Model Training:

Trained a Deep Neural Network (DNN) model using Keras for regression.
Trained a Linear Regression model for comparison.

## Model Evaluation:

Evaluated models using Mean Squared Error (MSE) and Mean Absolute Error (MAE).
Utilized K-Fold cross-validation for robust evaluation.

## Model Comparison:
Compared the performance of Linear Regression, Multi-Layer Neural Network (MLN), and Deep Neural Network (DNN) models based on MSE.
## Conclusion:
Determined the best-performing model based on MSE.

## Dependencies:
Python 3.x
TensorFlow
Keras
pandas
numpy
seaborn
matplotlib
scikit-learn
ydata-profiling
tensorflow-data-validation
fastparquet
pydot
cartopy

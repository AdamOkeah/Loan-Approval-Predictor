Loan Default Prediction Using Machine Learning

This project uses various machine learning models and techniques to predict loan default status, employing a combination of preprocessing, oversampling, and different classification algorithms.

Overview

The goal is to build robust predictive models to classify loan default status based on a dataset containing borrower attributes. The project demonstrates the application of preprocessing techniques, SMOTE for handling imbalanced datasets, and multiple machine learning models.

Key Features

Dataset: Includes borrower information and loan features.

Preprocessing: Handles missing values, categorical encoding, scaling, and feature selection.

Oversampling: Uses SMOTE to balance the dataset for better model performance.

Machine Learning Models:

K-Nearest Neighbors (KNN)

Random Forest

Support Vector Machine (SVM)

Deep Neural Networks (DNN)

Ensemble Methods

Evaluation Metrics:

Accuracy

Precision

Recall

F1-Score

Confusion Matrices

Installation

Prerequisites:

Python 3.7+

Required libraries:

pip install pandas numpy scikit-learn imbalanced-learn tensorflow matplotlib seaborn xgboost

Files and Scripts

all.py: Implements feature engineering, SMOTE oversampling, and trains multiple models (KNN, Random Forest, SVM).

balancingall.py: Explores SMOTE for balancing datasets and model training.

dnn.py: Defines and trains a deep neural network using TensorFlow/Keras for loan default prediction.

knn.py: Focuses on the KNN algorithm for classification and visualises confusion matrix.

imbalanace.py: Checks and explores target variable imbalance in the dataset.

Data Files:

loan_data.csv: Dataset used for training and testing.

Usage

Running the Models:

Clone the repository:

git clone <repository_url>
cd <repository_folder>

Run individual scripts for specific models or tasks:

python all.py
python dnn.py
python knn.py

Modify hyperparameters and configurations directly in the scripts for experimentation.

Results

Correlation with Loan Status:

Visualises features most correlated with loan default status.

Model Metrics:

Comparison of accuracy, precision, recall, and F1-score across models.

Confusion Matrices:

Plots confusion matrices to evaluate model predictions visually.

Visualisations

Correlation with Loan Status:


Confusion Matrices:

Example: KNN Confusion Matrix:


Future Work

Incorporate advanced ensemble methods (e.g., Gradient Boosting, XGBoost).

Hyperparameter tuning for optimal model performance.

Explore other oversampling and under-sampling techniques.

Evaluate on larger, real-world datasets.

Contact

For questions or contributions:

Author: Adam Okeahalam

Email: adamokeahalam@gmail.com

Feel free to open issues or submit pull requests!

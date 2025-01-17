import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
import seaborn as sns

# Load the dataset
df = pd.read_csv('loan_data.csv')


# Identify and handle non-numeric columns and apply one-hot encoding
for col in df.columns:
    if df[col].dtype == 'object':  # Check for non-numeric columns
        try:
            # Attempt to convert to datetime and extract features
            df[col] = pd.to_datetime(df[col])
            df[f'{col}_Year'] = df[col].dt.year
            df[f'{col}_Month'] = df[col].dt.month
            df[f'{col}_Day'] = df[col].dt.day
            df = df.drop(columns=[col])  # Drop original date column
        except:
            # For non-date objects, apply one-hot encoding
            df = pd.get_dummies(df, columns=[col], drop_first=True)
            



# Compute the correlation matrix (only on numeric columns)
correlation_matrix = df.corr()

# Select correlations with the target variable (assuming the target column is 'loan_status')
target_correlation = correlation_matrix['loan_status'].sort_values(ascending=False)

# Display correlations
print("\nCorrelation with Loan Status:")
print(target_correlation)

# Plot correlations with the target variable
plt.figure(figsize=(12, 8))
target_correlation.drop('loan_status').plot(kind='bar')
plt.title('Correlation with Loan Status', fontsize=16)
plt.ylabel('Correlation Coefficient', fontsize=14)
plt.xlabel('Features', fontsize=14)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Select features with correlation above the threshold
threshold = 0.003  # Parameter for deciding which features have a substantial impact on loan status
high_corr_features = correlation_matrix.index[abs(correlation_matrix["loan_status"]) > threshold].tolist()
if "loan_status" in high_corr_features:
    high_corr_features.remove("loan_status")
print("\nHigh Correlated Features:", high_corr_features)

# Define features and target
X = df[high_corr_features]
y = df["loan_status"]


missing_values = df.isnull().sum()
print(missing_values)

# Scale and normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.astype('float32'))
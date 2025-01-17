import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('loan_data.csv')

# Identify and handle non-numeric columns
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

# Check the structure of the data after handling categorical columns
print(f"Shape after preprocessing: {df.shape}")

# Compute the correlation matrix
correlation_matrix = df.corr()

# Select correlations with the target variable
target_correlation = correlation_matrix['loan_status'].sort_values(ascending=False)
print("Correlation with Loan Status:")
print(target_correlation)

# Plot correlations
plt.figure(figsize=(10, 6))
target_correlation.drop('loan_status').plot(kind='bar')
plt.title('Correlation with Loan Status')
plt.ylabel('Correlation Coefficient')
plt.xlabel('Features')
#plt.show()

# Select features with high correlation
threshold = 0.003
high_corr_features = correlation_matrix.index[abs(correlation_matrix["loan_status"]) > threshold].tolist()
high_corr_features.remove("loan_status")
X = df[high_corr_features]
y = df["loan_status"]

# Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(X.astype('float32'))

# Apply SMOTE to balance the classes
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Initialize the Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

# Fit the model to the training data
rf.fit(X_train, y_train)

# Make predictions
y_pred = rf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

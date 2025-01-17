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

# Compute the correlation matrix (only on numeric columns)
correlation_matrix = df.corr()

# Select correlations with the target variable (assuming the target column is 'Loan_Status')
target_correlation = correlation_matrix['loan_status'].sort_values(ascending=False)

# Display correlations
print("Correlation with Loan Status:")
print(target_correlation)

# Plot correlations with the target variable
plt.figure(figsize=(10, 6))
target_correlation.drop('loan_status').plot(kind='bar')
plt.title('Correlation with Loan Status')
plt.ylabel('Correlation Coefficient')
plt.xlabel('Features')
#plt.show()

threshold = 0.03
correlation_matrix = df.corr()
high_corr_features = correlation_matrix.index[abs(correlation_matrix["loan_status"]) > threshold].tolist()
high_corr_features.remove("loan_status")
print(high_corr_features)
X = df[high_corr_features]
y = df["loan_status"]

# Scale and normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X.astype('float32'))

# Apply SMOTE to oversample the minority class
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
print(y_resampled.value_counts())

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Initialize the Random Forest classifier
rf = RandomForestClassifier(n_estimators=200, random_state=42)  # Use 100 trees by default

# Fit the classifier to the resampled training data
rf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.3f}")

# Generate classification report as a dictionary
report = classification_report(y_test, y_pred, output_dict=True)

# Convert the dictionary to a pandas DataFrame for better formatting
report_df = pd.DataFrame(report).transpose()

# Round the metrics to 3 decimal places
report_df = report_df.round(3)

# Display the formatted classification report
print("\nClassification Report:")
print(report_df)

# Compute the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)


# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=True, linewidths=1.5)

# Add labels and title
plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('True Label', fontsize=14)
plt.title('Confusion Matrix', fontsize=16)
plt.xticks(ticks=[0.5, 1.5], labels=['Negative', 'Positive'], fontsize=12)
plt.yticks(ticks=[0.5, 1.5], labels=['Negative', 'Positive'], fontsize=12)

# Show the plot
plt.show()
import pandas as pd

# Load the dataset
df = pd.read_csv('loan_da.csv')
print(df.shape)

print(df.columns[-2])
# Check the distribution of the target variable (assuming last column is the target)
target_column = df.columns[-2]
target_distribution = df[target_column].value_counts()

# Print the distribution
print("Target Distribution:")
print(target_distribution)
import uproot
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Assuming ROOT file contains a tree named "myTree"
filename = "path/to/your/data.root"
tree_name = "myTree"

# Use uproot to open the file and load the tree
with uproot.open(f"{filename}:{tree_name}") as tree:
    df = tree.arrays(library="pd")

# Here you might want to select specific columns, assuming they are named 'feature1', 'feature2', ..., 'target'
# df = df[['feature1', 'feature2', ..., 'target']]

# Optionally, preprocess your dataframe (e.g., normalization, feature engineering)
# For example, split your data into features and labels
X = df.iloc[:, :-1]  # assuming the last column is the label
y = df.iloc[:, -1]

# Splitting the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save processed data
np.save('X_train.npy', X_train_scaled)
np.save('X_test.npy', X_test_scaled)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)


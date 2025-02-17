import pandas as pd

# Load dataset
data = pd.read_csv("mnist_train.csv")

# Separate features and labels
labels = data['label']
features = data.drop('label', axis=1)

print("Features shape:", features.shape)
print("Labels shape:", labels.shape)

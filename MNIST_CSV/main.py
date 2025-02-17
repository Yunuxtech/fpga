import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

csv_file = "./mnist_train.csv"
data = pd.read_csv(csv_file)

# Inspect the data
print(data.head())
print(data.shape)

# Separate labels and images
labels = data.iloc[:, 0].values
images = data.iloc[:, 1:].values

for i in range(9):
    plt.subplot(330 + 1 + i)  # 3x3 grid for images
    plt.imshow(images[i].reshape(28, 28), cmap=plt.get_cmap('gray'))
    plt.title(f"Label: {labels[i]}")
    plt.axis('off')

plt.show()

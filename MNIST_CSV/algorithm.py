import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

# Load the MNIST CSV file
csv_file = "mnist_train.csv"  # Replace with your dataset path
data = pd.read_csv(csv_file)

# Inspect columns to verify the structure
print("Columns in CSV:", data.columns)

# Split labels and pixel data
labels = data.iloc[:, 0].values  # First column as labels
images = data.iloc[:, 1:].values  # Remaining columns as image data

# Normalize the pixel values (scale 0-255 to 0-1)
images = images / 255.0

# Sobel filter kernels
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

sobel_y = np.array([[-1, -2, -1],
                    [0,  0,  0],
                    [1,  2,  1]])

# Function to apply Sobel filter
def sobel_filter(image):
    grad_x = convolve(image, sobel_x)
    grad_y = convolve(image, sobel_y)
    grad = np.sqrt(grad_x**2 + grad_y**2)
    grad = grad / grad.max()  # Normalize to range [0, 1]
    return grad

# Select a sample image (e.g., the first image)
sample_image = images[1].reshape(28, 28)  # Reshape to 28x28

# Apply Sobel filter
filtered_image = sobel_filter(sample_image)

# Display the original and Sobel-filtered images
plt.figure(figsize=(10, 5))

# Original image
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(sample_image, cmap='gray')
plt.axis('off')

# Sobel-filtered image
plt.subplot(1, 2, 2)
plt.title("Sobel Filtered Image")
plt.imshow(filtered_image, cmap='gray')
plt.axis('off')

plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

def improved_sobel_filter(image, kernel_type='sobel', threshold=None, return_gradients=False):
    """
    Improved Sobel filter with additional features:
    - Multiple kernel options (Sobel, Scharr, Prewitt)
    - Thresholding capability
    - Better edge handling
    - Optional gradient component returns
    - Numerically stable normalization
    """
    # Kernel definitions
    kernels = {
        'sobel': {
            'x': np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]]),
            'y': np.array([[-1, -2, -1],
                           [0, 0, 0],
                           [1, 2, 1]])
        },
        'scharr': {
            'x': np.array([[-3, 0, 3],
                           [-10, 0, 10],
                           [-3, 0, 3]]),
            'y': np.array([[-3, -10, -3],
                           [0, 0, 0],
                           [3, 10, 3]])
        },
        'prewitt': {
            'x': np.array([[-1, 0, 1],
                           [-1, 0, 1],
                           [-1, 0, 1]]),
            'y': np.array([[-1, -1, -1],
                           [0, 0, 0],
                           [1, 1, 1]])
        }
    }
    
    # Validate kernel type
    if kernel_type not in kernels:
        raise ValueError(f"Invalid kernel type. Choose from {list(kernels.keys())}")
    
    # Get selected kernels
    kx = kernels[kernel_type]['x']
    ky = kernels[kernel_type]['y']
    
    # Convert image to float32 for better precision
    image = image.astype(np.float32)
    
    # Convolve with zero-padding (better edge handling)
    grad_x = convolve(image, kx, mode='constant', cval=0.0)
    grad_y = convolve(image, ky, mode='constant', cval=0.0)
    
    # Calculate gradient magnitude
    grad_mag = np.hypot(grad_x, grad_y)  # More accurate than sqrt(x² + y²)
    
    # Normalize to [0, 1] with numerical stability
    max_val = grad_mag.max()
    if max_val > 0:
        grad_mag /= max_val
    
    # Apply threshold if specified
    if threshold is not None:
        grad_mag = (grad_mag > threshold).astype(np.float32)
    
    # Return requested values
    if return_gradients:
        return grad_mag, grad_x, grad_y
    return grad_mag

# Load and preprocess MNIST data
def load_mnist(csv_path):
    data = pd.read_csv(csv_path)
    labels = data.iloc[:, 0].values
    images = data.iloc[:, 1:].values.reshape(-1, 28, 28) / 255.0
    return images, labels

# Visualization function
def plot_results(original, filtered, title="Filtered Image"):
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title(title)
    plt.imshow(filtered, cmap='gray')
    plt.axis('off')
    
    plt.show()

# Example usage
if __name__ == "__main__":
    # Load data
    images, labels = load_mnist("mnist_train.csv")
    
    # Select sample image
    sample_idx = 3  # Change this index to see different images
    original_image = images[sample_idx]
    
    # Apply improved Sobel filter with different options
    filtered_image, grad_x, grad_y = improved_sobel_filter(
        original_image,
        kernel_type='scharr',  # Try 'sobel', 'scharr', or 'prewitt'
        threshold=0.2,
        return_gradients=True
    )
    
    # Plot results
    plot_results(original_image, filtered_image, "Scharr Filter + Threshold")
    
    # Optional: Plot gradient components
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1).imshow(grad_x, cmap='gray').set_title('Gradient X')
    plt.subplot(1, 3, 2).imshow(grad_y, cmap='gray').set_title('Gradient Y')
    plt.subplot(1, 3, 3).imshow(filtered_image, cmap='gray').set_title('Combined Magnitude')
    plt.show()
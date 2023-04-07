import cv2
import numpy as np

def sample_pixel_coordinates(image, temperature=1.0, samples=1):
    # Convert to grayscale if necessary
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Normalize the grayscale image
    gray = gray / 255.0

    # Compute the gradients using the Sobel filter
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Compute the gradient magnitude
    gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    # Normalize the gradient magnitude
    gradient_magnitude = gradient_magnitude / np.max(gradient_magnitude)

    # Compute the temperature-adjusted softmax probabilities
    probabilities = np.power(gradient_magnitude, 1 / temperature)
    probabilities /= np.sum(probabilities)

    # Sample coordinates
    flattened_probabilities = probabilities.flatten()
    sampled_indices = np.random.choice(flattened_probabilities.size, size=samples, p=flattened_probabilities,
        replace=False
    )
    sampled_coordinates = np.array(np.unravel_index(sampled_indices, probabilities.shape)).T

    return sampled_coordinates

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    image = cv2.imread('assets/images/chibi.jpg')
    # Compute the gradients using the Sobel filter
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Compute the gradient magnitude
    gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    gradient_magnitude = gradient_magnitude / np.max(gradient_magnitude)

    plt.imshow(gradient_magnitude)
    # plt.scatter(coordinates[:, 1], coordinates[:, 0], c='r', s=1)
    plt.show()
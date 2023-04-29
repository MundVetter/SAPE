import numpy as np
import imageio
import os

def save_images_from_numpy_array(np_file_path, key, output_dir, file_format='jpg'):
    # Load the NumPy array from the file
    images = np.load(np_file_path)[key]

    # Check if the output directory exists, if not, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over images and save them
    for i, img in enumerate(images):
        # Set the output file path
        output_path = os.path.join(output_dir, f'image_{i:03d}.{file_format}')

        # Save the image to the output file
        imageio.imwrite(output_path, img)

# Example usage
np_file_path = 'assets/natural_images/data_div2k.npz'
output_dir = 'assets/natural_images'
save_images_from_numpy_array(np_file_path, 'test_data', output_dir, file_format='jpg')
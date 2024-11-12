# \textbf{Q: Viết chương trình nén/giải nén bằng kĩ thuật Fractal?}\\
# A: Dưới đây là một chương trình Python để nén và giải nén ảnh bằng kỹ thuật Fractal sử dụng Hệ Thống Hàm Lặp (IFS - Iterated Function System).

import numpy as np
from PIL import Image

def ifs_compress(image, num_iterations):
    # Grayscale
    image = image.convert('L')

    # Get dimensions
    width, height = image.size

    # Initialize IFS codebook
    codebook = []

    # Iterate over image pixels
    for y in range(height):
        for x in range(width):
            # Get pixel value
            pixel = image.getpixel((x, y))

            # Apply affine transformation to pixel
            for _ in range(num_iterations):
                pixel = (pixel * 0.5) + (x * 0.2) + (y * 0.3)

            # Quantize pixel value
            pixel = int(pixel)

            # Add pixel to codebook
            codebook.append(pixel)

    return codebook

def ifs_decompress(codebook, width, height, num_iterations):
    # Initialize decompressed image
    image = np.zeros((height, width), dtype=np.int64)

    # Iterate over codebook
    for i, pixel in enumerate(codebook):
        # Get x and y coordinates from codebook index
        x = i % width
        y = i // width

        # Apply inverse affine transformation to pixel
        for _ in range(num_iterations):
            pixel = (pixel - (x * 0.2) - (y * 0.3)) / 0.5

        # Set pixel value in decompressed image
        image[y, x] = int(pixel)

    # Clip pixel values to valid range
    image = np.clip(image, 0, 255).astype(np.uint8)

    return Image.fromarray(image)

# Load image
image = Image.open('D:\\lena.png')

# IFS Compress
codebook = ifs_compress(image, 5)

# Save compressed image
np.save('D:\\compressed_image.npy', codebook)

# Load compressed image from file
loaded_codebook = np.load('compressed_image.npy')

# IFS Decompress
decompressed_image = ifs_decompress(loaded_codebook, image.width, image.height, 5)

# Save decompressed image
decompressed_image.save('D:\\decompressed_image.png')
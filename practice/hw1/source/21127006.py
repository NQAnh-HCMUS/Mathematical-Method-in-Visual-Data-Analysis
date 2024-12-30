from re import A
import numpy as np
from PIL import Image, ImageEnhance
import cv2

# Get input & output paths
input_path = input("Insert input file path (Example: D:\\lena.png): ")
print(input_path)
output_path = input(
    "Insert output folder path and name (Example: D:\\output\\lena.png): "
)
print(output_path)
# Load image
grayscale_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
rgb_image = cv2.imread(input_path)


print("\n- 1 & 2. Implement affine transformations: translation, rotation, scaling, reflection/mirroring for greyscale images & RGB images.")
def translation(image, x, y):
    # Get image dimensions
    rows = image.shape[1]
    cols = image.shape[0]

    # Create translation matrix
    matrix = np.float32([[1, 0, x], [0, 1, y]])

    # Use warpAffine to transform the image using the matrix
    output = cv2.warpAffine(image, matrix, (cols, rows))
    return output


x = int(input("Input translation value x: "))
y = int(input("Input translation value y: "))
grayscale_translated = translation(grayscale_image, x, y)
rgb_translated = translation(rgb_image, x, y)
# Save image
cv2.imwrite(output_path + "_grayscale_translated.jpg", grayscale_translated)
print(output_path + "_grayscale_translated.jpg")
cv2.imwrite(output_path + "_rgb_translated.jpg", rgb_translated)
print(output_path + "_rgb_translated.jpg")


def rotation(image, angle):
    # Get image dimensions
    rows = image.shape[1]
    cols = image.shape[0]

    # Get image center
    center = (cols // 2, rows // 2)

    # Create rotation matrix
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Apply the rotation matrix to the image
    output = cv2.warpAffine(image, matrix, (cols, rows))
    return output


angle = int(input("Input rotation angle: "))
grayscale_rotated = rotation(grayscale_image, angle)
rgb_rotated = rotation(rgb_image, angle)
# Save image
cv2.imwrite(output_path + "_grayscale_rotated.jpg", grayscale_rotated)
print(output_path + "_grayscale_rotated.jpg")
cv2.imwrite(output_path + "_rgb_rotated.jpg", rgb_rotated)
print(output_path + "_rgb_rotated.jpg")


def scaling(image, scale):
    # Get image dimensions
    rows = image.shape[1]
    cols = image.shape[0]

    # Create scaling matrix
    matrix = np.float32([[scale, 0, 0], [0, scale, 0]])

    # Apply the scaling matrix to the image
    output = cv2.warpAffine(image, matrix, (rows, cols))

    return output


scale = float(input("Input scaling value: "))
grayscale_scaled = scaling(grayscale_image, scale)
rgb_scaled = scaling(rgb_image, scale)
# Save image
cv2.imwrite(output_path + "_grayscale_scaled.jpg", grayscale_scaled)
cv2.imwrite(output_path + "_rgb_scaled.jpg", rgb_scaled)


def reflection(image, axis):
    # Get image dimensions
    rows = image.shape[1]
    cols = image.shape[0]

    if axis == 0:
        # Create horizontal reflection/mirroring matrix
        matrix = np.float32([[1, 0, 0], [0, -1, cols]])
    elif axis == 1:
        # Create vertical reflection/mirroring matrix
        matrix = np.float32([[-1, 0, rows], [0, 1, 0]])

    # Apply the reflection/mirroring matrix to the image
    output = cv2.warpAffine(image, matrix, (rows, cols))

    return output


axis = input("Input reflection axis (0: horizontal, 1: vertical): ")
grayscale_reflected = reflection(grayscale_image, 0)
rgb_reflected = reflection(rgb_image, 0)
# Save image
cv2.imwrite(output_path + "_grayscale_reflected.jpg", grayscale_reflected)
cv2.imwrite(output_path + "_rgb_reflected.jpg", rgb_reflected)


print("\n- 3. Implement fractal image compresion for greyscale images with affine transformations.")
def ifs_compress(image, num_iterations):
    # Grayscale
    image = image.convert("L")

    # Get dimensions
    width, height = image.size

    # Initialize IFS codebook
    codebook = []

    # Iterate over each pixel
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


# Load image
PIL_image = Image.open(input_path)
num_iterations = int(input("Input number of iterations (Example: 5): "))
# IFS Compress
codebook = ifs_compress(PIL_image, num_iterations)
# Save compressed image
np.save(output_path + "_compressed.npy", codebook)


print("\n- 4. Implement fractal image compresion for greyscale images with affine transformations, and contrast + brightness.")
def change_contrast(image, level):
    factor = (259 * (level + 255)) / (255 * (259 - level))

    def contrast(c):
        return 128 + factor * (c - 128)

    return image.point(contrast)


def ifs_compress_with_contrastNbrightness(image, num_iterations, contrast, brightness):
    # Grayscale
    image = image.convert("L")

    # Get dimensions
    width, height = image.size

    # Apply contrast and brightness
    image = change_contrast(image, contrast)

    image = ImageEnhance.Brightness(image)
    image = image.enhance(brightness)

    # Initialize IFS codebook
    codebook = []

    # Iterate over each pixel
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


# Load image
PIL_image = Image.open(input_path)
# Define the contrast and brightness values
contrast = float(input("Input contrast value (Example: 1.5): "))
brightness = float(input("Input brightness value (Example: 30): "))
num_iterations = int(input("Input number of iterations (Example: 5): "))
# IFS Compress with contrast & brightness
codebook = ifs_compress_with_contrastNbrightness(
    PIL_image, num_iterations, contrast, brightness
)
# Save compressed image
np.save(
    output_path + "_compressed_with_contrastNbrightness.npy",
    codebook,
)


print("\n- 5. (bonus) Implement fractal image compresion for RGB images")
def ifs_compress(image, num_iterations):
    # Get dimensions
    width, height = image.size

    # Initialize IFS codebook
    codebook = []

    # Iterate over each pixel
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


# Load image
PIL_image = Image.open(input_path).convert("RGB")
num_iterations = int(input("Input number of iterations (Example: 5): "))
# Split RGB into 3 Lists
r, g, b = PIL_image.split()
# IFS Compress to each list
Red = ifs_compress(r, num_iterations)
Green = ifs_compress(g, num_iterations)
Blue = ifs_compress(b, num_iterations)
# Combine into RGB List
output = Red + Green + Blue
# Save compressed image
np.save(output_path + "_compressed_RGB.npy", codebook)

# P/s: report the detailed experimental results for each tasks, with comments or explanations.
# LIBRARIES
import os
from pydoc import describe
import numpy as np
import faiss
import cv2

import matplotlib

matplotlib.use("TkAgg", force=True)
from matplotlib import pyplot as plt

print("Switched to:", matplotlib.get_backend())

# MODELS
import torch
import torchvision.models as models

# DATASET
import kagglehub

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


########################################


# Get a list of all image files in the dataset
def get_image_list(path):
    image_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                image_list.append(os.path.join(root, file))
    return image_list


# SIFT
def method_SIFT(image_list):
    sift = cv2.SIFT_create()
    SIFT_descriptors = []
    for image_file in image_list:
        image = cv2.imread(image_file, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(image, None)
        if descriptors is not None:
            SIFT_descriptors.append(descriptors)

    return SIFT_descriptors


# ORB
def method_ORB(image_list):
    orb = cv2.ORB_create()
    ORB_descriptors = []
    for image_file in image_list:
        image = cv2.imread(image_file, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = orb.detectAndCompute(image, None)
        if descriptors is not None:
            ORB_descriptors.append(descriptors)

    return ORB_descriptors


# ResNet
def method_ResNet(image_list):
    # Load ResNet model
    model = models.resnet18(pretrained=True)
    model.eval()

    # Compute descriptors
    ResNet_descriptors = []
    for image_file in image_list:
        image = cv2.imread(image_file, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (224, 224))
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)
        image = np.transpose(image, (0, 3, 1, 2))
        with torch.no_grad():
            features = model(torch.from_numpy(image).to(torch.float32)).numpy()
        ResNet_descriptors.append(features)

    return ResNet_descriptors


########################################

# # Create a Faiss index with Inner product
# index = faiss.IndexFlatIP(descriptors.shape[1])  # Inner product


def METRIC_L2(path, descriptors, no_neighbors, outputFile):
    # Stack descriptors into a single matrix
    if descriptors:
        descriptors = np.vstack(descriptors)
    else:
        print("No descriptors found.")
        return

    # Create a Faiss index
    INDEX = faiss.IndexFlatL2(descriptors.shape[1])  # L2 distance

    # Add descriptors to the index
    INDEX.add(descriptors)

    # Search for similar descriptors
    if descriptors.shape[0] > 0:
        query_desc = descriptors[0]  # Use the first descriptor as a query
        DISTANCE, I = INDEX.search(
            query_desc.reshape(1, -1), k=no_neighbors
        )  # Search for k nearest neighbors

        print("Similar descriptors:")
        for i, (dist, idx) in enumerate(zip(DISTANCE[0], I[0])):
            print(f"Image {idx}: distance={dist:.2f}")
        # Export distances to a .txt file
        outputFile = open("output.txt", "a")
        for i, (dist, idx) in enumerate(zip(DISTANCE[0], I[0])):
            outputFile.write(f"{idx},{dist:.2f}\n")
    else:
        print("No descriptors to search.")


def METRIC_Inner(path, descriptors, no_neighbors, outputFile):
    # Stack descriptors into a single matrix
    if descriptors:
        descriptors = np.vstack(descriptors)
    else:
        print("No descriptors found.")
        return

    # Create a Faiss index
    INDEX = faiss.IndexFlatIP(descriptors.shape[1])

    # Add descriptors to the index
    INDEX.add(descriptors)

    # Search for similar descriptors
    if descriptors.shape[0] > 0:
        query_desc = descriptors[0]  # Use the first descriptor as a query
        DISTANCE, I = INDEX.search(
            query_desc.reshape(1, -1), k=no_neighbors
        )  # Search for k nearest neighbors

        print("Similar descriptors:")
        for i, (dist, idx) in enumerate(zip(DISTANCE[0], I[0])):
            print(f"Image {idx}: distance={dist:.2f}")
        # Export distances to a .txt file
        outputFile = open("output.txt", "a")
        for i, (dist, idx) in enumerate(zip(DISTANCE[0], I[0])):
            outputFile.write(f"{idx},{dist:.2f}\n")
    else:
        print("No descriptors to search.")


def METRIC_Cosine(path, descriptors, no_neighbors, outputFile):
    # Stack descriptors into a single matrix
    if descriptors:
        descriptors = np.vstack(descriptors)
    else:
        print("No descriptors found.")
        return

    # Create a Faiss index
    INDEX = faiss.IndexFlatIP(descriptors.shape[1])
    faiss.normalize_L2(descriptors)

    # Add descriptors to the index
    INDEX.add(descriptors)

    # Search for similar descriptors
    if descriptors.shape[0] > 0:
        query_desc = descriptors[0]  # Use the first descriptor as a query
        DISTANCE, I = INDEX.search(
            query_desc.reshape(1, -1), k=no_neighbors
        )  # Search for k nearest neighbors

        print("Similar descriptors:")
        for i, (dist, idx) in enumerate(zip(DISTANCE[0], I[0])):
            print(f"Image {idx}: distance={dist:.2f}")
        # Export distances to a .txt file
        outputFile = open("output.txt", "a")
        for i, (dist, idx) in enumerate(zip(DISTANCE[0], I[0])):
            outputFile.write(f"{idx},{dist:.2f}\n")
    else:
        print("No descriptors to search.")


def METRIC_Mahalanobis(path, descriptors, no_neighbors, outputFile):
    # Stack descriptors into a single matrix
    if descriptors:
        descriptors = np.vstack(descriptors)
    else:
        print("No descriptors found.")
        return

    # Create a Faiss index
    INDEX = faiss.IndexPreTransform(
        faiss.IndexFlatL2(descriptors.shape[1]),
        faiss.MahalanobisVectorTransform(np.cov(descriptors, rowvar=False)),
    )
    # Add descriptors to the index
    INDEX.add(descriptors)

    # Search for similar descriptors
    if descriptors.shape[0] > 0:
        query_desc = descriptors[0]  # Use the first descriptor as a query
        DISTANCE, I = INDEX.search(
            query_desc.reshape(1, -1), k=no_neighbors
        )  # Search for k nearest neighbors

        print("Similar descriptors:")
        for i, (dist, idx) in enumerate(zip(DISTANCE[0], I[0])):
            print(f"Image {idx}: distance={dist:.2f}")
        # Export distances to a .txt file
        outputFile = open("output.txt", "a")
        for i, (dist, idx) in enumerate(zip(DISTANCE[0], I[0])):
            outputFile.write(f"{idx},{dist:.2f}\n")
    else:
        print("No descriptors to search.")


########################################


# Example
# Download latest version of the dataset
# path = kagglehub.dataset_download("crawford/cat-dataset")
# print("Path to dataset files:", path)
# path = "C:\\Users\\quoca\\.cache\\kagglehub\\datasets\\crawford\\cat-dataset\\versions\\2"
path = "D:\\dataset"

print("Path to dataset files:", path)

# Create a CSV file
outputFile = open("output.txt", "w")
outputFile.write("Image,Distance\n")

# Prepare descriptors
SIFTdescriptors = method_SIFT(get_image_list(path))
ORBdescriptors = method_ORB(get_image_list(path))
# ResNetdescriptors = method_ResNet(get_image_list(path))


# Set values
no_neighbors = 5

# Export distances
METRIC_L2(path, SIFTdescriptors, no_neighbors, outputFile)
METRIC_L2(path, ORBdescriptors, no_neighbors, outputFile)

METRIC_Inner(path, SIFTdescriptors, no_neighbors, outputFile)
METRIC_Inner(path, ORBdescriptors, no_neighbors, outputFile)

# METRIC_Cosine(path, SIFTdescriptors, no_neighbors,outputFile)
# METRIC_Cosine(path, ORBdescriptors, no_neighbors,outputFile)

# METRIC_Mahalanobis(path, SIFTdescriptors, no_neighbors,outputFile)
# METRIC_Mahalanobis(path, ORBdescriptors, no_neighbors,outputFile)

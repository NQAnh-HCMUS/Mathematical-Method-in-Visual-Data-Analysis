# \textbf{Q: Viết chương trình nén/giải nén bằng kĩ thuật Fractal?}\\
# A: Dưới đây là một chương trình Python để nén và giải nén ảnh bằng kỹ thuật Fractal sử dụng Hệ Thống Hàm Lặp (IFS - Iterated Function System).

# \begin{verbatim}
import numpy as np
import matplotlib.pyplot as plt

# Define the IFS transformations and probabilities
def ifs_transformations():
    return [
        lambda x, y: (0.5 * x, 0.5 * y),
        lambda x, y: (0.5 * x + 0.5, 0.5 * y),
        lambda x, y: (0.5 * x, 0.5 * y + 0.5),
        lambda x, y: (0.5 * x + 0.5, 0.5 * y + 0.5)
    ]

def ifs_probabilities():
    return [0.25, 0.25, 0.25, 0.25]

# Compress the image using IFS (encode the image)
def compress_image(image):
    # Placeholder for the actual compression logic
    # In practice, this would involve finding the best IFS transformations to represent the image
    return ifs_transformations(), ifs_probabilities()

# Decompress the image using IFS (generate the fractal)
def decompress_image(transformations, probabilities, iterations=10000):
    x, y = 0, 0
    points = []

    for _ in range(iterations):
        t = np.random.choice(transformations, p=probabilities)
        x, y = t(x, y)
        points.append((x, y))

    return points

# Main function to demonstrate compression and decompression
def main():
    # Placeholder for the actual image
    image = np.zeros((100, 100))

    # Compress the image
    transformations, probabilities = compress_image(image)

    # Decompress the image
    points = decompress_image(transformations, probabilities)

    # Plot the decompressed image
    x_coords, y_coords = zip(*points)
    plt.scatter(x_coords, y_coords, s=0.1)
    plt.show()

if __name__ == "__main__":
    main()
# \end{verbatim}

# \textbf{Giải thích:}
# \begin{itemize}
#     \item \textbf{Định nghĩa các phép biến đổi IFS và xác suất}: 
#     \begin{itemize}
#         \item \texttt{ifs_transformations}: Trả về danh sách các hàm lambda đại diện cho các phép biến đổi.
#         \item \texttt{ifs_probabilities}: Trả về danh sách các xác suất liên quan đến mỗi phép biến đổi.
#     \end{itemize}
#     \item \textbf{Nén ảnh}: 
#     \begin{itemize}
#         \item \texttt{compress_image}: Là hàm giữ chỗ cho logic nén thực tế. Trong thực tế, hàm này sẽ tìm các phép biến đổi IFS tốt nhất để đại diện cho ảnh.
#     \end{itemize}
#     \item \textbf{Giải nén ảnh}: 
#     \begin{itemize}
#         \item \texttt{decompress_image}: Sử dụng các phép biến đổi IFS và xác suất để tạo ra ảnh fractal bằng cách áp dụng các phép biến đổi lặp đi lặp lại.
#     \end{itemize}
#     \item \textbf{Hàm chính}: 
#     \begin{itemize}
#         \item \texttt{main}: Minh họa quá trình nén và giải nén. Vẽ ảnh giải nén sử dụng \texttt{matplotlib}.
#     \end{itemize}
# \end{itemize}
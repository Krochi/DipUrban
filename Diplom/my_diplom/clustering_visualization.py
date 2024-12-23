import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from image_features_extraction import extract_features_pytorch, extract_features_tensorflow

image_paths = [
    'media/images/image1.png',
    'media/images/image2.png',
    'media/images/image3.png',
    'media/images/image4.png',
]

features_pytorch = np.array([extract_features_pytorch(image_path) for image_path in image_paths])
features_tensorflow = np.array([extract_features_tensorflow(image_path) for image_path in image_paths])

kmeans_pytorch = KMeans(n_clusters=3, random_state=42)
kmeans_tensorflow = KMeans(n_clusters=3, random_state=42)

labels_pytorch = kmeans_pytorch.fit_predict(features_pytorch)
labels_tensorflow = kmeans_tensorflow.fit_predict(features_tensorflow)


plt.figure(figsize=(12, 6))


plt.subplot(1, 2, 1)
plt.scatter(features_pytorch[:, 0], features_pytorch[:, 1], c=labels_pytorch, cmap='viridis')
plt.plot(features_pytorch[:, 0], features_pytorch[:, 1], linestyle='-', color='grey')
plt.title('Кластеризация с использованием PyTorch')
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')


plt.subplot(1, 2, 2)
plt.scatter(features_tensorflow[:, 0], features_tensorflow[:, 1], c=labels_tensorflow, cmap='viridis')
plt.plot(features_tensorflow[:, 0], features_tensorflow[:, 1], linestyle='-', color='grey')
plt.title('Кластеризация с использованием TensorFlow')
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')

plt.tight_layout()
plt.show()


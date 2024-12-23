from torchvision.models import ResNet50_Weights, resnet18
from .models import Image
from sklearn.cluster import KMeans
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image as PILImage
from torchvision import models
import os

resnet = models.resnet18(weights='IMAGENET1K_V1')
resnet.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

imagenet_classes = ResNet50_Weights.IMAGENET1K_V1.meta["categories"]

def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f"Файл {image_path} не найден.")
        return None, None

    image = PILImage.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = resnet(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
    class_idx = torch.argmax(probabilities).item()
    class_label = imagenet_classes[class_idx]
    return class_idx, class_label

def extract_features(image_path):
    if not os.path.exists(image_path):
        print(f"Файл {image_path} не найден.")
        return None

    img = PILImage.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        features = resnet(img)
    return features.flatten().detach().numpy()


image_features = []
images = Image.objects.all()
for image in images:
    features = extract_features(image.image.path)
    if features is not None:
        image_features.append(features)

image_features = np.array(image_features)

n_samples = len(image_features)
n_clusters = 3

if n_samples >= n_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(image_features)

    labels = kmeans.labels_

    for image, label in zip(images, labels):
        image.cluster = label
        image.save()
else:
    print(f"Количество образцов ({n_samples}) меньше, чем количество кластеров ({n_clusters}). Уменьшите количество кластеров.")

def cluster_images(n_clusters=3):
    images = Image.objects.all()
    image_paths = [image.image.path for image in images]

    features = [extract_features(image_path) for image_path in image_paths if extract_features(image_path) is not None]
    features = np.array(features)

    if len(features) >= n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(features)
        labels = kmeans.labels_

        for image, label in zip(images, labels):
            image.cluster = label
            image.save()
        print(f"KMeans labels: {kmeans.labels_}")
    else:
        print(f"Количество образцов ({len(features)}) меньше, чем количество кластеров ({n_clusters}). Уменьшите количество кластеров.")

"""
В этом файле используется модель ResNet18 для предсказания класса объектов на изображениях, 
извлечения признаков изображений, а также их кластеризации с использованием алгоритма KMeans.

Основные функции:
1. predict_image(image_path)**:
   -- Принимает путь к изображению.
   -- Загружает изображение, преобразует его в тензор и передает в модель ResNet18 для предсказания класса.
   -- Возвращает индекс и название предсказанного класса (например, "кошка", "собака", "самолёт").

2. extract_features(image_path)**:
   - Извлекает признаки изображения, которые будут использованы для кластеризации.
   - Преобразует изображение в тензор и пропускает его через модель ResNet18.
   - Возвращает вектор признаков, представляющий изображение.

3. cluster_images(n_clusters=3)**:
   -- Кластеризует изображения с использованием алгоритма KMeans на основе извлеченных признаков.
   -- Принимает число кластеров, которое указывает на количество групп для кластеризации.
   -- Сохраняет результат кластеризации в базе данных, присваивая метку кластера каждому изображению.

Процесс кластеризации:
-- Изображения из базы данных загружаются и их признаки извлекаются с помощью функции `extract_features`.
-- Эти признаки используются в алгоритме KMeans для группировки изображений в заданное количество кластеров.
-- Каждому изображению присваивается метка кластера, которая сохраняется в базе данных.

Зависимости:
-- `torch`: Для работы с моделью ResNet18.
-- `torchvision`: Для загрузки и применения трансформаций к изображениям.
-- `numpy`: Для работы с массивами данных и признаками.
-- `sklearn.cluster.KMeans`: Для кластеризации изображений.
-- `PIL.Image`: Для загрузки и обработки изображений.
-- `models.Image`: Модель Django для работы с изображениями в базе данных.
"""
# Обработка изображений и кластеризация с помощью модели ResNet18
from torchvision.models import ResNet50_Weights, resnet18
from .models import Image
from sklearn.cluster import KMeans
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image as PILImage
from torchvision import models
import os

# Инициализация модели ResNet18
resnet = models.resnet18(weights='IMAGENET1K_V1')
resnet.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Получение списка класса из модели ResNet
imagenet_classes = ResNet50_Weights.IMAGENET1K_V1.meta["categories"]

# Предсказание класса изображения, Возврат результата
def predict_image(image_path):
    """
        Параметры:
        -- image_path (str): Путь к изображению.

        Возвращает:
        -- class_idx (int): Индекс предсказанного класса.
        -- class_label (str): Название предсказанного класса.
        """
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
# Извлечение признаков изображения с использованием модели ResNet18
def extract_features(image_path):
    """
        Параметры:
        -- image_path (str): Путь к изображению.

        Возвращает:
        -- features (np.ndarray): Вектор признаков изображения.
        """
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

# Кластерезация изображений с помощбю KMeans.
def cluster_images(n_clusters=3): 
    """
        Параметры:
        -- n_clusters (int): Количество кластеров для алгоритма KMeans.
        """
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

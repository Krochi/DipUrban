# В файле реализуется логика предсказания классов объектов на изображениях с использованием ResNet50
"""
Файл содержит логику для загрузки модели ResNet50, преобразования изображений для обработки в модели и
получения предсказанных классов для изображений.

Модель ResNet50 предобучена на наборе данных ImageNet и используется для классификации объектов на изображениях.

Основные функции:
-- `transform_image`: Преобразует изображение в формат, для использования моделью для предсказания.
-- `get_class_label`: Возвращает метку класса по индексу предсказания.
-- `predict_image`: Возвращает индекс предсказанного класса для изображения.

Поля:
-- Модель ResNet50 загружается с использованием весов 'IMAGENET1K_V1', которые обучены на большом наборе изображений.
-- Для преобразования изображений используется стандартный набор трансформаций, таких как изменение размера, центрирование и нормализация.

Зависимости:
-- `torch`: Для работы с моделями глубокого обучения.
-- `torchvision`: Для использования предобученной модели ResNet50.
-- `PIL`: Для обработки изображений.
"""
import torch
from torchvision import models, transforms
from PIL import Image

# Загрузка предобученной модели ResNet50
model = models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V1')
model.eval()

# Преобразование изображений в тензор и создание последовательности
def transform_image(image_path):
    """
        Преобразует изображение в формат, подходящий для модели ResNet50.
        Шаги преобразования:
        1. Изменение размера до 256 пикселей.
        2. Центрирование и обрезка до 224 пикселей.
        3. Преобразование в тензор.
        4. Нормализация с использованием статистик ImageNet.

        Параметры:
            image_path (str): Путь к изображению для преобразования.

        Возвращает:
            torch.Tensor: Преобразованное изображение в виде тензора.
        """
    transformation = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path)
    image = transformation(image).unsqueeze(0)
    return image

 # Название классов согласно ResNet
def get_class_label(predicted_idx): 
    """
        Возвращает название класса по его индексу.
        Предполагается, что файл 'imagenet_classes.txt' содержит список классов в формате строки.
        Которые будут ассоциированы с индексами для последующего использования.

        Параметры:
           -- predicted_idx (int): Индекс предсказанного класса.

        Возвращает:
           -- str: Название класса для переданного индекса.
        """
    labels = {idx: label for idx, label in enumerate(open('imagenet_classes.txt').read().splitlines())}
    return labels.get(predicted_idx, 'Unknown')

# Получение индекса класса с использованием модели ResNet50
def predict_image(image):
    """
        Предсказывает класс изображения.
        Параметры:
           -- image (str): Путь к изображению, которое необходимо классифицировать.

        Возвращает:
           -- int: Индекс класса.
        """
    image_tensor = transform_image(image)
    with torch.no_grad():
        output = model(image_tensor)
    _, predicted_idx = torch.max(output, 1)
    return predicted_idx.item()

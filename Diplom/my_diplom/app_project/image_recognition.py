# Скрипт для распознавания изображений с помощью ResNet50 и библиотеки PyTorch
"""
Основные этапы:
1. Загрузка предобученной модели ResNet50.
2. Преобразование изображений.
3. Прогнозирование класса изображения и вывод его вероятностей.
4. Возвращение наименования класса, его индекса и вероятностей для всех классов.

Скрипт включает несколько функций:
--`predict_image`: Классификация одного изображения с выводом предсказанного класса.
--`preprocess_image`: Предобработка изображения для подачи в модель.
--`predict_image_probabilities`: Получение вероятности для всех классов на основе изображения.

Зависимости:
-- PyTorch
-- torchvision
-- PIL (Pillow)

"""
import torch
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image
import torch.nn.functional as F
from PIL import Image as PILImage
import logging


# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.eval()
    logger.info("Model ResNet50 loaded")
except Exception as e:
    logger.error("Model ResNet50 not loaded")
    raise

# Предобработка изображений
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

imagenet_classes = ResNet50_Weights.IMAGENET1K_V1.meta["categories"]
# Путь к изображению, классификация и возвращение навания и класса
def predict_image(image_path):
    """
        Классификация одного изображения с выводом предсказанного класса.
        Параметры:
            image_path (str): Путь к изображению, которое необходимо классифицировать.

        Возвращаем:
            -- class_idx (int): Индекс предсказанного класса.
            -- class_label (str): Название предсказанного класса.
        """
    try:
        # Проверка существования файла
        if not image_path:
            logger.error("No image path provided")
            return None, None
        image = PILImage.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        class_idx = torch.argmax(probabilities).item()
        class_label = imagenet_classes[class_idx]
        logger.info(f"Изображение успешно классифицированно: {class_label}(format(индекс: {class_idx})")
        return class_idx, class_label  # Возвращается кортеж

    except FileNotFoundError:
        logger.error(f"Файл не найден: {image_path}")
        return None, None
    except PILImage.UnidentifiedImageError:
        logger.error(f"Некорректный формат изображения: {image_path}")
    except Exception as e:
        logger.error(f"Ошибка при классификации изображения: {image_path}: {e}")
        return None, None


# Загрузка и преобразование изображения перед передачей в ResNet50
def preprocess_image(image_path):
    """
       Параметры:
           image_path (str): Путь к изображению для предобработки.

       Возвращаеv:
           Tensor: Тензор изображения, готовый для подачи в модель.
    """
    try:
        if not image_path:
            logger.error("Путь к изображению не указан")
            return None, None

        image = Image.open(image_path).convert('RGB')
        return transform(image).unsqueeze(0)

    except FileNotFoundError:
        logger.error(f"Файл не найден: {image_path}")
        return None
    except PILImage.UnidentifiedImageError:
        logger.error(f"Некорректный формат изображения: {image_path}")
        return None
    except Exception as e:
        logger.error(f"Ошибка при обработке изображения: {image_path}: {e}")
        return None


 # Возвращение вероятности для всех классов в виде словаря, где ключи - это имена классов, а значение - вероятности для классов.
def predict_image_probabilities(image_path):
    """
        Параметры:
            image_path (str): Путь к изображению для анализа.

        Возвращает:
            predictions (dict).
        """
    try:
        if not image_path:
            logger.error("Путь к изображению не указан.")
            return {}

        input_tensor = PILImage.open(image_path).convert("RGB")
        input_tensor = transform(input_tensor).unsqueeze(0)

        with torch.no_grad(): # Предсказание и вычисление вероятности для всех классов
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)


        probabilities = probabilities.squeeze().tolist()
        predictions = {imagenet_classes[i]: round(prob * 100, 2) for i, prob in enumerate(probabilities) if prob > 0.0}
        logger.info(f"Вероятности для изображения {image_path} успешно рассчитаны.")
        return predictions

    except FileNotFoundError:
        logger.error(f"Файл не найден: {image_path}")
        return {}

    except PILImage.UnidentifiedImageError:
        logger.error(f"Некорректный формат изображения: {image_path}")
        return {}
    except Exception as e:
        logger.error(f"Ошибка при расчете вероятностей для изображения {image_path}: {e}")
        return {}
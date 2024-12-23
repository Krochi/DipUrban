# DipUrban

# Сравнение различных библиотек для машинного обучения: scikit-learn, TensorFlow и PyTorch

## Описание проекта
Этот проект направлен на сравнение производительности и удобства использования различных библиотек для машинного обучения: scikit-learn, TensorFlow и PyTorch. 

## Структура проекта
Проект включает следующие основные файлы и каталоги:
- `image_features_extraction.py` - скрипт для извлечения признаков изображений с помощью PyTorch и TensorFlow.
- `clustering_visualization.py` - скрипт для выполнения кластеризации и визуализации результатов.
- `media/images/` - каталог, содержащий изображения для обработки.

## Установка зависимостей
Перед началом проверьте, что у вас установлены все необходимые библиотеки. Для этого выполните команду:
```bash
pip install matplotlib torch torchvision scikit-learn tensorflow

Использование
Скопируйте изображения в каталог media/images/.

Обновите пути к изображениям в файле clustering_visualization.py:

image_paths = [
    'media/images/image1.png',
    'media/images/image2.png',
    'media/images/image3.png'
    # Добавьте пути ко всем вашим изображениям
]

Запустите скрипт clustering_visualization.py:

python clustering_visualization.py

Основные функции
image_features_extraction.py
Этот файл содержит функции для извлечения признаков изображений с использованием PyTorch и TensorFlow:

extract_features_pytorch(image_path) - извлечение признаков с использованием модели ResNet18 в PyTorch.

extract_features_tensorflow(image_path) - извлечение признаков с использованием модели ResNet50 в TensorFlow.

clustering_visualization.py
Этот файл выполняет следующие функции:

Извлечение признаков изображений с использованием функций из image_features_extraction.py.

Кластеризация изображений с использованием KMeans из scikit-learn.

Визуализация результатов кластеризации с использованием Matplotlib.

Анализ кода
Извлечение признаков
Файл image_features_extraction.py содержит функции для извлечения признаков изображений с использованием моделей ResNet18 и ResNet50 в PyTorch и TensorFlow соответственно. Преобразование изображений включает изменение их размера, нормализацию и преобразование в тензоры для дальнейшей обработки.

Кластеризация и визуализация
Файл clustering_visualization.py извлекает признаки изображений с помощью функций из image_features_extraction.py, выполняет кластеризацию с использованием алгоритма KMeans из scikit-learn и визуализирует результаты с использованием Matplotlib. Графики показывают распределение изображений по кластерам на основе извлеченных признаков.

# DipUrban

Сравнение различных библиотек для машинного обучения: scikit-learn, TensorFlow и PyTorch
Данный проект предназначен для сравнения производительности и удобства использования библиотек машинного обучения: scikit-learn, TensorFlow и PyTorch, с акцентом на задачи классификации изображений и кластеризации. Реализованы алгоритмы классификации и кластеризации с использованием различных подходов.


Описание проекта

Проект включает в себя:

    • Классификацию изображений с использованием различных библиотек машинного обучения: scikit-learn, TensorFlow и PyTorch.
    
    • Кластеризацию изображений с помощью метода KMeans.
    
    • Визуализацию и управление изображениями через веб-интерфейс на Django.
    
    • Сравнение производительности и точности разных моделей машинного обучения.
    
Задачи, которые решает проект:

    • Оценка производительности различных библиотек.
    
    • Обработка и кластеризация изображений.
    
    • Разработка веб-интерфейса для работы с изображениями и результатами обработки.
    
    


Установка проекта

Установка проекта на Windows

    1. Клонируйте проект с GitHub:
    

       git clone https://github.com/DipUrban.git
       
    2. Перейдите в папку проекта:
    

       cd my_diplom
       
    3. Создайте виртуальное окружение:
    

       python -m venv venv
       
    4. Активируйте виртуальное окружение:
    

       venv\Scripts\activate
       
    5. Установите зависимости:
    

       pip install -r requirements.txt
       
       

Использование проекта

После установки всех зависимостей и активации виртуального окружения, вы можете запустить проект.

Запуск проекта:

Для запуска проекта в локальном сервере, выполните команду:

python manage.py runserver

Это запустит сервер, и вы сможете открыть приложение в браузере по адресу http://127.0.0.1:8000/.

Пример работы с кластеризацией:

    1. Перейдите на страницу дашборда.
    
    2. Загрузите изображения.
    
    3. Нажмите кнопку "Кластеризовать изображения" для выполнения кластеризации.
    
    4. Изображения будут автоматически распределены по кластерам, и результаты отобразятся на странице.
    
    

Основной функционал

Проект включает несколько ключевых функций:

Классификация изображений:

    • Загрузка изображений с помощью Django.
    
    • Прогон изображений через модели машинного обучения (scikit-learn, TensorFlow, PyTorch).
    
    • Получение предсказанных меток и классов для изображений.
    
Кластеризация изображений:

    • Использование метода KMeans для кластеризации изображений.
    
    • Преобразование изображений в признаки с использованием предобученной модели ResNet50.
    
    • Разделение изображений на кластеры на основе их признаков.
    
Веб-интерфейс:

    • Загрузка, отображение и удаление изображений через Django админку.
    
    • Возможность кластеризации изображений через веб-форму.
    
    

Структура проекта

Вот основные файлы и папки в проекте:

    • app_project/ — Папка с приложением, содержащая бизнес-логику.
    
        ◦ models.py — Определение модели данных для изображений.
        
        ◦ utils.py — Модули для извлечения признаков из изображений и кластеризации.
        
        ◦ views.py — Обработчики представлений, включая логику кластеризации.
        
        ◦ templates/app_project/ — HTML-шаблоны для веб-интерфейса.

        ◦ urls.py — Маршруты для различных страниц.
        
    • manage.py — Главный файл для управления проектом Django.
    
    • requirements.txt — Список всех зависимостей для установки.
    
    • README.md — Документация проекта.
    https://github.com/Krochi/DipUrban/blob/main/Diplom/my_diplom/media/screenshot/dashbord-delete.png

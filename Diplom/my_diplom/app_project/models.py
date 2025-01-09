# Описание модели Image, которая представляет данные (прогнозы, кластеры и вероятности) связанные с изображениями.
"""
Эта модель используется для хранения изображений, предсказанных классов и меток,
а также дополнительной информации о кластеризации и вероятностях для анализа и дальнейшей обработки.

Поля модели:
-- `image`: Поле для хранения изображения. Изображения загружаются в папку 'images/'.
-- `uploaded_at`: Дата и время загрузки изображения.
-- `predicted_class`: Класс, предсказанный моделью для данного изображения.
-- `predicted_label`: Метка, предсказанная для изображения (например, название объекта).
-- `cluster`: Кластер, к которому было отнесено изображение после кластеризации.
-- `probability`: Вероятность того, что изображение принадлежит к предсказанному классу.
-- `objects`: Стандартный менеджер для работы с объектами модели.

Методы:
-- `delete`: Переопределенный метод для удаления объекта из базы данных.
-- `__str__`: Переопределенный метод для строкового представления объекта.
            Используется для отображения объекта в админке Django.
"""
from django.db import models
import os

# Определяем модель Image, которая будет представлять таблицу в базе данных
class Image(models.Model):
    """
        Модель используется для хранения изображений, предсказанных классов и меток,
        а также дополнительной информации о кластеризации и вероятностях для анализа и дальнейшей обработки.
       """
    # Поля модели
    image = models.ImageField(upload_to='images/', verbose_name='Изображение')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    predicted_class = models.CharField(max_length=255, blank=True, null=True)
    predicted_label = models.CharField(max_length=255, blank=True, null=True)
    cluster = models.IntegerField(null=True, blank=True, verbose_name='Кластер')
    probability = models.FloatField(null=True, blank=True)
    objects = models.Manager()

    def delete(self, *args, **kwargs): # Метод определяет удаление объекта из базы данных
        """
                Параметры:
                    *args, **kwargs: Дополнительные аргументы, передаваемые методу родительского класса.
                """
        if self.image:
            if os.path.isfile(self.image.path):
                os.remove(self.image.path)
        super().delete(*args, **kwargs)

    def __str__(self):# Метод определяет как будет отображаться объект в текстовом формате
        """
              Отображение в админке Django или в консоли.

               Возвращает:
                   str: Строковое представление объекта, включая его ID и дату загрузки.
               """
        # (в консоли или панели администратора)
        return f"Изображение {self.id} загружено {self.uploaded_at}"

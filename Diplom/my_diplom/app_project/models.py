from django.db import models
import os

class Image(models.Model):
    image = models.ImageField(upload_to='images/', verbose_name='Изображение')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    predicted_class = models.CharField(max_length=255, blank=True, null=True)
    predicted_label = models.CharField(max_length=255, blank=True, null=True)
    cluster = models.IntegerField(null=True, blank=True, verbose_name='Кластер')
    probability = models.FloatField(null=True, blank=True)
    objects = models.Manager()

    def delete(self, *args, **kwargs):
        if self.image:
            if os.path.isfile(self.image.path):
                os.remove(self.image.path)
        super().delete(*args, **kwargs)

    def __str__(self):
        return f"Изображение {self.id} загружено {self.uploaded_at}"

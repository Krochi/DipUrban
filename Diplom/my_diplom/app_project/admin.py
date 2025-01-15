from django.contrib import admin
from .models import Image

# Регистрируем модель Image в админке
@admin.register(Image)
class ImageAdmin(admin.ModelAdmin):
    """
        Класс для настройки отображения модели Image в административной панели Django.
        """
    list_display = ('id', 'image', 'uploaded_at', 'predicted_class', 'predicted_label', 'cluster', 'probability')  # Поля, которые будут отображаться в списке
    list_filter = ('uploaded_at', 'cluster')  # Фильтры по дате загрузки и кластеру
    search_fields = ('predicted_label', 'predicted_class')  # Поиск по метке и классу
    readonly_fields = ('uploaded_at',)  # Поле uploaded_at будет только для чтения

    # Опционально: настройка отображения формы редактирования
    fieldsets = (
        (None, {
            'fields': ('image', 'uploaded_at', 'predicted_class', 'predicted_label', 'cluster', 'probability')
        }),
    )
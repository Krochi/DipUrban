# Файл для описания форм, которые будут использоваться в веб-приложении
"""
Формы:
-- `ImageUploadForm`: для загрузки изображения.
-- `ImageForm`: для добавления изображения с возможностью кластеризации.
-- `RegistrationForm`: для регистрации нового пользователя.
-- `ProbabilityPredictionForm`: для выбора изображения для прогноза вероятностей.

Каждая из форм использует Django ModelForm, который упрощает взаимодействие с моделями базы данных.

Формы в этом файле позволяют пользователю загружать изображения, регистрировать учетные записи,
а также делать прогнозы на основе изображений.

"""
from django import forms
from .models import Image
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm
from django.core.validators import FileExtensionValidator

# Форма загрузки изображения
class ImageUploadForm(forms.ModelForm):
    """
        Форма используется для загрузки изображений в базу данных.

            -- image (ImageField): Поле для загрузки изображения.
        """
    class Meta:
        model = Image
        fields = ['image']

# Добавление поля с возможностью кластеризации
class ImageForm(forms.ModelForm):
    """
        Форма используется для загрузки изображения и выбора кластеризации, к которой оно будет отнесено.

            -- image (ImageField): Поле для загрузки изображения.
            -- cluster (CharField): Поле для указания кластера изображения.

        """
    class Meta:
        model = Image
        fields = ['image','cluster']  #

# Форма регистрации новых пользователей
class RegistrationForm(UserCreationForm):
    """
       Форма используется для регистрации нового пользователя в системе.

          -- username (CharField): Имя пользователя.
          -- email (EmailField): Электронная почта.
          -- password1 (CharField): Пароль.
          -- password2 (CharField): Подтверждение пароля.

       """
    email = forms.EmailField(required=True, label="Email", help_text="Введите действующий email")

    class Meta:
        model = User
        fields = ['username', 'email', 'password1', 'password2']

# Форма выбора изображения и прогнозирования вероятностей
class ProbabilityPredictionForm(forms.ModelForm):
     """
       Форма используется для выбора изображения, на основе которого будет производиться прогноз вероятности.

           -- image (ImageField): Поле для выбора изображения для прогноза.
       """
     class Meta:
        model = Image
        fields = ['image']


class ImageUploadForm(forms.ModelForm):

    class Meta:
        model = Image
        fields = ['image']

    # Валидация форматов
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['image'].validators.append(FileExtensionValidator(['jpg', 'jpeg', 'png']))

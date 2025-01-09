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


class ImageUploadForm(forms.ModelForm):
    """
        Форма используется для загрузки изображений в базу данных. Можно загрузить одно изображение,
        которое будет сохранено в поле `image` модели `Image`.

            -- image (ImageField): Поле для загрузки изображения.
        """
    class Meta:
        model = Image
        fields = ['image']


class ImageForm(forms.ModelForm):
    """
        Форма используется для загрузки изображения и выбора кластеризации, к которой оно будет отнесено.
        Можно выбрать изображение и указать, к какому кластеру оно должно быть отнесено.

            -- image (ImageField): Поле для загрузки изображения.
            -- cluster (CharField): Поле для указания кластера изображения.

        """
    class Meta:
        model = Image
        fields = ['image','cluster']  #


class RegistrationForm(UserCreationForm):
    """
       Форма используется для регистрации нового пользователя в системе.
       Она включает в себя поля для ввода имени пользователя, email-адреса и пароля.

          -- username (CharField): Имя пользователя.
          -- email (EmailField): Электронная почта.
          -- password1 (CharField): Пароль.
          -- password2 (CharField): Подтверждение пароля.

       """
    email = forms.EmailField(required=True, label="Email", help_text="Введите действующий email")

    class Meta:
        model = User
        fields = ['username', 'email', 'password1', 'password2']


class ProbabilityPredictionForm(forms.ModelForm):
     """
       Форма используется для выбора изображения, на основе которого будет производиться прогноз вероятности.
       Выбирается одно изображение, и оно передается в систему для дальнейшего анализа.

           -- image (ImageField): Поле для выбора изображения для прогноза.
       """
    class Meta:
        model = Image
        fields = ['image']

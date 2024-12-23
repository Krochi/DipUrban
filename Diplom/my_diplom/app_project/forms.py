from django import forms
from .models import Image
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm


class ImageUploadForm(forms.ModelForm):
    class Meta:
        model = Image
        fields = ['image']


class ImageForm(forms.ModelForm):
    class Meta:
        model = Image
        fields = ['image','cluster']  #


class RegistrationForm(UserCreationForm):
    email = forms.EmailField(required=True, label="Email", help_text="Введите действующий email")

    class Meta:
        model = User
        fields = ['username', 'email', 'password1', 'password2']


class ProbabilityPredictionForm(forms.ModelForm):
    class Meta:
        model = Image
        fields = ['image']
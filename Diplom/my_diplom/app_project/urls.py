# Определение маршрутов URL для приложения Django
"""
В этом файле указаны все маршруты, которые будут обрабатывать различные запросы к веб-приложению.

Основные маршруты:
-- Главная страница (`home`) - отображает домашнюю страницу приложения.
-- Страница регистрации (`register`) - форма для регистрации нового пользователя.
-- Страница дашборда (`dashboard`) - отображение информации о текущем пользователе.
-- Страница загрузки изображения (`add_image_feed`) - форма для загрузки изображений.
-- Страница удаления изображения (`delete_image`) - удаление изображения по ID.
-- Страница логина (`login`) - форма для входа в систему с использованием стандартного механизма Django.
-- Страница логаута (`logout`) - выход пользователя из системы с помощью стандартного механизма Django.
-- Страница предсказаний вероятностей (`predict_probabilities`) - страница, на которой отображаются вероятности предсказания классов для изображений.

Маршруты:
Каждому маршруту соответствует своя функция или класс.

Зависимости:
-- `django.urls.path`: Для определения путей.
-- `views`: Модуль с функциями представлений, обрабатывающими запросы.
-- `django.contrib.auth.views`: Стандартные представления Django для входа и выхода из системы.
-- `django.conf.settings`: Для работы с настройками Django.
-- `django.conf.urls.static`: Для обслуживания статических и медиафайлов.
"""
from django.urls import path
from . import views
from django.contrib.auth import views as auth_views
from django.conf import settings
from django.conf.urls.static import static

# "Path" функции из Django, с определённой функцией или классом представления. Список маршрутов:
urlpatterns = [
    path('', views.home, name='home'),
    path('register/', views.register, name='register'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('add_image_feed/', views.add_image_feed, name='add_image_feed'),
    path('delete_image/<int:image_id>/', views.delete_image, name='delete_image'),
    path('login/', auth_views.LoginView.as_view(template_name='app_project/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),
    path('predict_probabilities/', views.predict_probabilities, name='predict_probabilities'),

]
# Обработка медиафайлов. Добавляет маршруты для медиафайлов
static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)






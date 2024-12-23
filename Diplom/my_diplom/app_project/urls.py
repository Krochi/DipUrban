from django.urls import path
from . import views
from django.contrib.auth import views as auth_views
from django.conf import settings
from django.conf.urls.static import static

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
static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)






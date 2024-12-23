from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from .forms import RegistrationForm, ImageUploadForm, ImageForm, ProbabilityPredictionForm
from .models import Image
from .image_recognition import predict_image, predict_image_probabilities
from .utils import cluster_images

def home(request):
    return render(request, 'app_project/home.html')

@login_required
def dashboard(request):
    if request.method == "POST":
        if 'cluster_button' in request.POST:
            cluster_images()
        return redirect('dashboard')

    images = Image.objects.all()
    return render(request, 'app_project/dashboard.html', {'images': images})

def upload_image(request):
    if request.method == 'POST' and request.FILES['image']:
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.save(commit=False)
            image.label = "example_label"
            image.cluster = None
            image.save()
            return render(request, 'image_upload_success.html', {'image': image})
    else:
        form = ImageForm()
    return render(request, 'image_upload.html', {'form': form})


def add_image_feed(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.save()
            image_path = image.image.path

            try:
                result = predict_image(image_path)
                print(f"Result from predict_image: {result}")

                predicted_class, predicted_label = predict_image(image_path)

                image.predicted_class = predicted_class
                image.predicted_label = predicted_label
                image.save()
                cluster_images()

                return redirect('dashboard')
            except Exception as e:
                print(f"Error during prediction: {e}")
                return render(request, 'app_project/add_image.html', {'form': form, 'error': str(e)})
    else:
        form = ImageUploadForm()
    return render(request, 'app_project/add_image.html', {'form': form})


def delete_image(request, image_id):
    image = get_object_or_404(Image, id=image_id)
    image.delete()
    return redirect('dashboard')

def register(request):
    if request.method == 'POST':
        form = RegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('dashboard')
    else:
        form = RegistrationForm()
    return render(request, 'app_project/registration.html', {'form': form})

def predict_probabilities(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.save(commit=False)
            image.save()
            image_path = image.image.path
            predictions = predict_image_probabilities(image_path)
            return render(request, 'app_project/predict_probabilities.html', {
                'form': form,
                'predictions': predictions,
                'uploaded_image': image.image.url
            })
    else:
        form = ImageUploadForm()
    return render(request, 'app_project/predict_probabilities.html', {'form': form})

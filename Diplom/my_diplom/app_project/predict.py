import torch
from torchvision import models, transforms
from PIL import Image


model = models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V1')
model.eval()


def transform_image(image_path):
    transformation = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path)
    image = transformation(image).unsqueeze(0)
    return image


def get_class_label(predicted_idx):
    labels = {idx: label for idx, label in enumerate(open('imagenet_classes.txt').read().splitlines())}
    return labels.get(predicted_idx, 'Unknown')


def predict_image(image):
    image_tensor = transform_image(image)
    with torch.no_grad():
        output = model(image_tensor)
    _, predicted_idx = torch.max(output, 1)
    return predicted_idx.item()

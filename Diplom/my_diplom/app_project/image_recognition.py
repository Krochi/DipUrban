import torch
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image
import torch.nn.functional as F
from PIL import Image as PILImage

model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

imagenet_classes = ResNet50_Weights.IMAGENET1K_V1.meta["categories"]

def predict_image(image_path):
    image = PILImage.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
    class_idx = torch.argmax(probabilities).item()
    class_label = imagenet_classes[class_idx]
    return class_idx, class_label  # Возвращается кортеж


def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def predict_image_probabilities(image_path):
    input_tensor = PILImage.open(image_path).convert("RGB")
    input_tensor = transform(input_tensor).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)


    probabilities = probabilities.squeeze().tolist()
    predictions = {imagenet_classes[i]: round(prob * 100, 2) for i, prob in enumerate(probabilities) if prob > 0.0}

    return predictions
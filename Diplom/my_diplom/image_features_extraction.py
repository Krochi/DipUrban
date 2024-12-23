import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image as PILImage
import numpy as np
import tensorflow as tf

resnet_pytorch = models.resnet18(weights='IMAGENET1K_V1')
resnet_pytorch.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features_pytorch(image_path):
    img = PILImage.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        features = resnet_pytorch(img)
    return features.flatten().detach().numpy()

resnet_tensorflow = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, pooling='avg')

def preprocess_image_tf(image_path):
    img = PILImage.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img)
    img = tf.keras.applications.resnet50.preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

def extract_features_tensorflow(image_path):
    img = preprocess_image_tf(image_path)
    features = resnet_tensorflow(img)
    return features.numpy().flatten()

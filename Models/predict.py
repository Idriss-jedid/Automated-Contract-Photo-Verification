# save this as `predict.py`

from PIL import Image
import torch
from torchvision import transforms
import timm
from torchvision import models
import os

# Define the device and load models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_models():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    vit_model_path = os.path.join(script_dir, 'best_vit_model3.pth')
    vit_model = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=2)
    vit_model.load_state_dict(torch.load(vit_model_path, weights_only=True))
    vit_model.eval()

    resnet_model_path = os.path.join(script_dir, 'best_model.pth')
    resnet_model = models.resnet18(pretrained=True)
    num_ftrs = resnet_model.fc.in_features
    resnet_model.fc = torch.nn.Linear(num_ftrs, 5)
    resnet_model.load_state_dict(torch.load(resnet_model_path, map_location=torch.device('cpu'), weights_only=True))
    resnet_model.eval()

    return vit_model.to(device), resnet_model.to(device)

vit_model, resnet_model = load_models()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class_names = ['autre', 'avant', 'arrier', 'droite', 'gauche']

def predict_vit(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = vit_model(image)
        _, predicted = outputs.max(1)
    return predicted.item()

def predict_resnet(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = resnet_model(image)
        _, preds = torch.max(outputs, 1)
        predicted_class = class_names[preds.item()]
    return predicted_class

if __name__ == "__main__":
    import sys
    import json
    image_path = sys.argv[1]
    model = sys.argv[2]  # 'vit' or 'resnet'

    if model == 'vit':
        prediction = predict_vit(image_path)
    elif model == 'resnet':
        prediction = predict_resnet(image_path)
    else:
        prediction = 'Invalid model specified'

    print(json.dumps({'prediction': prediction}))

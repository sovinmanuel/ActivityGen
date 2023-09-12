import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18, resnet50
import cv2
from PIL import Image
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ResNetClassifier:
    def __init__(self, model_path):
        self.class_names = [
            "Button",
            "Checkbox",
            "EditText",
            "Image",
            "Icon",
            "RadioButton",
            "Switch",
        ]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = resnet50(pretrained=False)
        # self.model = resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, len(self.class_names))
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def classify_image(self, image_np):
        image = Image.fromarray(np.uint8(image_np)).convert("RGB")
        image = self.transform(image)
        image = image.to(self.device)
        with torch.no_grad():
            outputs = self.model(image.unsqueeze(0))
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted_idx = torch.max(outputs, 1)
            predicted_label = self.class_names[predicted_idx.item()]
            predicted_probabilities = probabilities.squeeze(0).tolist()
            label_probability = predicted_probabilities[predicted_idx.item()]

        return predicted_label, label_probability

    def load_and_classify_image(self, image_path):
        image = self.load_image(image_path)

        with torch.no_grad():
            output = self.model(image.unsqueeze(0))
            _, predicted_idx = torch.max(output, 1)
            predicted_label = self.class_names[predicted_idx.item()]

        return predicted_label

    def load_image(self, image_path):
        image = Image.open(image_path)
        image = self.transform(image)
        return image

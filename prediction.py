import torch
import torch.nn as nn
from torchvision import transforms, models
import os
from PIL import Image
import pandas as pd
from response_model import PredictionResponse


class Classifier:
    def __init__(self) -> None:
        self.class_names = [
            "burger",
            "chakli",
            "cheese_cakes",
            "chole_bhature",
            "cookies",
            "fafta",
            "french_fries",
            "gobi_manchurian",
            "golgappa",
            "gulab_jamun",
            "jilebi",
            "ladoo",
            "masala_dosa",
            "milkshake",
            "mysore_pak",
            "noodles",
            "pancakes",
            "panner_butter_masala",
            "panner_tikka",
            "pasta",
            "pav_bhaji",
            "pizza",
            "shushi",
            "tacos",
            "veg_biriyani",
        ]
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, 1000)
        self.model.load_state_dict(torch.load("model/torch_model.v1.pth"))
        self.model.eval()
        self.data = pd.read_csv("data/recipes.csv")

    def preprocess(self, image_path: str):
        image = Image.open(image_path).convert("RGB")
        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)
        return input_batch

    def predict(self, image_path: str):
        input_batch = self.preprocess(image_path)
        with torch.no_grad():
            output = self.model(input_batch)
        _, predicted_class = output.max(1)
        print(predicted_class.item())
        row = self.data.iloc[predicted_class]
        print(row)
        response = PredictionResponse(
            name=row["name"][1],
            image=row["image"][1],
            diet=row["diet"][1],
            course=row["course"][1],
            time=row["time"][1],
            ingredients=row["ingredients"][1],
            instructions=row["instructions"][1],
        )
        return response


if __name__ == "__main__":
    classifier = Classifier()
    predicted_class = classifier.predict("images/Baked-gobi-manchurian.jpg")
    print(predicted_class)

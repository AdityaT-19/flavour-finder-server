import os

from typing import Union
from prediction import Classifier
from recommendation import Recommendation
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

app = FastAPI()
classifier = Classifier()
recommendation = Recommendation()


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/predict")
async def predictImage(file: UploadFile):
    file_location = f"images/{file.filename}"
    f = open(file_location, "wb")
    f.write(file.file.read())
    f.close()
    response = classifier.predict(file_location)
    os.remove(file_location)
    return response


class IngredientRequest(BaseModel):
    ingredients: str


@app.post("/recommend")
async def recommendDish(ingredient_request: IngredientRequest):
    ingredients = ingredient_request.ingredients.split(" ")
    response = recommendation.getdishes(ingredients)
    return response

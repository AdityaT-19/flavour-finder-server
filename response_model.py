from pydantic import BaseModel


class PredictionResponse(BaseModel):
    name: str
    image: str
    diet: str
    course: str
    time: int
    ingredients: str
    instructions: str


class RecommendationResponse(BaseModel):
    name: str
    ingredients: str
    instructions: str
    link: str
    diet: str
    course: str
    time: int

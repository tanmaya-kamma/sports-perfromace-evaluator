from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Request body schema
class InputData(BaseModel):
    speed: float
    angle: float

# Root endpoint
@app.get("/")
def home():
    return {"message": "Sports Performance Evaluator API is running 🚀"}

# Example prediction endpoint
@app.post("/predict")
def predict(data: InputData):
    # Placeholder logic (replace with ML model later)
    result = {
        "message": "Prediction endpoint working",
        "input_received": data.dict()
    }
    return result
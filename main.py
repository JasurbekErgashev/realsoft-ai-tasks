from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import joblib
import pandas as pd
import logging


# Initializing logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Loading the model
with open("final_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Initializing the FastAPI app
app = FastAPI()


# Defining the request schema
class DataPoint(BaseModel):
    gender: str
    race_ethnicity: str
    parental_level_of_education: str
    lunch: str
    test_preparation_course: str


@app.post("/predict")
async def predict(data: DataPoint):
    # Converting the request data to a DataFrame
    data_df = pd.DataFrame([data.model_dump()])

    try:
        # Performing prediction
        predictions = model.predict(data_df)
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

    return {"prediction": predictions[0]}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")

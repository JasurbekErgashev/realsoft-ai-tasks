from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError
from typing_extensions import Annotated
from pydantic.types import StringConstraints
from api.constants import TRAINING_COLUMNS, CATEGORICAL_FEATURES
import uvicorn
import joblib
import pandas as pd
import logging


# Initializing logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Loading the model
model = joblib.load("training/final_model.pkl")


# Preprocessing the input data
def preprocess_data(input_data):
    data_dict = input_data.model_dump()
    data_df = pd.DataFrame([data_dict])

    data_df = pd.get_dummies(data_df, columns=CATEGORICAL_FEATURES, dtype=int)
    for col in TRAINING_COLUMNS:
        if col not in data_df.columns:
            data_df[col] = 0

    data_df = data_df[TRAINING_COLUMNS]
    return data_df


# Initializing the FastAPI app
app = FastAPI()


# Defining the request schema
class DataPoint(BaseModel):
    gender: Annotated[str, StringConstraints(pattern="^(female|male)$")]
    race_ethnicity: Annotated[str, StringConstraints(pattern="^(group [A-E])$")]
    parental_level_of_education: Annotated[
        str,
        StringConstraints(
            pattern="^(bachelor's degree|some college|master's degree|associate's degree|high school|some high school)$"
        ),
    ]
    lunch: Annotated[str, StringConstraints(pattern="^(standard|free/reduced)$")]
    test_preparation_course: Annotated[
        str, StringConstraints(pattern="^(none|completed)$")
    ]


@app.post("/predict")
async def predict(data: DataPoint):
    # Preprocessing the input data
    data_df = preprocess_data(data)

    try:
        # Performing prediction
        predictions = model.predict(data_df)
        return {
            "overall_score": predictions[0],
            "details": data.model_dump(),
            "model": "RandomForestRegressor",
        }
        # Error handling
    except ValidationError as e:
        error_msg = ", ".join(
            [f"{error['loc'][0]}: {error['msg']}" for error in e.errors()]
        )
        raise HTTPException(
            status_code=422, detail=f"Input data validation failed: {error_msg}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")

# Machine Learning Model Pipeline Documentation

## Table of Contents

- [Overview](#overview)
- [Dataset Information](#dataset-information)
- [Preprocessing Steps](#preprocessing-steps)
- [Preprocessing Pipelines](#preprocessing-pipelines)
- [Model Training and Pipeline](#model-training-and-pipeline)
- [How to Use the Trained Model](#how-to-use-the-trained-model)
  - [Prerequisites](#prerequisites)
  - [Loading the Model](#loading-the-model)
  - [Preparing Input Data](#preparing-input-data)
  - [Making Predictions](#making-predictions)
- [Conclusion](#conclusion)

## Overview

This documentation provides a comprehensive guide on how to use the trained machine learning model pipeline, including details on loading the model from a `.pkl` file, preparing the input data, and making predictions.

## Dataset Information

The dataset comprises survey results from computer science students, aiming to identify correlations between their depression levels, class performance through data analysis.

The columns are:

- **Age**: Represents the age of the individuals.
- **Gender**: Indicates the gender of each individual.
- **AcademicPerformance**: Reflects the academic achievements of individuals.
- **TakingNoteInClass**: Describes if individuals take notes during class.
- **DepressionStatus**: Indicates the presence or absence of depressive symptoms _(Target variable)_.
- **FaceChallangesToCompleteAcademicTask**: Explores whether individuals encounter challenges in completing academic tasks.
- **LikePresentation**: Reflects individuals' preferences for presentations.
- **SleepPerDayHours**: Represents the average hours of sleep individuals get per day.
- **NumberOfFriend**: Quantifies the number of friends each individual has.
- **LikeNewThings**: Explores individuals' receptiveness to new experiences or concepts.

## Preprocessing Steps

- Numerical Features:
  `Age` `SleepPerDayHours` `NumberOfFriend`
- Categorical Features:
  `Gender` `AcademicPerformance` `TakingNoteInClass` `FaceChallangesToCompleteAcademicTask` `LikePresentation` `LikeNewThings`

## Preprocessing Pipelines

- Numerical Features:
  - Imputation using the mean.
  - Standardization using `StandardScaler`.
- Categorical Features:
  - Imputation using the most frequent value.
  - One-hot encoding using `OneHotEncoder`.

## Model Training and Pipeline

The model used is a `GradientBoostingClassifier`, integrated into a pipeline that includes preprocessing steps.

## How to Use the Trained Model

#### Prerequisites

- Python 3.6+
- `pandas`, `scikit-learn`, and `joblib` libraries installed

#### Loading the Model

```python
from joblib import load
clf = load('trained_pipeline.pkl')
```

#### Preparing Input Data

Ensure the input data matches the format used during training, including column names and preprocessing steps. Here is an example of how to prepare and standardize the input data:

```python
import pandas as pd

data = {
    'Age': [23],
    'Gender': ['Male'],
    'AcademicPerformance': ['Excellent'],
    'TakingNoteInClass': ['Yes'],
    'FaceChallangesToCompleteAcademicTask': ['No'],
    'LikePresentation': ['Yes'],
    'SleepPerDayHours': [8],
    'NumberOfFriend': [15],
    'LikeNewThings': ['Yes']
}

input_df = pd.DataFrame(data)
```

#### Making Predictions

Use the loaded pipeline to make predictions on new data:

```python
predictions = clf.predict(input_df)
print("Predictions:", predictions)
```

## Conclusion

This documentation provides detailed steps on how to load, preprocess input data, and use the trained machine learning model pipeline for making predictions. Ensure the input data format is consistent with the training data to achieve accurate predictions.

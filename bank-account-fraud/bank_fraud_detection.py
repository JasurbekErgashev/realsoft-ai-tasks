# -*- coding: utf-8 -*-
"""bank_fraud_detection.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1mkBNIQsHxzGDtXAsMFqkiP_9VkDP5jFo
"""

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

data = pd.read_csv('/content/drive/MyDrive/NLP/Base.csv')
pd.set_option('display.max_columns', None)

data.head()

X = data.drop('fraud_bool', axis=1)
y = data['fraud_bool']

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

resampling = SMOTE()

model = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_split=5, class_weight='balanced', random_state=42)

pipeline = ImbPipeline(steps=[('preprocessor', preprocessor),
                              ('resampling', resampling),
                              ('classifier', model)])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1]))

import joblib
joblib.dump(pipeline, 'fraud_detection_model.pkl')
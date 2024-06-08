from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_predict_valid_data():
    """Test case with valid input data."""
    response = client.post(
        "/predict",
        json={
            "gender": "female",
            "race_ethnicity": "group B",
            "parental_level_of_education": "bachelor's degree",
            "lunch": "standard",
            "test_preparation_course": "none",
        },
    )
    assert response.status_code == 200


def test_predict_invalid_gender():
    """Test case with an invalid gender value."""
    response = client.post(
        "/predict",
        json={
            "gender": "invalid_gender",
            "race_ethnicity": "group B",
            "parental_level_of_education": "bachelor's degree",
            "lunch": "standard",
            "test_preparation_course": "none",
        },
    )
    assert response.status_code == 422


def test_predict_invalid_race_ethnicity():
    """Test case with an invalid race_ethnicity value."""
    response = client.post(
        "/predict",
        json={
            "gender": "female",
            "race_ethnicity": "group Z",
            "parental_level_of_education": "bachelor's degree",
            "lunch": "standard",
            "test_preparation_course": "none",
        },
    )
    assert response.status_code == 422


def test_predict_missing_field():
    """Test case with a missing required field."""
    response = client.post(
        "/predict",
        json={
            "gender": "female",
            "race_ethnicity": "group B",
            "parental_level_of_education": "bachelor's degree",
            "lunch": "standard",
            # Missing test_preparation_course
        },
    )
    assert response.status_code == 422


def test_predict_extra_field():
    """Test case with an extra, unexpected field."""
    response = client.post(
        "/predict",
        json={
            "gender": "female",
            "race_ethnicity": "group B",
            "parental_level_of_education": "bachelor's degree",
            "lunch": "standard",
            "test_preparation_course": "none",
            "extra_field": "extra_value",
        },
    )
    assert response.status_code == 200


def test_predict_invalid_combination():
    """Test case with multiple invalid fields."""
    response = client.post(
        "/predict",
        json={
            "gender": "invalid_gender",
            "race_ethnicity": "group Z",
            "parental_level_of_education": "invalid_education",
            "lunch": "invalid_lunch",
            "test_preparation_course": "invalid_course",
        },
    )
    assert response.status_code == 422


if __name__ == "__main__":
    test_predict_valid_data()
    test_predict_invalid_gender()
    test_predict_invalid_race_ethnicity()
    test_predict_missing_field()
    test_predict_extra_field()
    test_predict_invalid_combination()

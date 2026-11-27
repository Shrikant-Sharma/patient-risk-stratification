"""
The data dictionary: what each of the 13 inputs means, its unit, and the values it may take.

One place holds it. The request schema is checked against it, GET /data-dictionary returns it,
and the model's metadata adds the range seen in training. Source of the meanings: the UCI
Heart Disease documentation for the Cleveland file.
"""

FIELDS = {
    "age": {"meaning": "Age", "unit": "years", "kind": "number", "min": 18, "max": 100},
    "sex": {"meaning": "Sex", "kind": "code", "values": {"0": "female", "1": "male"}},
    "cp": {"meaning": "Chest pain type", "kind": "code",
           "values": {"1": "typical angina", "2": "atypical angina", "3": "non-anginal pain", "4": "no chest pain (asymptomatic)"}},
    "trestbps": {"meaning": "Resting blood pressure on admission", "unit": "mm Hg", "kind": "number", "min": 50, "max": 250},
    "chol": {"meaning": "Serum cholesterol", "unit": "mg/dL", "kind": "number", "min": 100, "max": 700},
    "fbs": {"meaning": "Fasting blood sugar above 120 mg/dL", "kind": "code", "values": {"0": "no", "1": "yes"}},
    "restecg": {"meaning": "Resting ECG result", "kind": "code",
                "values": {"0": "normal", "1": "ST-T wave abnormality", "2": "probable or definite left ventricular hypertrophy"}},
    "thalach": {"meaning": "Highest heart rate reached in the exercise test", "unit": "beats per minute", "kind": "number", "min": 50, "max": 250},
    "exang": {"meaning": "Angina brought on by exercise", "kind": "code", "values": {"0": "no", "1": "yes"}},
    "oldpeak": {"meaning": "ST depression brought on by exercise, relative to rest", "unit": "mm", "kind": "number", "min": 0, "max": 10},
    "slope": {"meaning": "Slope of the ST segment at peak exercise", "kind": "code", "values": {"1": "upsloping", "2": "flat", "3": "downsloping"}},
    "ca": {"meaning": "Major vessels colored by fluoroscopy", "kind": "code", "values": {"0": "none", "1": "one", "2": "two", "3": "three"}},
    "thal": {"meaning": "Thallium stress test result", "kind": "code", "values": {"3": "normal", "6": "fixed defect", "7": "reversible defect"}},
}

TARGET = {"meaning": "Heart disease found by angiography: more than 50% narrowing of a major vessel",
          "values": {"0": "no", "1": "yes"},
          "note": "The file grades disease 0 to 4. Grades 1 to 4 are counted as yes."}

SOURCE = {"name": "UCI Heart Disease, Cleveland file", "patients": 303, "collected": "Cleveland Clinic Foundation, published 1988",
          "url": "https://archive.ics.uci.edu/dataset/45/heart+disease"}


def allowed(name):
    """The codes a coded field may take, as integers."""
    return sorted(int(k) for k in FIELDS[name]["values"])

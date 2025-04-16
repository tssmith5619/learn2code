import joblib

def save_model(model, filename="trained_model.joblib"):
    joblib.dump(model, filename)

def load_model(filename="trained_model.joblib"):
    return joblib.load(filename)

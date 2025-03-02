from sklearn.linear_model import LogisticRegression
import numpy as np
import joblib
import os
import time

# ✅ Initialize model
model = LogisticRegression(max_iter=1000, solver="liblinear")
batch_X = []
batch_y = []
BATCH_SIZE = 5  # ✅ Mini-batch size for training

# ✅ Model file to save and load
MODEL_FILE = "trained_model.pkl"

def train_model(task):
    """Trains the model using mini-batch updates and saves after training."""
    global batch_X, batch_y, model

    time.sleep(0.1)  # ⏳ Simulate processing time

    batch_X.append(task["data"])
    batch_y.append(task["label"])

    # ✅ Train when batch is full and has at least 2 classes
    if len(batch_X) >= BATCH_SIZE and len(set(batch_y)) > 1:
        X = np.array(batch_X)
        y = np.array(batch_y)

        print(f"✅ Training with {len(batch_X)} samples, Labels: {set(y)}")
        model.fit(X, y)  # Train model

        save_model()  # ✅ Save after training
        batch_X.clear()
        batch_y.clear()

    return {"task_id": task["task_id"], "passenger_id": task["passenger_id"], "prediction": None}  # ✅ No prediction during training

def predict(task):
    """Predicts using the trained model."""
    global model

    time.sleep(0.1)  # ⏳ Simulate processing time

    if not os.path.exists(MODEL_FILE):
        print(f"⚠️ Model file not found! Skipping prediction for Task ID {task['task_id']}")
        return {"task_id": task["task_id"], "passenger_id": task["passenger_id"], "prediction": [-1]}  # Default to -1

    model = load_model()  # ✅ Load model before predicting
    prediction = model.predict(np.array([task["data"]])).tolist()

    print(f"✅ Prediction for Task ID {task['task_id']}: {prediction}")
    return {"task_id": task["task_id"], "passenger_id": task["passenger_id"], "prediction": prediction}

def save_model():
    """Saves the trained model to a file."""
    joblib.dump(model, MODEL_FILE)
    print("✅ Model saved successfully!")

def load_model():
    """Loads the trained model from a file."""
    return joblib.load(MODEL_FILE)

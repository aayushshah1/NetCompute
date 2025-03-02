import pandas as pd
import os

train_df = pd.read_csv(os.path.join("Data", "submission_check.csv"))  # Original labels
pred_df = pd.read_csv(os.path.join("Data", "submission_test.csv"))

merged = train_df.merge(pred_df, on="PassengerId", suffixes=("_actual", "_predicted"))
accuracy = (merged["Survived_actual"] == merged["Survived_predicted"]).mean()

print(f"✅ Training Accuracy: {accuracy:.2%}")

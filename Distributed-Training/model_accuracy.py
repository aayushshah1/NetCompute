import pandas as pd

# train_df = pd.read_csv("processed_train.csv")  # Original labels
# pred_df = pd.read_csv("submission_train.csv")  # Model predictions

train_df = pd.read_csv("Kaggle_submission.csv")  # Original labels
pred_df = pd.read_csv("submission_test.csv") 

merged = train_df.merge(pred_df, on="PassengerId", suffixes=("_actual", "_predicted"))
accuracy = (merged["Survived_actual"] == merged["Survived_predicted"]).mean()

print(f"✅ Training Accuracy: {accuracy:.2%}")

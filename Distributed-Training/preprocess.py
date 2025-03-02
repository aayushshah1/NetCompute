import pandas as pd
import os

def preprocess_data(input_file, output_file):
    """Preprocess Titanic dataset and save cleaned version."""
    df = pd.read_csv(input_file)

    # Drop unnecessary columns
    df.drop(["Name", "Ticket", "Cabin"], axis=1, inplace=True)

    # Convert categorical variables
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"] = df["Embarked"].map({"C": 0, "Q": 1, "S": 2})

    # Handle missing values
    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Fare"].fillna(df["Fare"].median(), inplace=True)
    df["Embarked"].fillna(2, inplace=True)  # Default to 'S' (Southampton)

    # Save cleaned data
    df.to_csv(output_file, index=False)
    print(f"✅ Preprocessed data saved to {output_file}")

if __name__ == "__main__":
    preprocess_data(os.path.join("Data", "test.csv"), os.path.join("Data", "processed_test.csv"))
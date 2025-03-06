import pandas as pd
import numpy as np

# Load the existing dataset
file_path = "/Users/aayushshah/Work/Programming/NetCompute/Distributed-Training/Data/processed_train.csv"  # Update this with the actual file path
df = pd.read_csv(file_path)

# Function to generate synthetic variations
def generate_variations(df, target_size=50000):
    new_rows = []
    current_id = df["PassengerId"].max() + 1  # Start sequential numbering

    while len(df) + len(new_rows) < target_size:
        row = df.sample(n=1).iloc[0].copy()  # Randomly pick a row
        
        row["PassengerId"] = current_id  # Assign new sequential ID
        current_id += 1

        row["Age"] = round(max(0, row["Age"] + np.random.randint(-5, 6)))  # Ensure integer values
        row["Fare"] = round(max(0, row["Fare"] * np.random.uniform(0.8, 1.2)), 2)  # Round fare

        row["Embarked"] = np.random.choice([0, 1, 2])  # Random embarkation

        new_rows.append(row)

    return pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

# Generate the augmented dataset
augmented_df = generate_variations(df, target_size=50000)

# Ensure PassengerId starts from 1 and increments sequentially
augmented_df["PassengerId"] = range(1, 50001)

# Save to CSV
augmented_df.to_csv("augmented_dataset.csv", index=False)

print("Dataset expanded to:", augmented_df.shape)
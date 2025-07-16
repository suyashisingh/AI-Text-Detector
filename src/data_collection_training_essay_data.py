import pandas as pd
import os

# === CONFIGURATION ===
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_path = os.path.join(project_root, "data", "raw", "Training_Essay_Data.csv")
output_path = os.path.join(project_root, "data", "processed", "training_essay_processed.csv")

# Ensure output directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Load the CSV
df = pd.read_csv(input_path)
print("Columns in your CSV:", list(df.columns))
print("First few rows:\n", df.head())

# Map 0 to 'human', 1 to 'ai'
df['label'] = df['generated'].map({0: 'human', 1: 'ai'})

# Remove empty or NaN texts
df['text'] = df['text'].astype(str).str.strip()
df = df[df['text'] != ""]
df = df.dropna(subset=['text', 'label'])

# Keep only the columns needed
final_df = df[['text', 'label']]

# Save the cleaned data
final_df.to_csv(output_path, index=False)
print(f"✅ Cleaned Training Essay data saved as {output_path}")
print(f"Total samples: {len(final_df)}")
print(final_df['label'].value_counts())

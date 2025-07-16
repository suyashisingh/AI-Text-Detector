import os
import pandas as pd

# Get the project root (two levels up from this script)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_path = os.path.join(project_root, "data", "raw", "AI_Human.csv")
output_path = os.path.join(project_root, "data", "processed", "ai_human_processed.csv")

os.makedirs(os.path.dirname(output_path), exist_ok=True)

try:
    df = pd.read_csv(input_path)
    print("✅ File loaded successfully!")
except FileNotFoundError:
    print(f"❌ Error: File not found at {input_path}")
    print("Verify the file exists in the correct location and the filename is exact.")
    exit(1)

# Map labels
df['label'] = df['generated'].map({0: 'human', 1: 'ai'})

# Clean data
df['text'] = df['text'].astype(str).str.strip()
df = df[df['text'] != ""]
df = df.dropna(subset=['text', 'label'])

# Save processed data
df[['text', 'label']].to_csv(output_path, index=False)
print(f"✅ Cleaned data saved to: {output_path}")
print(f"Total samples: {len(df)}")
print(df['label'].value_counts())

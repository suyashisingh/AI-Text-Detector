import pandas as pd

# Paths
DATA_PATH = r"C:\Users\hp\Desktop\DMPROJECT\AI-detector\data\processed\combined_dataset.csv"
OUTPUT_PATH = r"C:\Users\hp\Desktop\DMPROJECT\AI-detector\data\processed\combined_dataset_balanced.csv"

# Load the dataset
df = pd.read_csv(DATA_PATH, low_memory=False)
df['label'] = df['label'].astype(str).str.strip().str.lower()
df['text'] = df['text'].astype(str).str.strip()

# Get class counts
ai_count = (df['label'] == 'ai').sum()

# Downsample 'human' to match 'ai'
human_sample = df[df['label'] == 'human'].sample(n=ai_count, random_state=42)
ai_sample = df[df['label'] == 'ai']

# Concatenate and shuffle
df_balanced = pd.concat([ai_sample, human_sample]).sample(frac=1, random_state=42).reset_index(drop=True)

# Save to new CSV
df_balanced.to_csv(OUTPUT_PATH, index=False)
print(f"Balanced dataset saved to: {OUTPUT_PATH}")

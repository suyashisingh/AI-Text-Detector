import pandas as pd
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_path = os.path.join(project_root, "data", "raw", "figshare_data.csv")
output_path = os.path.join(project_root, "data", "processed", "figshare_processed.csv")

os.makedirs(os.path.dirname(output_path), exist_ok=True)

df = pd.read_csv(input_path)
df['Type'] = df['Type'].astype(str).str.strip()
print("Unique values in 'Type':", df['Type'].unique())

# Treat all as human
df['label'] = 'human'

def combine_text(row):
    abstract = str(row.get('Abstract', '')).strip()
    discussion = str(row.get('Discussion', '')).strip()
    return f"{abstract} {discussion}".strip()

df['text'] = df.apply(combine_text, axis=1)
df = df[df['text'].astype(str).str.strip() != ""]
final_df = df[['text', 'label']]
final_df.to_csv(output_path, index=False)
print(f"✅ Cleaned Figshare data saved as {output_path}")
print(f"Total samples: {len(final_df)}")
print(final_df['label'].value_counts())

import os
import pandas as pd

# ==== CONFIGURATION ====
RAW_DIR = r"C:\Users\hp\Desktop\DMPROJECT\AI-detector\data\raw"
OUTPUT_DIR = r"C:\Users\hp\Desktop\DMPROJECT\AI-detector\data\processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_training_essay():
    path = os.path.join(OUTPUT_DIR, "training_essay_processed.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        # If only raw file exists, process from raw
        raw_path = os.path.join(RAW_DIR, "Training_Essay_Data.csv")
        df = pd.read_csv(raw_path)
        df['label'] = df['generated'].map({0: 'human', 1: 'ai'})
        df['text'] = df['text'].astype(str).str.strip()
        df = df[df['text'] != ""]
        df = df.dropna(subset=['text', 'label'])
    df['label'] = df['label'].astype(str).str.strip()
    df['text'] = df['text'].astype(str).str.strip()
    df = df[df['text'] != ""]
    df = df.dropna(subset=['text', 'label'])
    return df[['text', 'label']]

def process_human_vs_machine_wiki():
    path = os.path.join(RAW_DIR, "human_vs_machine_wiki.csv")
    df = pd.read_csv(path)
    df['label'] = df['label'].map({0: 'human', 1: 'ai'})
    df['text'] = df['text'].astype(str).str.strip()
    df = df[df['text'] != ""]
    df = df.dropna(subset=['text', 'label'])
    return df[['text', 'label']]

def process_human_vs_machine_research():
    path = os.path.join(RAW_DIR, "human_vs_machine_research.csv")
    df = pd.read_csv(path)
    df['label'] = df['label'].map({0: 'human', 1: 'ai'})
    df['text'] = df['text'].astype(str).str.strip()
    df = df[df['text'] != ""]
    df = df.dropna(subset=['text', 'label'])
    return df[['text', 'label']]

def process_figshare():
    path = os.path.join(OUTPUT_DIR, "figshare_processed.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        raw_path = os.path.join(RAW_DIR, "figshare_data.csv")
        df = pd.read_csv(raw_path)
        df['label'] = 'human'
        df['text'] = df['Abstract'].astype(str).str.strip() + " " + df['Discussion'].astype(str).str.strip()
    df['label'] = df['label'].astype(str).str.strip()
    df['text'] = df['text'].astype(str).str.strip()
    df = df[df['text'] != ""]
    df = df.dropna(subset=['text', 'label'])
    return df[['text', 'label']]

def process_ai_human():
    path = os.path.join(OUTPUT_DIR, "ai_human_processed.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        raw_path = os.path.join(RAW_DIR, "AI_Human.csv")
        df = pd.read_csv(raw_path)
        df['label'] = df['generated'].map({0: 'human', 1: 'ai'})
        df['text'] = df['text'].astype(str).str.strip()
    df['label'] = df['label'].astype(str).str.strip()
    df['text'] = df['text'].astype(str).str.strip()
    df = df[df['text'] != ""]
    df = df.dropna(subset=['text', 'label'])
    return df[['text', 'label']]

def main():
    print("Processing all datasets...")
    dfs = []
    try:
        dfs.append(process_training_essay())
        print("✓ training_essay_processed")
    except Exception as e:
        print(f"Error: training_essay_processed - {e}")
    try:
        dfs.append(process_human_vs_machine_wiki())
        print("✓ human_vs_machine_wiki")
    except Exception as e:
        print(f"Error: human_vs_machine_wiki - {e}")
    try:
        dfs.append(process_human_vs_machine_research())
        print("✓ human_vs_machine_research")
    except Exception as e:
        print(f"Error: human_vs_machine_research - {e}")
    try:
        dfs.append(process_figshare())
        print("✓ figshare_processed")
    except Exception as e:
        print(f"Error: figshare_processed - {e}")
    try:
        dfs.append(process_ai_human())
        print("✓ ai_human_processed")
    except Exception as e:
        print(f"Error: ai_human_processed - {e}")

    # Combine all dataframes
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=['text', 'label'])
    combined_path = os.path.join(OUTPUT_DIR, "combined_dataset.csv")
    combined.to_csv(combined_path, index=False)
    print(f"✅ Combined dataset saved to: {combined_path}")
    print(f"Total samples: {len(combined)}")

if __name__ == "__main__":
    main()

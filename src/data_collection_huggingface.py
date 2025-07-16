# import os
# from datasets import load_dataset
#
# # Make sure the output directory exists
# os.makedirs("data/processed", exist_ok=True)
#
# # List of configs to download
# configs = [
#     ("research_abstracts_labeled", "human_vs_machine_research.csv"),
#     ("wiki_labeled", "human_vs_machine_wiki.csv")
# ]
#
# for config_name, output_csv in configs:
#     print(f"Downloading config: {config_name} ...")
#     dataset = load_dataset('NicolaiSivesind/human-vs-machine', config_name)
#     # Convert 'train' split to pandas DataFrame
#     train_df = dataset['train'].to_pandas()
#     # Save as CSV
#     csv_path = os.path.join("data/processed", output_csv)
#     train_df.to_csv(csv_path, index=False)
#     print(f"Saved {config_name} split as {csv_path}")
#
# print("✅ All datasets downloaded and saved as CSV.")
import os
from datasets import load_dataset

# Get the project root (parent directory of src)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
processed_dir = os.path.join(project_root, "data", "processed")
os.makedirs(processed_dir, exist_ok=True)

# List of configs to download
configs = [
    ("research_abstracts_labeled", "human_vs_machine_research.csv"),
    ("wiki_labeled", "human_vs_machine_wiki.csv")
]

for config_name, output_csv in configs:
    print(f"Downloading config: {config_name} ...")
    dataset = load_dataset('NicolaiSivesind/human-vs-machine', config_name)
    # Convert 'train' split to pandas DataFrame
    train_df = dataset['train'].to_pandas()
    # Save as CSV in the processed folder outside src
    csv_path = os.path.join(processed_dir, output_csv)
    train_df.to_csv(csv_path, index=False)
    print(f"Saved {config_name} split as {csv_path}")

print("✅ All datasets downloaded and saved as CSV in data/processed.")

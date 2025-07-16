import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from tqdm import tqdm
import pickle

# ==== CONFIGURATION ====
DATA_PATH = r"C:\Users\hp\Desktop\DMPROJECT\AI-detector\data\processed\combined_dataset_balanced.csv"
MODEL_DIR = r"C:\Users\hp\Desktop\DMPROJECT\AI-detector\model"
MODEL_PATH = os.path.join(MODEL_DIR, "pytorch_fast_meanembed.pt")
VOCAB_PATH = os.path.join(MODEL_DIR, "vocab_fast.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder_fast.pkl")
os.makedirs(MODEL_DIR, exist_ok=True)
BATCH_SIZE = 256
EMBED_DIM = 64
EPOCHS = 2
MAX_VOCAB_SIZE = 5000
MAX_LEN = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== DATA LOADING (sample for speed) ====
df = pd.read_csv(DATA_PATH, low_memory=False)
df['label'] = df['label'].astype(str).str.strip()
df['text'] = df['text'].astype(str).str.strip()
df = df.dropna(subset=['text', 'label'])
df = df[df['text'] != ""]
df = df[df['label'] != ""]

# --- Debug: Show class counts and unique labels ---
print("Label counts before filtering:")
print(df['label'].value_counts(dropna=False))
print("Unique labels:", df['label'].unique())

# Remove classes with fewer than 2 samples
counts = df['label'].value_counts()
valid_labels = counts[counts >= 2].index
df = df[df['label'].isin(valid_labels)]
print("Label counts after filtering:")
print(df['label'].value_counts())

if len(df['label'].unique()) < 2:
    print("ERROR: Only one class remains after filtering. You need at least two classes with at least 2 samples each.")
    exit(1)

# Sample for speed (adjust n as needed)
if len(df) > 10000:
    df = df.sample(n=10000, random_state=42)

# Encode labels to 0/1
le = LabelEncoder()
df['label_enc'] = le.fit_transform(df['label'])

# Train/test split
try:
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].tolist(), df['label_enc'].tolist(), test_size=0.2, random_state=42, stratify=df['label_enc']
    )
except ValueError as e:
    print("ERROR during train_test_split:", e)
    print("Label distribution:", pd.Series(df['label_enc']).value_counts())
    exit(1)

# ==== TOKENIZATION & VOCAB ====
def tokenize(text):
    return text.lower().split()

counter = Counter()
for text in train_texts:
    counter.update(tokenize(text))

vocab = ['<PAD>', '<UNK>'] + [word for word, _ in counter.most_common(MAX_VOCAB_SIZE - 2)]
word2idx = {word: idx for idx, word in enumerate(vocab)}

def encode(text):
    return [word2idx.get(token, word2idx['<UNK>']) for token in tokenize(text)][:MAX_LEN]

# ==== DATASET ====
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.encoded = [torch.tensor(encode(text), dtype=torch.long) for text in texts]
        self.labels = torch.tensor(labels, dtype=torch.float32)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.encoded[idx], self.labels[idx]

def collate_fn(batch):
    texts, labels = zip(*batch)
    lengths = [len(seq) for seq in texts]
    padded = pad_sequence(texts, batch_first=True, padding_value=word2idx['<PAD>'])
    return padded, torch.tensor(lengths), torch.tensor(labels)

train_ds = TextDataset(train_texts, train_labels)
val_ds = TextDataset(val_texts, val_labels)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# ==== MODEL (Simple Average Embedding) ====
class MeanEmbedder(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=word2idx['<PAD>'])
        self.fc = nn.Linear(embed_dim, 1)
        self.dropout = nn.Dropout(0.5)
    def forward(self, x, lengths):
        mask = (x != word2idx['<PAD>']).float().unsqueeze(-1)
        x = self.embedding(x)
        x = x * mask
        summed = x.sum(1)
        length = mask.sum(1)
        mean = summed / (length + 1e-8)
        out = self.dropout(mean)
        return torch.sigmoid(self.fc(out)).squeeze(1)

model = MeanEmbedder(len(vocab), EMBED_DIM).to(DEVICE)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ==== TRAINING LOOP ====
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for x, lengths, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        x, lengths, y = x.to(DEVICE), lengths.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        preds = model(x, lengths)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
    avg_loss = total_loss / len(train_ds)
    print(f"Train Loss: {avg_loss:.4f}")

    # Validation
    model.eval()
    val_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, lengths, y in val_loader:
            x, lengths, y = x.to(DEVICE), lengths.to(DEVICE), y.to(DEVICE)
            preds = model(x, lengths)
            loss = criterion(preds, y)
            val_loss += loss.item() * len(y)
            pred_labels = (preds > 0.5).long()
            correct += (pred_labels == y.long()).sum().item()
            total += len(y)
            all_preds.extend(pred_labels.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    avg_val_loss = val_loss / len(val_ds)
    val_acc = correct / total
    print(f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

# ==== FINAL METRICS ====
from sklearn.metrics import classification_report, accuracy_score
print("\nValidation Classification Report:")
print(classification_report(all_labels, all_preds, target_names=le.classes_))
print(f"Validation Accuracy: {accuracy_score(all_labels, all_preds):.4f}")

# ==== SAVE MODEL ====
torch.save(model.state_dict(), os.path.join(MODEL_DIR, "pytorch_fast_meanembed.pt"))
with open(os.path.join(MODEL_DIR, "vocab_fast.pkl"), "wb") as f:
    pickle.dump(word2idx, f)
with open(os.path.join(MODEL_DIR, "label_encoder_fast.pkl"), "wb") as f:
    pickle.dump(le, f)
print("Model and vocabulary saved.")

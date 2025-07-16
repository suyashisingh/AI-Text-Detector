import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# === Paths ===
DATA_PATH = r"C:\Users\hp\Desktop\DMPROJECT\AI-detector\data\processed\combined_dataset.csv"
MODEL_DIR = r"C:\Users\hp\Desktop\DMPROJECT\AI-detector\model"
os.makedirs(MODEL_DIR, exist_ok=True)

# === Load Data ===
df = pd.read_csv(DATA_PATH, low_memory=False)
if 'text' not in df.columns or 'label' not in df.columns:
    raise ValueError("Input CSV must contain 'text' and 'label' columns.")

# === Clean Data ===
df['label'] = df['label'].astype(str).str.strip()
df['text'] = df['text'].astype(str).str.strip()
df = df.dropna(subset=['text', 'label'])
df = df[df['text'] != ""]
df = df[df['label'] != ""]

# === Debug: Print Class Counts and Unique Labels ===
print("Class distribution (before filtering):")
print(df['label'].value_counts(dropna=False))
print("Unique labels:", df['label'].unique())

# === Remove Underrepresented Classes ===
counts = df['label'].value_counts()
valid_labels = counts[counts >= 2].index
df = df[df['label'].isin(valid_labels)]

# === Check Again ===
print("\nClass distribution (after filtering):")
print(df['label'].value_counts())
if len(df['label'].unique()) < 2:
    print("\nError: Only one class remains after filtering. Model training requires at least two classes.")
    exit(1)

X = df['text']
y = df['label']

# === Split Data ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save test indices for evaluation
test_indices_path = os.path.join(MODEL_DIR, "test_indices.joblib")
joblib.dump(X_test.index, test_indices_path)


# === Feature Extraction ===
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2), stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# === Model Training ===
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_tfidf, y_train)

# === Evaluation ===
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# === Save Model and Vectorizer ===
model_path = os.path.join(MODEL_DIR, "logistic_regression_model.joblib")
vectorizer_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
joblib.dump(model, model_path)
joblib.dump(vectorizer, vectorizer_path)
print(f"\nModel saved to: {model_path}")
print(f"Vectorizer saved to: {vectorizer_path}")

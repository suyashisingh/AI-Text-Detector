from flask import Flask, render_template, request, session
import torch
import pickle
import torch.nn as nn
import os
import re
from datetime import datetime
from langdetect import detect, DetectorFactory
from werkzeug.utils import secure_filename
from torch.nn.utils.rnn import pad_sequence

# ==== CONFIG ====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, '..', 'model')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 150
ALLOWED_EXTENSIONS = {'txt'}

# ==== LOAD VOCAB & LABEL ENCODER ====
with open(os.path.join(MODEL_DIR, "vocab_fast.pkl"), "rb") as f:
    word2idx = pickle.load(f)

with open(os.path.join(MODEL_DIR, "label_encoder_fast.pkl"), "rb") as f:
    label_encoder = pickle.load(f)

# ==== TOKENIZER & ENCODER ====
def tokenize(text):
    return text.lower().split()

def encode(text):
    return [word2idx.get(token, word2idx['<UNK>']) for token in tokenize(text)][:MAX_LEN]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ==== MODEL DEFINITION ====
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=word2idx['<PAD>'])
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hidden, _) = self.lstm(packed)
        hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)
        out = self.dropout(hidden_cat)
        return torch.sigmoid(self.fc(out)).squeeze(1)

# ==== LOAD MODEL ====
model = LSTMClassifier(len(word2idx), 128)
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "pytorch_fast_lstm.pt"), map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ==== FLASK APP ====
app = Flask(__name__)
app.secret_key = 'REPLACE_WITH_A_SECURE_RANDOM_STRING_FOR_PRODUCTION'

DetectorFactory.seed = 0  # For consistent langdetect results

def get_metrics(text):
    words = re.findall(r'\b\w+\b', text)
    word_count = len(words)
    ascii_only = all(ord(c) < 128 for c in text)
    try:
        if word_count < 4:
            language = "Unknown"
        else:
            language = detect(text)
            if language in ("so", "sw") and ascii_only:
                language = "en"
    except:
        language = "Unknown"

    ai_flags = []
    # You can define your own AI detection rules here

    return {
        "word_count": word_count,
        "language": language,
        "ai_flags": ai_flags
    }

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    input_text = ""
    file_error = None
    metrics = {}

    if 'past_results' not in session:
        session['past_results'] = []

    if request.method == "POST":
        if 'file' in request.files and request.files['file'].filename != "":
            file = request.files['file']
            if allowed_file(file.filename):
                file_content = file.read().decode('utf-8')
                input_text = file_content
            else:
                file_error = "Only .txt files are allowed!"
        else:
            input_text = request.form.get("text")

        if input_text and input_text.strip() != "" and not file_error:
            encoded = encode(input_text)
            tensor = torch.tensor([encoded], dtype=torch.long).to(DEVICE)
            length = torch.tensor([len(encoded)], dtype=torch.long).to(DEVICE)
            with torch.no_grad():
                pred = model(tensor, length)
                pred_label = "AI-Generated" if pred.item() >= 0.5 else "Human-Written"
                confidence = round(pred.item(), 2)
                result = f"Prediction: {pred_label} (Confidence: {confidence})"
                metrics = get_metrics(input_text)

                entry = {
                    "date": datetime.now().strftime("%Y-%m-%d %I:%M %p"),
                    "result": result,
                    "input": (input_text[:120] + "...") if len(input_text) > 120 else input_text,
                    "metrics": metrics
                }
                session['past_results'] = ([entry] + session['past_results'])[:10]

    return render_template("index.html",
                           result=result,
                           input_text=input_text,
                           file_error=file_error,
                           metrics=metrics,
                           past_results=session.get('past_results', [])
                           )

if __name__ == "__main__":
    app.run(debug=True, port=5001)

import pickle
with open('model/vocab_fast.pkl', 'rb') as f:
    word2idx = pickle.load(f)
print(len(word2idx))  # Should print 5000 to match your model .pt file

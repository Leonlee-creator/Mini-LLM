import numpy as np
import os
import random

with open("train.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Get all unique characters in the text
vocab = sorted(list(set(text)))
vocab_size = len(vocab)

# Create mappings
stoi = {ch: i for i, ch in enumerate(vocab)}
itos = {i: ch for i, ch in enumerate(vocab)}

# Encoder and decoder functions
def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

# Encode the entire dataset
data = encode(text)
data = np.array(data, dtype=np.uint16)  # Save memory

# Split into train and validation (90% train, 10% val)
split_index = int(0.9 * len(data))
train_data = data[:split_index]
val_data = data[split_index:]

# Save to binary files
train_data.tofile("train.bin")
val_data.tofile("val.bin")

# Save the vocab as well
import json
meta = {
    'vocab': vocab,
    'vocab_size': vocab_size,
    'stoi': stoi,
    'itos': itos,
}
with open('meta.json', 'w') as f:
    json.dump(meta, f)

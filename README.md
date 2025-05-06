# Mini LLM - Character-Level Language Model

This is a simple character-level LLM (Large Language Model) built from scratch using PyTorch. It learns from raw text and generates new text based on what it has learned.

##  Features

- Pure Python and PyTorch implementation
- Trains on any plain text file (`input.txt`)
- Saves/loads model weights and vocab
- Generates original text character by character

##  Project Structure

Mini LLM/
│
├── train.py # Trains the model and saves weights
├── generate.py # Loads the model and generates text
├── model.py # Defines the transformer-based LLM
├── input.txt # Raw training text file
├── train.bin # Encoded training data (auto-generated)
├── val.bin # Encoded validation data (auto-generated)
├── meta.json # Stores vocab and mappings (auto-generated)
└── README.md # This file


##  How It Works

1. `train.py`:
   - Reads `input.txt`
   - Tokenizes the text into character IDs
   - Splits into training/validation sets
   - Trains a mini transformer
   - Saves model weights + vocab

2. `generate.py`:
   - Loads the trained model and vocab
   - Starts with a prompt (e.g., `"Elara"`)
   - Generates new text character-by-character

## ⚙️ Requirements

- Python 3.8+
- PyTorch

Install dependencies:

pip install torch

(Model is small, so results may be gibberish without longer training — still fun!)
its still on training.
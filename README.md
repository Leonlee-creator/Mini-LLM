# 🧠 Mini LLM — Character-Level Multilingual Language Model

This is a lightweight character-level LLM (Large Language Model) built from scratch using Python and PyTorch.  
Originally designed for simple story generation, it now supports **Shona**, **English**, and **Venda** translation-style prompts using custom datasets.

---

## ✨ Features

- 🧱 Built from scratch with PyTorch and pure Python.
- 🧠 Learns from **raw text** (fantasy stories, translations, etc.)
- 🔁 Trains on your own `train.txt` and `val.txt`.
- 🌍 Supports **translation-style completion** across 3 languages:
  - Shona  
  - English  
  - Venda
- 💾 Saves model weights, vocab, and metadata.
- 📈 Generates coherent sequences **character-by-character**.

---

## 📁 Project Structure

Mini LLM/
├── train.py # Trains the model and saves weights
├── generate.py # Generates text from the trained model
├── model.py # Transformer-based mini LLM definition
├── train.txt # Mixed training data
├── val.txt # Validation data
├── trained_llm.pth # Saved model weights
├── data/ # Folder for cleaned Shona & Venda corpora
└── README.md # This file

## 🧪 How It Works

### 1. Training (`train.py`)
- Reads `train.txt` and `val.txt`
- Tokenizes characters → integer IDs
- Trains a **mini transformer model** on the data
- Saves model weights as `trained_llm.pth`

### 2. Generation (`generate.py`)
- Loads the model + vocab
- Accepts a prompt (e.g. `"Shona: Ndinokuda.\nEnglish:"`)
- Generates text for continuation (e.g., translation)
- Uses temperature sampling and softmax decoding

---

## 🌍 Example Translation-Style Prompts

Shona: Ndinokuda.
English: I love you.
Venda: Ndi a ni funa.

Output will vary depending on training data and tokens.

---

## 🛠 Requirements

- Python 3.8+
- PyTorch

Install PyTorch:
pip install torch


---

## 📌 Notes

- This model is small and lightweight. Results improve as you train longer and feed more data.
- Supports multi-language prompts using custom datasets.
- Future improvements: Attention scoring, better tokenization, grammar correction.

---

## 🔮 Future Ideas

- Word-level tokenizer (instead of character-based).
- Expand to Swahili, Zulu, or other African languages.
- Real-time translation UI or agent with microphone input.


import torch
from model import LLM

with open("train.txt", "r") as f:
    train_text = f.read()

with open("val.txt", "r") as f:
    val_text = f.read()

text = train_text + val_text  

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Hyperparameters 
context_length = 64
device = 'cuda' if torch.cuda.is_available() else 'cpu'


model = LLM(vocab_size, embedding_dim=64, context_length=context_length)
model.load_state_dict(torch.load("trained_llm.pth", map_location=device))
model.to(device)
model.eval()

def generate_text(starting_text="Elara", max_new_tokens=300):
    context = torch.tensor(encode(starting_text), dtype=torch.long, device=device)[None, :]
    
    for _ in range(max_new_tokens):
        context_condensed = context[:, -context_length:]
        logits, _ = model(context_condensed)
        logits = logits[:, -1, :]  
        temperature = 1.0  
        probs = torch.nn.functional.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        context = torch.cat((context, next_token), dim=1)

    return decode(context[0].tolist())

generated = generate_text("Elara", 300)
print("=== Generated Text ===")
print(generated)

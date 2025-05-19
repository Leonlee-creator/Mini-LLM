import torch
from transformers import pipeline
from model import LLM


print("⏳ Loading FLAN-T5 model...")
translator = pipeline("text2text-generation", model="google/flan-t5-small")
print("✅ FLAN-T5 ready.")

use_custom_llm = False 
if use_custom_llm:
    with open("train.txt", "r", encoding='utf-8') as f:
        train_text = f.read()
    with open("val.txt", "r", encoding='utf-8') as f:
        val_text = f.read()

    text = train_text + val_text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    encode = lambda s: [stoi[c] for c in s if c in stoi]
    decode = lambda l: ''.join([itos[i] for i in l if i in itos])

    context_length = 128
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LLM(vocab_size, embedding_dim=128, context_length=context_length)
    model.load_state_dict(torch.load("trained_llm.pth", map_location=device))
    model.to(device)
    model.eval()

    def generate_with_local_llm(starting_text="Shona: Ndinokuda.\nEnglish:", max_new_tokens=300):
        context = torch.tensor(encode(starting_text), dtype=torch.long, device=device)[None, :]
        for _ in range(max_new_tokens):
            context_condensed = context[:, -context_length:]
            logits, _ = model(context_condensed)
            logits = logits[:, -1, :]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            context = torch.cat((context, next_token), dim=1)
        return decode(context[0].tolist())


prompts = [
    "Translate Shona to English: Ndinokuda",
    "Translate English to Shona: Thank you",
    "Translate English to Venda: I love you",
    "Translate Venda to English: Ndi a ni funa",
    "Translate Shona to English: Ndatenda",
    "Translate English to Venda: I am happy"
]


for i, prompt in enumerate(prompts, 1):
    print(f"--- Test {i} ---")
    print(f"Prompt: {prompt}")
    
    if use_custom_llm:
        print("Generated (Local LLM):")
        print(generate_with_local_llm(prompt))
    else:
        result = translator(prompt, max_new_tokens=100)[0]['generated_text']
        print("Generated (FLAN-T5):")
        print(result)
    
    print()

from transformers import pipeline
from model import LLM
import torch

print("⏳ Loading FLAN-T5 model...")
translator = pipeline("text2text-generation", model="google/flan-t5-small")
print("✅ FLAN-T5 ready.")


flan_t5 = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    device=0,  # GPU
    torch_dtype=torch.float16,  # Half-precision
    truncation=True,
    max_length=128  # Limit output length
)
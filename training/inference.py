from transformers import pipeline, T5ForConditionalGeneration, T5Tokenizer
import torch


def get_translator(source_lang, target_lang):
    model_path = f"models/{source_lang}_to_{target_lang}"
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = T5Tokenizer.from_pretrained(model_path)

    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
        torch_dtype=torch.float16 if device == 0 else torch.float32
    )


if __name__ == "__main__":
    en_to_shi = get_translator("English", "Shona")
    shi_to_ven = get_translator("Shona", "Venda")


    english_text = "How are you today?"
    shona_result = en_to_shi(f"Translate English to Shona: {english_text}", max_length=50)
    print(f"Shona: {shona_result[0]['generated_text']}")


    venda_result = shi_to_ven(f"Translate Shona to Venda: {shona_result[0]['generated_text']}", max_length=50)
    print(f"Venda: {venda_result[0]['generated_text']}")
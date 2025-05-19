from transformers import T5ForConditionalGeneration, T5Tokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer
import torch
from torch.utils.data import Dataset, DataLoader
import re
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


class TurboConfig:
    BASE_MODEL = "google/flan-t5-small"
    BATCH_SIZE = 16
    GRAD_ACCUM_STEPS = 2
    LEARNING_RATE = 2e-4
    NUM_EPOCHS = 4
    MAX_LENGTH = 64
    OUTPUT_DIR = "turbo_models/"
    NUM_WORKERS = 4
    DATA_DIR = str(Path(__file__).parent.parent)


def turbo_parse(file_name):
    """Load and parse training data from project root"""
    file_path = os.path.join(TurboConfig.DATA_DIR, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip().split('\n\n')

    parsed_data = []
    for entry in content:
        if not entry.strip():
            continue
        lines = [line.strip() for line in entry.split('\n') if line.strip()]
        if len(lines) >= 3:
            shona = re.sub(r'^Shona:\s*', '', lines[0], flags=re.IGNORECASE)
            english = re.sub(r'^English:\s*', '', lines[1], flags=re.IGNORECASE)
            venda = re.sub(r'^Venda:\s*', '', lines[2], flags=re.IGNORECASE)
            parsed_data.append({
                'shona': shona,
                'english': english,
                'venda': venda
            })
    return parsed_data


class TurboDataset(Dataset):
    def __init__(self, data, tokenizer, direction):
        self.data = data
        self.tokenizer = tokenizer
        self.source, self.target = direction.split('-')
        self.examples = self._prep_examples()

    def _prep_examples(self):
        with ThreadPoolExecutor(max_workers=4) as executor:
            return list(executor.map(self._create_example, self.data))

    def _create_example(self, entry):
        try:
            source_text = entry[self.source.lower()]
            target_text = entry[self.target.lower()]
            prompt = f"Translate {self.source} to {self.target}: {source_text}"
            return prompt, target_text
        except KeyError:
            print(f"Warning: Missing translation pair for {self.source}-{self.target}")
            return "", ""

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        prompt, target = self.examples[idx]
        inputs = self.tokenizer(
            prompt,
            max_length=TurboConfig.MAX_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        labels = self.tokenizer(
            target,
            max_length=TurboConfig.MAX_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        ).input_ids
        labels[labels == self.tokenizer.pad_token_id] = -100


        return {
            'input_ids': inputs.input_ids.squeeze(),
            'attention_mask': inputs.attention_mask.squeeze(),
            'labels': labels.squeeze()
        }


def turbo_collate(batch):
    batch = [item for item in batch if item['input_ids'].numel() > 0]


    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }


def turbo_train(train_data, val_data, direction):
    print(f"\n⚡ Turbo Training {direction} model...")


    tokenizer = T5Tokenizer.from_pretrained(
        TurboConfig.BASE_MODEL,
        legacy=False,
        model_max_length=TurboConfig.MAX_LENGTH
    )
    model = T5ForConditionalGeneration.from_pretrained(TurboConfig.BASE_MODEL)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")


    train_set = TurboDataset(train_data, tokenizer, direction)
    val_set = TurboDataset(val_data, tokenizer, direction)


    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(TurboConfig.OUTPUT_DIR, direction.replace('-', '_to_')),
        eval_strategy="steps",  # Changed from evaluation_strategy to eval_strategy
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        learning_rate=TurboConfig.LEARNING_RATE,
        per_device_train_batch_size=TurboConfig.BATCH_SIZE,
        gradient_accumulation_steps=TurboConfig.GRAD_ACCUM_STEPS,
        num_train_epochs=TurboConfig.NUM_EPOCHS,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=TurboConfig.NUM_WORKERS,
        logging_steps=50,
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        save_total_limit=2
    )


    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=val_set,
        tokenizer=tokenizer,
        data_collator=turbo_collate
    )


    print(f"Starting training for {direction}...")
    trainer.train()


    model.save_pretrained(os.path.join(TurboConfig.OUTPUT_DIR, direction.replace('-', '_to_')))
    print(f"✅ {direction} model saved!")


if __name__ == "__main__":
    os.makedirs(TurboConfig.OUTPUT_DIR, exist_ok=True)

    try:
        print("Loading training data...")
        train_data = turbo_parse("train.txt")
        val_data = turbo_parse("val.txt")
        print(f"Loaded {len(train_data)} training and {len(val_data)} validation examples")

        directions = [
            'english-shona', 'english-venda',
            'shona-english', 'venda-english',
            'shona-venda', 'venda-shona'
        ]

        for direction in directions:
            turbo_train(train_data, val_data, direction)

        print("\n🎉 All models trained successfully!")
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        raise
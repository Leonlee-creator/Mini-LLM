from transformers import T5ForConditionalGeneration, T5Tokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer
import torch
from torch.utils.data import Dataset
import re
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import transformers
import warnings
from packaging import version

# Suppress future warnings
warnings.simplefilter('ignore', FutureWarning)


class TurboConfig:
    BASE_MODEL = "google/flan-t5-small"
    BATCH_SIZE = 8  # Reduced for better stability
    GRAD_ACCUM_STEPS = 4  # Memory optimization
    LEARNING_RATE = 3e-4  # Adjusted learning rate
    NUM_EPOCHS = 4  # Slightly increased epochs
    MAX_LENGTH = 128  # Increased max length
    OUTPUT_DIR = "turbo_models/"
    NUM_WORKERS = min(4, os.cpu_count())  # Optimal workers for system
    DATA_DIR = str(Path(__file__).parent.parent)
    RESUME_CHECKPOINT = True  # Smart checkpoint resuming


def turbo_parse(file_name):
    """Enhanced data parser with validation"""
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
            try:
                shona = re.sub(r'^Shona:\s*', '', lines[0], flags=re.IGNORECASE)
                english = re.sub(r'^English:\s*', '', lines[1], flags=re.IGNORECASE)
                venda = re.sub(r'^Venda:\s*', '', lines[2], flags=re.IGNORECASE)
                if shona and english and venda:  # Validate non-empty strings
                    parsed_data.append({
                        'shona': shona,
                        'english': english,
                        'venda': venda
                    })
            except Exception as e:
                print(f"⚠️ Error parsing entry: {e}")
                continue
    return parsed_data


class TurboDataset(Dataset):
    def __init__(self, data, tokenizer, direction):
        self.data = data
        self.tokenizer = tokenizer
        self.source, self.target = direction.split('-')
        self.examples = self._prep_examples()

    def _prep_examples(self):
        with ThreadPoolExecutor(max_workers=TurboConfig.NUM_WORKERS) as executor:
            return list(executor.map(self._create_example, self.data))

    def _create_example(self, entry):
        try:
            source_text = entry[self.source.lower()]
            target_text = entry[self.target.lower()]
            prompt = f"translate {self.source} to {self.target}: {source_text}"
            return prompt, target_text
        except KeyError:
            print(f"⚠️ Missing translation pair for {self.source}-{self.target}")
            return None, None

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        prompt, target = self.examples[idx]
        if prompt is None or target is None:
            return {'input_ids': torch.tensor([]), 'attention_mask': torch.tensor([]), 'labels': torch.tensor([])}

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
    if not batch:
        return {}

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

    # Initialize tokenizer
    tokenizer = T5Tokenizer.from_pretrained(
        TurboConfig.BASE_MODEL,
        legacy=False,
        model_max_length=TurboConfig.MAX_LENGTH
    )

    # Handle checkpoints
    output_dir = os.path.join(TurboConfig.OUTPUT_DIR, direction.replace('-', '_to_'))
    checkpoint = None
    if TurboConfig.RESUME_CHECKPOINT and os.path.exists(output_dir):
        try:
            checkpoints = [f for f in os.listdir(output_dir) if f.startswith('checkpoint-')]
            if checkpoints:
                checkpoint = os.path.join(output_dir, max(checkpoints, key=lambda x: int(x.split('-')[-1])))
                print(f"🔄 Resuming from checkpoint: {checkpoint}")
        except Exception as e:
            print(f"⚠️ Checkpoint loading warning: {e}")
            checkpoint = None

    # Load model
    model = T5ForConditionalGeneration.from_pretrained(
        TurboConfig.BASE_MODEL if checkpoint is None else checkpoint,
        ignore_mismatched_sizes=True
    )

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    # Dataset preparation
    train_set = TurboDataset(train_data, tokenizer, direction)
    val_set = TurboDataset(val_data, tokenizer, direction)

    # Version-aware parameter names
    is_new_version = version.parse(transformers.__version__) >= version.parse("4.26.0")
    eval_key = "eval_strategy" if is_new_version else "evaluation_strategy"
    save_key = "save_strategy" if is_new_version else "save_strategy"

    # Training arguments
    training_args = {
        "output_dir": output_dir,
        eval_key: "steps",
        "eval_steps": 500,
        save_key: "steps",
        "save_steps": 500,
        "learning_rate": TurboConfig.LEARNING_RATE,
        "per_device_train_batch_size": TurboConfig.BATCH_SIZE,
        "gradient_accumulation_steps": TurboConfig.GRAD_ACCUM_STEPS,
        "num_train_epochs": TurboConfig.NUM_EPOCHS,
        "predict_with_generate": True,
        "fp16": torch.cuda.is_available(),
        "dataloader_num_workers": TurboConfig.NUM_WORKERS,
        "logging_steps": 100,
        "warmup_ratio": 0.1,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "report_to": "none",
        "save_total_limit": 3,
        "gradient_checkpointing": torch.cuda.is_available(),
        "optim": "adamw_torch",  # Using AdamW instead of Adafactor
        "no_cuda": not torch.cuda.is_available()
    }

    training_args = Seq2SeqTrainingArguments(**training_args)

    # Trainer setup
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=val_set,
        tokenizer=tokenizer,
        data_collator=turbo_collate
    )

    # Start training
    print(f"Starting training for {direction}...")
    try:
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_metrics("train", metrics)

        # Evaluate and save metrics
        eval_metrics = trainer.evaluate()
        trainer.save_metrics("eval", eval_metrics)

        print(f"✅ Training completed for {direction} with metrics:")
        print(f"Training loss: {metrics['train_loss']:.4f}")
        print(f"Evaluation loss: {eval_metrics['eval_loss']:.4f}")

    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise

    # Save final model
    trainer.save_model(output_dir)
    print(f"💾 Model saved to {output_dir}")


if __name__ == "__main__":
    os.makedirs(TurboConfig.OUTPUT_DIR, exist_ok=True)

    try:
        print("🔍 Loading training data...")
        train_data = turbo_parse("train.txt")
        val_data = turbo_parse("val.txt")
        print(f"📊 Loaded {len(train_data)} training and {len(val_data)} validation examples")

        directions = [
            'english-shona', 'english-venda',
            'shona-english', 'venda-english',
            'shona-venda', 'venda-shona'
        ]

        for direction in directions:
            turbo_train(train_data, val_data, direction)

        print("\n🎉 All models trained successfully!")
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        raise
import argparse
import os
import random

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "google/gemma-3-4b-it"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataset(seed: int):
    examples = [
        {
            "instruction": "Write a Python function that returns Fibonacci numbers up to n.",
            "response": "def fib_up_to(n):\n    out, a, b = [], 0, 1\n    while a <= n:\n        out.append(a)\n        a, b = b, a + b\n    return out",
        },
        {
            "instruction": "Explain why we use a validation split while fine-tuning.",
            "response": "A validation split helps track generalization and prevents tuning decisions from overfitting the training set.",
        },
        {
            "instruction": "Give three practical tips to stabilize LLM fine-tuning.",
            "response": "Use a smaller learning rate, monitor validation loss each epoch, and start with parameter-efficient fine-tuning like LoRA.",
        },
        {
            "instruction": "What does gradient accumulation solve?",
            "response": "It simulates a larger effective batch size by combining gradients from several smaller forward/backward passes.",
        },
        {
            "instruction": "Write one sentence describing transfer learning.",
            "response": "Transfer learning adapts pretrained knowledge to a new task with less data and lower training cost.",
        },
        {
            "instruction": "How does LoRA reduce memory usage?",
            "response": "LoRA freezes base model weights and trains only small low-rank adapter matrices, cutting trainable parameters dramatically.",
        },
        {
            "instruction": "When should I stop training?",
            "response": "Stop when validation metrics plateau or degrade consistently across checkpoints.",
        },
        {
            "instruction": "Define catastrophic forgetting in one sentence.",
            "response": "Catastrophic forgetting is when a model loses previously learned capabilities while adapting to new data.",
        },
    ]

    def format_example(item):
        return f"<start_of_turn>user\n{item['instruction']}<end_of_turn>\n<start_of_turn>model\n{item['response']}<end_of_turn>"

    texts = [format_example(item) for item in examples]
    dataset = Dataset.from_dict({"text": texts})
    return dataset.train_test_split(test_size=0.25, seed=seed)


def tokenize_dataset(split_dataset, tokenizer, max_length: int):
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    return split_dataset.map(tokenize, batched=True, remove_columns=["text"])


def load_model_and_tokenizer(model_name: str, hf_token: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"token": hf_token}
    if torch.cuda.is_available():
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model_kwargs.update(
            {
                "device_map": "auto",
                "quantization_config": quantization_config,
            }
        )
    else:
        model_kwargs.update({"device_map": None, "torch_dtype": torch.float32})

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    if torch.cuda.is_available():
        model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft_config)
    return model, tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Gemma 3 4B with LoRA")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--output-dir", default="outputs/gemma-3-4b-lora")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    split_dataset = build_dataset(args.seed)
    print({"train_rows": len(split_dataset["train"]), "eval_rows": len(split_dataset["test"])})

    if args.dry_run:
        print("Dry run complete. Dataset creation and split succeeded.")
        return

    if not args.hf_token:
        raise ValueError("HF token is required. Set HF_TOKEN or pass --hf-token for Gemma model access.")

    model, tokenizer = load_model_and_tokenizer(args.model_name, args.hf_token)
    tokenized = tokenize_dataset(split_dataset, tokenizer, args.max_length)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=4,
        warmup_ratio=0.05,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=1,
        report_to="none",
        fp16=torch.cuda.is_available(),
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=collator,
    )

    trainer.train()

    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    prompt = "<start_of_turn>user\nExplain transfer learning in one sentence.<end_of_turn>\n<start_of_turn>model\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=48, do_sample=True, top_p=0.95, temperature=0.8)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()

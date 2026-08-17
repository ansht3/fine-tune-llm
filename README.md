# Fine-Tune-LLM (End-to-End)

This repository contains a fully original, end-to-end example for fine-tuning a small **open-source GPT-2 style language model**.

To keep the workflow reproducible in restricted environments, the notebook trains:

- a local tokenizer from scratch
- a tiny GPT-2 architecture from scratch
- then fine-tunes it on a custom instruction dataset

## What is included

- `notebooks/fine_tune_tiny_gpt2.ipynb` — complete offline fine-tuning notebook
- `Fine_Tuning_LLM_Presentation.md` — human-style presentation notes/slides
- `requirements.txt` — Python dependencies

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run the notebook

```bash
jupyter notebook
```

Open `notebooks/fine_tune_tiny_gpt2.ipynb` and run all cells.

## Notebook workflow

1. Build a small instruction/response dataset
2. Train a WordLevel tokenizer locally (offline)
3. Tokenize train/eval splits
4. Initialize a tiny GPT-2 model with `GPT2Config`
5. Fine-tune with Hugging Face `Trainer`
6. Save artifacts and run a generation sanity check

## Validation performed

The notebook itself was executed end-to-end using:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/fine_tune_tiny_gpt2.ipynb
```

## Notes

- This is intentionally compact for fast local experimentation.
- For stronger performance, scale up dataset quality/size and tune hyperparameters.

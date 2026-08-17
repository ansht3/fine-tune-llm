# Fine-Tuning an Open-Source LLM (Tiny Offline GPT-2)

## Slide 1 — Goal
- Build an end-to-end fine-tuning project from scratch
- Keep it lightweight and runnable on a normal laptop
- Show every step clearly: data → tokenizer → model → training → inference

## Slide 2 — Why this model setup?
- Used an open-source GPT-2 style transformer architecture
- Built a tiny config so training can run quickly
- No external model download required, so it works in restricted environments

## Slide 3 — Dataset design
- Created a compact instruction-response dataset
- Focused on practical ML/coding prompts
- Formatted each sample with:
  - `### Instruction:`
  - `### Response:`

## Slide 4 — Tokenization approach
- Trained a local WordLevel tokenizer from the dataset
- Added special tokens (`[PAD]`, `[UNK]`, `[BOS]`, `[EOS]`)
- Tokenized with truncation and fixed sequence length

## Slide 5 — Fine-tuning pipeline
- Initialized tiny GPT-2 with `GPT2Config`
- Used Hugging Face `Trainer` for training/eval
- Logged progress, evaluated each epoch, and saved checkpoints

## Slide 6 — Outputs
- Saved final model and tokenizer artifacts
- Ran a generation sanity check on an unseen instruction
- Verified complete workflow is functional end-to-end

## Slide 7 — Next upgrades
- Increase dataset size and quality
- Add richer eval metrics (not just loss)
- Try PEFT methods (LoRA/QLoRA)
- Track experiments for comparison and reproducibility

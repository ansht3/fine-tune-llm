# Fine-Tune-LLM (Gemma 4B End-to-End)

This repository provides an original end-to-end starter for fine-tuning a **Gemma 4B-class model** with LoRA.

## What is included

- `/home/runner/work/fine-tune-llm/fine-tune-llm/fine_tune_gemma_4b.py` — standalone Python fine-tuning script for `google/gemma-3-4b-it`
- `/home/runner/work/fine-tune-llm/fine-tune-llm/notebooks/fine_tune_gemma_4b.ipynb` — notebook version of the workflow
- `/home/runner/work/fine-tune-llm/fine-tune-llm/Fine_Tuning_LLM_Presentation.pptx` — PowerPoint presentation
- `/home/runner/work/fine-tune-llm/fine-tune-llm/requirements.txt` — dependencies

## Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Fine-tune from Python script

```bash
export HF_TOKEN=<your_huggingface_token>
python fine_tune_gemma_4b.py --epochs 1 --batch-size 1 --output-dir outputs/gemma-3-4b-lora
```

## Run notebook

```bash
jupyter notebook
```

Open `/home/runner/work/fine-tune-llm/fine-tune-llm/notebooks/fine_tune_gemma_4b.ipynb`.

## Notes

- Full fine-tuning requires model access permissions for `google/gemma-3-4b-it` and sufficient GPU memory.
- For quick validation of repository setup without model download, run:

```bash
python fine_tune_gemma_4b.py --dry-run
jupyter nbconvert --to notebook --execute --inplace notebooks/fine_tune_gemma_4b.ipynb
```

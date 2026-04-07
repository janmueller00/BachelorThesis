# Rating Privacy Policies with Large Language Models

Bachelor's Thesis - Jan Müller  
Universität Duisburg-Essen, Wintersemester 2025/2026

---

## Overview

This repository contains the full implementation for the thesis *Rating Privacy Policies with Large Language Models*. The project fine-tunes two architecturally distinct language models, LegalBERT and LLaMA 3 8B, on the OPP-115 privacy policy corpus to classify policy segments across eight privacy practice categories. Segment-level classifications are aggregated into a document-level privacy-friendliness score through a rule-based scoring pipeline. A third evaluation approach uses LLaMA 3 8B in a zero-shot setting to directly score policies without fine-tuning.

---

## Dataset

This project uses the **OPP-115 corpus** (Wilson et al., 2016). The only files included in this repository are the threshold 0.75 consolidation files and the sanitized policies needed for using this project. Any other files must be obtained at: 

https://usableprivacy.org/data


---

## Pipeline

### Step 0 - Frequency Analysis

Run the frequency analysis to inspect attribute value distributions before deciding on reductions:

```bash
cd preprocess_data
python analysis.py
# Output written to frequency_analysis.txt
```

### Step 1 - Preprocessing

Preprocess OPP-115 into train/val/test splits with the reduced annotation schema:

```bash
cd preprocess_data
python preprocessing.py
# Output: data/train.json, data/val.json, data/test.json, data/label_schema.json
```

### Step 2 - Train LegalBERT

```bash
cd train_legalbert
python train_legalbert.py
# Best checkpoint saved to models/legalbert_best/
# Metrics saved to results/
```

### Step 3 - Train LLaMA 3 8B (optional)

Requires the model `meta-llama/Meta-Llama-3-8B-Instruct` from HuggingFace (gated access required).

```bash
cd train_llama
python train_llama.py
# Checkpoints saved to models/llama_checkpoints/
# Training resumes automatically from latest checkpoint if interrupted
```

**Note:** The best performing checkpoint was checkpoint-100 (approximately 0.5 epochs). Use `LORA_PATH = "./models/llama_checkpoints/checkpoint-100"` in `llama_inference.py` to load it.

### Step 4 - Inference and Scoring

Run LegalBERT inference on a policy file:

```bash
cd inference
python legalbert_inference.py --file path/to/policy.txt --score --output results/output.json
```

Run fine-tuned LLaMA inference on a policy file:

```bash
python llama_inference.py --file path/to/policy.txt --score --output results/output.json
```

Both scripts accept `--text "segment text"` for single-segment classification without scoring.

The scoring pipeline (`scoring.py`) is shared between both inference scripts and is imported automatically. It aggregates the best observed attribute value per head across all segments into a document-level privacy-friendliness score on a 0-10 scale.

### Step 5 - Generalization Evaluation

Place segmented modern policy files and their annotation files in `modern_policies/`. Each policy requires two files:

- `<name>.txt` - policy text with `|||` as segment delimiter
- `<name>_ann.txt` - annotations, one line per positive label: `segment_id,Category Name` (segment_id starting at 0)

Then run:

```bash
cd inference
python generalization.py
# Results saved to results/generalization_results.json
```

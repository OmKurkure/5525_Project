# 5525_Project

# Cross-Lingual Transfer through Phonetic Representations

Code for the project on cross-lingual transfer for low-resource intent classification through phonetic representations. Both text and speech are converted to a shared phonetic representation, and a model is trained on the combined data to perform intent classification on the MASSIVE 1.1 dataset.

We evaluate on two language pairs: French--Italian and French--Haitian Creole. French is the high-resource source language; Italian and Haitian Creole are the low-resource targets.

## Repository Structure

The repo is organized into three top-level directories, one per model family.

### `CamemBERT/`

CamemBERT-based experiments. The model uses CamemBERT's pretrained French weights with the input vocabulary resized for the phone vocabulary.

- `FinetuningData/` — fine-tuning data for intent classification, organized by language:
  - `Haitian/` — Haitian Creole 400 training and 100 test examples
  - `Italian/` — Italian 400 training and 100 test examples
- `PretrainingData/` — MLM pretraining data, organized by language pair:
  - `FrHt/` — French--Haitian Creole pretraining data
  - `FrIt/` — French--Italian pretraining data
- `finetune_cam.py` — fine-tunes CamemBERT directly on raw text (used for the text-only CamemBERT baseline; no phones or phonemes)
- `train_cam_haitian.py` — MLM pretraining followed by fine-tuning for the French--Haitian pair
- `train_cam_italian.py` — MLM pretraining followed by fine-tuning for the French--Italian pair

### `Encoder/`

From-scratch transformer encoder experiments. A small transformer is trained from scratch on phone or phoneme sequences.

- `FinetuningData/` — fine-tuning data for intent classification. Similar structure as Finetuning Dir in CamemBERT directory, only difference it does not raw italian text datasets since the encoder does not require them. 
- `PretrainingData/` — MLM pretraining data:
  - `FrHt/` — French--Haitian Creole pretraining data
  - `FrIt/` — French--Italian pretraining data
- `train_haitian.py` — MLM pretraining followed by fine-tuning for the French--Haitian pair
- `train_italian.py` — MLM pretraining followed by fine-tuning for the French--Italian pair

### `Text_Baseline/`

LLM baselines fine-tuned on raw text only (no phonetic conversion).

- `train.py` — fine-tunes Llama and Qwen models on intent classification using LoRA
- `data/` — 400 training and 100 test examples for each of Haitian Creole and Italian

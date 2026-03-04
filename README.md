# Multitask NLP Framework for Content Moderation with LIME Explainability

A comprehensive multitask natural language processing framework for simultaneous **emotion detection**, **hate speech identification**, and **violence classification**, integrating back-translation-based data augmentation and LIME-based explainability.

---

## Overview

This project addresses three parallel text classification tasks under a unified experimental pipeline:

- **Emotion Detection** — 6 classes: Sadness, Joy, Love, Anger, Fear, Surprise
- **Hate Speech Detection** — 3 classes: Hate Speech, Offensive Speech, Neither
- **Violence Detection** — 5 classes: Harmful Traditional Practice, Physical Violence, Economic Violence, Emotional Violence, Sexual Violence

The framework systematically evaluates classical ML models, deep learning architectures (Bi-LSTM, Bi-GRU), and transformer-based models (BERT, DistilBERT, RoBERTa) under consistent data conditions, with a focus on mitigating class imbalance through back-translation augmentation.

---

## Key Results

| Model | Overall Accuracy | Overall F1-Score |
|---|---|---|
| LightGBM (Imbalanced) | 0.9229 | 0.8303 |
| LightGBM (Balanced) | 0.9307 | 0.9216 |
| Bi-GRU | 0.9951 | 0.9951 |
| BERT | 0.9964 | 0.9964 |
| **RoBERTa (Optimized)** | **0.9976** | **0.9976** |

---

## Project Structure
```
multitask-nlp-content-moderation/
├── notebooks/                   # Step-by-step Jupyter notebooks
├── results/                     # Confusion matrices, LIME plots, comparison figures
└── README.md
```

## Requirements

Key dependencies:
```
torch
transformers
tensorflow
keras
scikit-learn
xgboost
lightgbm
catboost
lime
pandas
numpy
matplotlib
seaborn
sentencepiece
sacremoses
```

## Methodology Summary

### Data Augmentation
Back-translation via French is applied exclusively to minority classes using the formula:

$$x^{\text{aug}} = \mathcal{T}^{-1}_{FR \to EN}\!\left(\mathcal{T}_{EN \to FR}(x)\right)$$

Augmented samples are included **only in the training set** to prevent data leakage.

### Models Evaluated
- **Classical ML:** SVM, Complement Naïve Bayes, Random Forest, XGBoost, LightGBM, CatBoost
- **Deep Learning:** Bi-LSTM, Bi-GRU (64 hidden units, dropout=0.5, Adam optimizer)
- **Transformers:** BERT, DistilBERT, RoBERTa (fine-tuned, LR=2e-5, 20 epochs)

### Explainability
LIME (Local Interpretable Model-Agnostic Explanations) is applied to the best-performing RoBERTa model to identify per-word contributions to individual predictions across all three tasks.

---

## Datasets

All datasets are publicly available on [Kaggle](https://www.kaggle.com):

| Dataset | Classes | Link |
|---|---|---|
| Emotion Detection | 6 (Sadness, Joy, Love, Anger, Fear, Surprise) | [Emotions Dataset](https://www.kaggle.com/datasets/nelgiriyewithana/emotions) |
| Violence Detection | 5 (Harmful Traditional Practice, Physical Violence, Economic Violence, Emotional Violence, Sexual Violence) | [Gender Based Violence Tweet Classification](https://www.kaggle.com/datasets/gauravduttakiit/gender-based-violence-tweet-classification) |
| Hate Speech Detection | 3 (Hate Speech, Offensive Speech, Neither) | [Hate Speech and Offensive Language Dataset](https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset) |

> **Note:** Due to Kaggle's terms of use, raw datasets are not included in this repository.
> Download each dataset from the links above and place the CSV files in `data/raw/` as follows:
>
> ```
> data/
> └── raw/
>     ├── emotion_dataset.csv
>     ├── violence_dataset.csv
>     └── hate_speech_dataset.csv
> ```

---

## Usage

### Step 1 — Preprocessing
```bash
jupyter notebook notebooks/01_preprocessing.ipynb
```
Handles column removal, label standardization, controlled downsampling of overrepresented classes, and label encoding.

### Step 2 — Back-Translation Augmentation
```bash
jupyter notebook notebooks/02_augmentation_backtranslation.ipynb
```
Uses Helsinki-NLP MarianMT models (`opus-mt-en-fr`, `opus-mt-fr-en`) via HuggingFace Transformers to generate paraphrased samples for minority classes only.

### Step 3 — Classical ML Models (Imbalanced)
```bash
jupyter notebook notebooks/03_ml_models_imbalanced.ipynb
```
Trains SVM, Complement Naïve Bayes, Random Forest, XGBoost, LightGBM, and CatBoost on original imbalanced data to establish baselines.

### Step 4 — Classical ML Models (Balanced)
```bash
jupyter notebook notebooks/04_ml_models_balanced.ipynb
```
Retrains the same ML models on the augmented balanced datasets to quantify the impact of data balancing.

### Step 5 — Deep Learning Models
```bash
jupyter notebook notebooks/05_deep_learning_bilstm_bigru.ipynb
```
Trains Bi-LSTM and Bi-GRU architectures with shared embeddings on balanced augmented data.

### Step 6 — Transformer Models (BERT & DistilBERT)
```bash
jupyter notebook notebooks/06_transformers_bert_distilbert.ipynb
```
Fine-tunes BERT and DistilBERT using HuggingFace AutoTokenizer with max_length=50.

### Step 7 — RoBERTa Fine-Tuning
```bash
jupyter notebook notebooks/07_roberta_finetuned.ipynb
```
Optimized fine-tuning of RoBERTa with 512/256 hidden units, dropout=0.2, clipnorm=1.0, and 20 training epochs.

### Step 8 — LIME Explainability Analysis
```bash
jupyter notebook notebooks/08_lime_explainability.ipynb
```
Applies LIME to the best-performing RoBERTa model to generate per-word importance scores across all three tasks.

---

## Methodology Summary

### Data Augmentation
Back-translation via French is applied exclusively to minority classes. Augmented samples are included **only in the training set** to prevent data leakage. An 80:20 stratified train-test split is used across all tasks.

### Class Balancing Results

| Dataset | Class | Original | After Augmentation |
|---|---|---|---|
| Emotion | Surprise | 14,972 | 29,944 |
| Violence | Harmful Traditional Practice | 188 | 12,000 |
| Violence | Physical Violence | 5,946 | 18,000 |
| Violence | Economic Violence | 217 | 12,000 |
| Violence | Emotional Violence | 651 | 12,000 |
| Hate Speech | Hate Speech | 1,430 | 9,000 |
| Hate Speech | Neither | 4,163 | 9,000 |

### Models Evaluated

| Category | Models |
|---|---|
| Classical ML | SVM, Complement Naïve Bayes, Random Forest, XGBoost, LightGBM, CatBoost |
| Deep Learning | Bi-LSTM, Bi-GRU (64 hidden units, dropout=0.5, Adam optimizer, 10 epochs) |
| Transformers | BERT, DistilBERT, RoBERTa (LR=2e-5, Adam optimizer) |

### Explainability
LIME (Local Interpretable Model-Agnostic Explanations) is applied to the best-performing RoBERTa model to generate per-word importance scores. It works by generating perturbed variations of an input text instance, observing changes in predicted probabilities, and fitting a locally interpretable surrogate model to approximate the transformer's behavior around that instance.

---

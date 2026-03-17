# Pet Triage — Multimodal Veterinary Urgency System

A deep learning system for veterinary triage: upload a pet image and describe the situation to get **urgency priority (1-5)**, **dog breed**, and **extracted clinical entities** (symptoms, medications, etc.).

## How It Works

The system has two independent models that run in parallel:

**Image model (Keras/TensorFlow):** Takes a photo of the dog, runs it through an EfficientNetB3 CNN pretrained on ImageNet and fine-tuned on 120 dog breeds. Outputs the predicted breed with confidence scores.

**Text model (PyTorch/BioBERT):** Takes the owner's free-text description (e.g. "my dog collapsed, gums are white, hasn't eaten in 2 days"). A pretrained BioBERT transformer processes the text and produces two outputs simultaneously:
- **Urgency classification** — which of 5 triage levels (Red/Immediate to Blue/Non-Urgent)
- **Named Entity Recognition (NER)** — extracts clinical entities like symptoms, medications, breed, age, duration from the text

The Streamlit app calls both models and displays the combined results.

## Text Model — How It Works

### The Problem

A pet owner texts something like: *"My 5-year-old Golden Retriever has been vomiting for 2 days and won't eat. He takes insulin for diabetes."*

From this single message, we need to:
1. **Classify urgency** — how quickly does this pet need to be seen? (Level 3: Yellow / Urgent)
2. **Extract clinical entities** — what specific information is in the text?
   - `[AGE] 5-year-old`
   - `[BREED] Golden Retriever`
   - `[SYMPTOM] vomiting`
   - `[SYMPTOM] won't eat`
   - `[DURATION] for 2 days`
   - `[MEDICATION] insulin`
   - `[PRE_EXISTING] diabetes`

### The Approach: Transfer Learning + Multi-Task

Instead of training a model from scratch on our 10k samples (too small for a language model), we use **transfer learning**:

1. **Start with BioBERT** — a BERT model pretrained on millions of biomedical papers. It already understands medical terminology, symptoms, and clinical language.
2. **Add two task-specific heads** on top — thin linear layers that map BioBERT's rich representations to our specific outputs.
3. **Fine-tune everything together** — the backbone learns to adapt its representations for veterinary triage while the heads learn the classification tasks.

### Tokenization and BIO Alignment

Before training, each text sample goes through:

1. **Tokenization** — BioBERT's WordPiece tokenizer splits text into subword tokens: `"vomiting"` → `["vom", "##iting"]`
2. **BIO label alignment** — each entity span from the training data is mapped to the corresponding tokens using character offsets:
   - `"vomiting"` → `["vom"=B-SYMPTOM, "##iting"=I-SYMPTOM]`
   - `B-` = Beginning of entity, `I-` = Inside (continuation), `O` = Outside (not an entity)
3. **Padding** — all sequences are padded to 256 tokens. Special tokens (`[CLS]`, `[SEP]`, `[PAD]`) get label `-100` (ignored in loss).

### Multi-Task Training

Both tasks share the same BioBERT backbone and train simultaneously:

- **Urgency loss** — standard cross-entropy on the 5-class prediction from `[CLS]`
- **NER loss** — per-token cross-entropy on BIO tags, weighted at 0.3x to prevent NER from dominating gradients
- **Combined loss** = `urgency_loss + 0.3 * ner_loss`

This multi-task setup is beneficial because the tasks are related: learning to identify symptoms (NER) helps predict urgency, and understanding urgency context helps locate relevant entities.

### Training Details

- **Optimizer**: AdamW (LR=2e-5) — standard for transformer fine-tuning
- **Early stopping**: patience=4, monitors validation loss
- **Data split**: 80% train / 10% val / 10% test
- **Training time**: ~10 minutes on Tesla T4 GPU (15 epochs max, typically stops at epoch 7)

### Architecture (nlp_gpu.py)

```
___________________________________________________________________________________________
 Layer (type)                    Output Shape              Param #        Connected to
===========================================================================================
 input_ids (Input)               (batch, 256)              0
___________________________________________________________________________________________
 attention_mask (Input)          (batch, 256)              0
___________________________________________________________________________________________
 BioBERT (AutoModel)             (batch, 256, 768)         110,104,890    input_ids,
   12x Transformer layers                                                 attention_mask
   - Multi-Head Self-Attention
   - Feed-Forward
   - LayerNorm + Residual
___________________________________________________________________________________________
                          ┌──────────┴──────────┐
                          │                     │
 cls_output [:, 0, :]     │              token_outputs
 (batch, 768)             │              (batch, 256, 768)
                          │                     │
___________________________________________________________________________________________
 urgency_head (Linear)    (batch, 5)            3,845      cls_output
___________________________________________________________________________________________
 ner_head (Linear)        (batch, 256, 19)      14,611     token_outputs
===========================================================================================
 Total params: 110,123,346

 Losses:
   urgency: CrossEntropyLoss          (weight: 1.0)
   ner:     CrossEntropyLoss(-100)    (weight: 0.3)
___________________________________________________________________________________________
```

One backbone, two heads. BioBERT processes the text once, then:
- **[CLS] token** (position 0) → urgency classification (5 classes)
- **All 256 tokens** → NER per-token classification (19 BIO tags across 9 entity types: AGE, BREED, DURATION, EXPOSURE, MEDICATION, PRE_EXISTING, SEX_STATUS, SYMPTOM, TOXIN)

## Components

| File | Framework | Purpose |
|------|-----------|---------|
| `nlp_gpu.py` | PyTorch | Train BioBERT multi-task model (urgency + NER) |
| `image_processing_gpu.py` | Keras | Train EfficientNetB3 breed classifier |
| `app_gpu.py` | Streamlit | Interactive demo (both models) |
| `embedding.py` | Keras | Original BoW + Embedding urgency models |
| `text_models_bow_and_embedding.py` | Keras | BoW + BiLSTM urgency models |
| `MultimodalFusionLayer.py` | Keras | Image+text fusion (original approach) |
| `app.py` | Streamlit | Original demo (fusion-based) |

## Pipeline

```
# Text model (run on GPU VM)
python nlp_gpu.py                    → triage_multitask_model_10k.pt

# Image model (run on GPU VM)
python image_processing_gpu.py       → dog_breed_model_gpu_B3.keras

# Demo app (runs locally)
streamlit run app_gpu.py
```

## Image Model Training Evolution (79% → 94%)

| Step | Change | Test Accuracy | Gain |
|------|--------|--------------|------|
| Baseline | B0, 160x160, 20% train (small_train) | 79% | - |
| 1 | Full training set | 79% | ~0% |
| 2 | Image size 160→224 | 86.4% | +7.4% |
| 3 | Fine-tune top 20 layers (1e-4) | 86.8% | +0.4% |
| 4 | EfficientNetB0→B3, 224→300 | 93.5% | +6.7% |
| 5 | Fine-tune top 100 layers (1e-5, BN frozen) | **94.0%** | +0.5% |

Key takeaways:
- Image size and model size gave the biggest gains (~14% combined)
- Fine-tuning gave marginal improvements (~1% total) — needed careful handling (freeze BatchNorm, low LR)

## Text Model Training Evolution (80% → 84%)

| Step | Change | Urgency Acc | NER Acc | Notes |
|------|--------|-------------|---------|-------|
| Baseline | BioBERT, 2.5k data | 80.4% | 93.5% | B-SYMPTOM F1: 0.60 |
| 1 | 10k data | **84.1%** | 92.5% | B-SYMPTOM F1: 0.64 |
| 2 | Class-weighted loss | 80.3% | 89.5% | Made things worse, reverted |
| 3 | DeBERTa v3 base | 84.3% | **95.3%** | Better NER but DeBERTa tokenizer issues |
| Final | BioBERT + 10k (unweighted) | **84.1%** | **92.5%** | Best stability + compatibility |

Key takeaways:
- 4x more data (2.5k→10k) gave the biggest urgency gain
- Green/Semi-Urgent improved the most (F1: 0.479 → 0.710)
- NER symptom extraction improved dramatically (0.1 → 3.9 symptoms/sample)

## Dataset Generation

Training data is generated synthetically using a two-agent Gemini pipeline (`nlp_demo_batch.py`):
1. **Writer agent** (Gemini Flash) generates realistic pet owner messages with diverse tones, personas, and formatting
2. **Annotator agent** (Gemini Pro) extracts structured clinical data and NER entities

See `BATCH_HOWTO.md` in the vet directory for batch generation instructions.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# For text model inference
pip install torch transformers

# For GPU training
pip install tensorflow[and-cuda]   # image model
pip install torch --index-url https://download.pytorch.org/whl/cu124  # text model
```

## Requirements

- Python 3.11+
- TensorFlow 2.12+ (image model)
- PyTorch 2.x (text model)
- transformers (HuggingFace)
- streamlit (demo app)

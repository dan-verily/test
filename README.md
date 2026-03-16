# Pet Triage — Multimodal Veterinary Urgency System

A multimodal deep learning system for veterinary triage: upload a pet image and describe the situation to get **urgency priority (1–5)** and **dog breed** prediction.

## Components

1. **Image Classification** (`image_processing.ipynb` / `image_processing_gpu.py`) — CNN (EfficientNet) trained on 120 dog breeds
2. **Text Classification** (`embedding.py`, `text_models_bow_and_embedding.py`) — Urgency classification from text descriptions (Bag-of-Words, Embedding, BiLSTM)
3. **Multimodal Fusion** (`MultimodalFusionLayer.py`) — Combines image and text models for joint urgency prediction
4. **Streamlit App** (`app.py`) — Interactive demo UI

## Pipeline

```
1. python embedding.py                → urgency_embedding_model.keras
2. image_processing.ipynb / .py       → dog_breed_model.keras
3. python MultimodalFusionLayer.py    → multimodal_fusion_model.keras
4. streamlit run app.py               → interactive demo
```

## Image Model Training Evolution (79% → 94%)

| Step | Change | Test Accuracy | Gain |
|------|--------|--------------|------|
| Baseline | B0, 160×160, 20% train (small_train) | 79% | — |
| 1 | Full training set | 79% | ~0% |
| 2 | Image size 160→224 | 86.4% | +7.4% |
| 3 | Fine-tune top 20 layers (1e-4) | 86.8% | +0.4% |
| 4 | EfficientNetB0→B3, 224→300 | 93.5% | +6.7% |
| 5 | Fine-tune top 100 layers (1e-5, BN frozen) | **94.0%** | +0.5% |

**Key takeaways:**
- Image size and model size gave the biggest gains (~14% combined)
- Fine-tuning gave marginal improvements (~1% total) — needed careful handling (freeze BatchNorm, low LR)
- Using full train vs small_train didn't help much with B0, but likely helped B3 converge better at 93%+

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Requirements

- Python 3.11+
- TensorFlow 2.12+
- See `requirements.txt` for full list

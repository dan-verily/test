"""
Multi-task veterinary triage model:
  - Urgency classification (1-5)
  - NER extraction (BIO tags for 9 entity types)
  - Structured field classification (sex, reproductive_status, appetite,
    urination, defecation, energy_level)

Uses BioBERT with shared backbone and multiple heads.

Run: python nlp_gpu.py
Requires: pip install transformers torch scikit-learn
"""

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ---- Config ----
MODEL_NAME = "dmis-lab/biobert-base-cased-v1.2"
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 15
LR = 1e-5
URGENCY_LOSS_WEIGHT = 2.0
NER_LOSS_WEIGHT = 0.3
FIELD_LOSS_WEIGHT = 0.3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = "triage_dataset_2500_clean.jsonl"

# ---- NER label scheme (BIO) ----
ENTITY_TYPES = ["AGE", "BREED", "DURATION", "EXPOSURE", "MEDICATION",
                "PRE_EXISTING", "SEX_STATUS", "SYMPTOM", "TOXIN"]

NER_LABELS = ["O"]
for etype in ENTITY_TYPES:
    NER_LABELS.append(f"B-{etype}")
    NER_LABELS.append(f"I-{etype}")
NER_LABEL2ID = {l: i for i, l in enumerate(NER_LABELS)}
NUM_NER_LABELS = len(NER_LABELS)

NUM_URGENCY_CLASSES = 5

URGENCY_NAMES = {
    0: "Red / Immediate",
    1: "Orange / Emergent",
    2: "Yellow / Urgent",
    3: "Green / Semi-Urgent",
    4: "Blue / Non-Urgent",
}

# ---- Structured field definitions ----
# Each field: (key_in_json, classes_list)
# "null" is a valid class for fields that may be absent
STRUCTURED_FIELDS = {
    "sex":                  ["male", "female", "null"],
    "reproductive_status":  ["intact", "spayed", "neutered", "null"],
    "appetite":             ["normal", "reduced", "anorexic", "null"],
    "urination":            ["normal", "straining", "absent", "excessive", "null"],
    "defecation":           ["normal", "diarrhea", "bloody", "null"],
    "energy_level":         ["alert", "lethargic", "obtunded", "unresponsive", "null"],
}

# Build label-to-id mappings for each field
FIELD_LABEL2ID = {}
for field, classes in STRUCTURED_FIELDS.items():
    FIELD_LABEL2ID[field] = {c: i for i, c in enumerate(classes)}


# ---- Data loading ----
def load_dataset(path):
    """Load JSONL, return list of sample dicts."""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            sr = d.get("systems_review") or {}
            sample = {
                "text": d["text"],
                "urgency": d["urgency_level"] - 1,  # 1-5 -> 0-4
                "entities": d.get("entities", []),
            }
            # Structured fields
            for field, classes in STRUCTURED_FIELDS.items():
                if field in ("appetite", "urination", "defecation", "energy_level"):
                    raw = sr.get(field)
                else:
                    raw = d.get(field)
                value = raw if raw in classes else "null"
                sample[field] = FIELD_LABEL2ID[field][value]
            samples.append(sample)
    return samples


def align_entities_to_tokens(text, entities, tokenizer, max_len):
    """
    Tokenize text and create BIO labels aligned to subword tokens.
    Returns: encoding dict + ner_labels list (one per token, padded to max_len).
    """
    encoding = tokenizer(
        text,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_offsets_mapping=True,
        return_tensors="pt",
    )

    offsets = encoding["offset_mapping"][0].tolist()
    ner_labels = [-100] * max_len  # -100 = ignore (special tokens, padding)

    # Mark real tokens as "O" first
    for i, (start, end) in enumerate(offsets):
        if start == 0 and end == 0:
            continue
        ner_labels[i] = NER_LABEL2ID["O"]

    # Find each entity phrase in text and tag matching tokens
    for ent in entities:
        phrase = ent["phrase"]
        label = ent["label"]
        if label not in ENTITY_TYPES:
            continue

        text_lower = text.lower()
        phrase_lower = phrase.lower()
        char_start = text_lower.find(phrase_lower)
        if char_start == -1:
            continue
        char_end = char_start + len(phrase)

        first = True
        for i, (tok_start, tok_end) in enumerate(offsets):
            if tok_start == tok_end == 0:
                continue
            if tok_start >= char_start and tok_end <= char_end:
                tag = f"B-{label}" if first else f"I-{label}"
                ner_labels[i] = NER_LABEL2ID[tag]
                first = False

    encoding.pop("offset_mapping")
    return encoding, ner_labels


# ---- Dataset ----
class TriageDataset(Dataset):
    def __init__(self, samples, tokenizer, max_len):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        encoding, ner_labels = align_entities_to_tokens(
            s["text"], s["entities"], self.tokenizer, self.max_len
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "urgency_label": torch.tensor(s["urgency"], dtype=torch.long),
            "ner_labels": torch.tensor(ner_labels, dtype=torch.long),
        }
        for field in STRUCTURED_FIELDS:
            item[field] = torch.tensor(s[field], dtype=torch.long)
        return item


# ---- Model ----
class TriageMultiTaskModel(nn.Module):
    """BioBERT with multiple heads: urgency, NER, and structured fields."""

    def __init__(self, model_name, num_urgency_classes, num_ner_labels, structured_fields):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size

        # Urgency head: [CLS] -> 5 classes
        self.urgency_head = nn.Linear(hidden_size, num_urgency_classes)

        # NER head: per-token -> BIO tags
        self.ner_head = nn.Linear(hidden_size, num_ner_labels)

        # Structured field heads: [CLS] -> N classes each
        self.field_heads = nn.ModuleDict({
            field: nn.Linear(hidden_size, len(classes))
            for field, classes in structured_fields.items()
        })

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        token_outputs = outputs.last_hidden_state

        urgency_logits = self.urgency_head(cls_output)
        ner_logits = self.ner_head(token_outputs)
        field_logits = {
            field: head(cls_output) for field, head in self.field_heads.items()
        }

        return urgency_logits, ner_logits, field_logits


# ---- Training ----
def train_epoch(model, dataloader, optimizer, urgency_loss_fn, ner_loss_fn, field_loss_fn):
    model.train()
    total_loss = 0
    urgency_correct = 0
    urgency_total = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        urgency_labels = batch["urgency_label"].to(DEVICE)
        ner_labels = batch["ner_labels"].to(DEVICE)

        urgency_logits, ner_logits, field_logits = model(input_ids, attention_mask)

        loss_urg = urgency_loss_fn(urgency_logits, urgency_labels)
        loss_ner = ner_loss_fn(
            ner_logits.view(-1, NUM_NER_LABELS),
            ner_labels.view(-1),
        )

        # Structured field losses
        loss_fields = 0
        for field in STRUCTURED_FIELDS:
            field_labels = batch[field].to(DEVICE)
            loss_fields = loss_fields + field_loss_fn(field_logits[field], field_labels)

        loss = URGENCY_LOSS_WEIGHT * loss_urg + NER_LOSS_WEIGHT * loss_ner + FIELD_LOSS_WEIGHT * loss_fields

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = urgency_logits.argmax(dim=1)
        urgency_correct += (preds == urgency_labels).sum().item()
        urgency_total += urgency_labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = urgency_correct / urgency_total
    return avg_loss, accuracy


def evaluate(model, dataloader, urgency_loss_fn, ner_loss_fn, field_loss_fn):
    model.eval()
    total_loss = 0
    all_urgency_preds = []
    all_urgency_labels = []
    all_ner_preds = []
    all_ner_labels = []
    all_field_preds = {f: [] for f in STRUCTURED_FIELDS}
    all_field_labels = {f: [] for f in STRUCTURED_FIELDS}

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            urgency_labels = batch["urgency_label"].to(DEVICE)
            ner_labels = batch["ner_labels"].to(DEVICE)

            urgency_logits, ner_logits, field_logits = model(input_ids, attention_mask)

            loss_urg = urgency_loss_fn(urgency_logits, urgency_labels)
            loss_ner = ner_loss_fn(
                ner_logits.view(-1, NUM_NER_LABELS),
                ner_labels.view(-1),
            )
            loss_fields = 0
            for field in STRUCTURED_FIELDS:
                field_labels = batch[field].to(DEVICE)
                loss_fields = loss_fields + field_loss_fn(field_logits[field], field_labels)

            total_loss += (loss_urg + loss_ner + loss_fields).item()

            all_urgency_preds.extend(urgency_logits.argmax(dim=1).cpu().tolist())
            all_urgency_labels.extend(urgency_labels.cpu().tolist())

            # NER
            ner_pred = ner_logits.argmax(dim=2).cpu()
            ner_true = ner_labels.cpu()
            for i in range(ner_true.size(0)):
                for j in range(ner_true.size(1)):
                    if ner_true[i, j] != -100:
                        all_ner_preds.append(ner_pred[i, j].item())
                        all_ner_labels.append(ner_true[i, j].item())

            # Structured fields
            for field in STRUCTURED_FIELDS:
                fl = batch[field].tolist()
                fp = field_logits[field].argmax(dim=1).cpu().tolist()
                all_field_labels[field].extend(fl)
                all_field_preds[field].extend(fp)

    avg_loss = total_loss / len(dataloader)
    return (avg_loss, all_urgency_preds, all_urgency_labels,
            all_ner_preds, all_ner_labels, all_field_preds, all_field_labels)


# ---- Main ----
if __name__ == "__main__":
    print(f"Device: {DEVICE}")

    # Load data
    print("Loading data...")
    samples = load_dataset(DATA_PATH)
    print(f"Total samples: {len(samples)}")

    # Split: 80% train, 10% val, 10% test
    train_samples, temp_samples = train_test_split(samples, test_size=0.2, random_state=42)
    val_samples, test_samples = train_test_split(temp_samples, test_size=0.5, random_state=42)
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")

    # Tokenizer + datasets
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = TriageDataset(train_samples, tokenizer, MAX_LEN)
    val_ds = TriageDataset(val_samples, tokenizer, MAX_LEN)
    test_ds = TriageDataset(test_samples, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # Model
    print(f"Loading model: {MODEL_NAME}")
    model = TriageMultiTaskModel(
        MODEL_NAME, NUM_URGENCY_CLASSES, NUM_NER_LABELS, STRUCTURED_FIELDS
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    urgency_loss_fn = nn.CrossEntropyLoss()
    ner_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    field_loss_fn = nn.CrossEntropyLoss()

    # Train
    best_val_loss = float("inf")
    patience_counter = 0
    patience = 4

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, urgency_loss_fn, ner_loss_fn, field_loss_fn
        )
        val_result = evaluate(model, val_loader, urgency_loss_fn, ner_loss_fn, field_loss_fn)
        val_loss = val_result[0]
        val_urg_preds, val_urg_labels = val_result[1], val_result[2]
        val_acc = sum(p == l for p, l in zip(val_urg_preds, val_urg_labels)) / len(val_urg_labels)

        print(
            f"Epoch {epoch+1}/{EPOCHS} — "
            f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f} — "
            f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "triage_multitask_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best and evaluate on test
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)

    model.load_state_dict(torch.load("triage_multitask_model.pt", weights_only=True))
    result = evaluate(model, test_loader, urgency_loss_fn, ner_loss_fn, field_loss_fn)
    _, urg_preds, urg_labels, ner_preds, ner_labels, field_preds, field_labels = result

    # Urgency report
    urgency_names = [URGENCY_NAMES[i] for i in range(NUM_URGENCY_CLASSES)]
    print("\n--- Urgency Classification ---")
    print(classification_report(urg_labels, urg_preds, target_names=urgency_names, digits=3))

    # NER report
    ner_label_names_used = sorted(set(ner_labels + ner_preds))
    ner_target_names = [NER_LABELS[i] for i in ner_label_names_used]
    print("--- NER (token-level) ---")
    print(classification_report(
        ner_labels, ner_preds,
        labels=ner_label_names_used,
        target_names=ner_target_names,
        digits=3,
        zero_division=0,
    ))

    # Structured field reports
    for field, classes in STRUCTURED_FIELDS.items():
        print(f"--- {field} ---")
        print(classification_report(
            field_labels[field], field_preds[field],
            target_names=classes, digits=3, zero_division=0,
        ))

    # ---- Example predictions ----
    print("=" * 60)
    print("EXAMPLE PREDICTIONS")
    print("=" * 60)

    examples = [
        "My dog is eating and playing normally, just here for a checkup.",
        "Cat has been vomiting for two days and not eating.",
        "Dog was hit by a car, bleeding from the leg, having trouble standing.",
    ]

    model.eval()
    for text in examples:
        encoding = tokenizer(
            text, max_length=MAX_LEN, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(DEVICE)
        attention_mask = encoding["attention_mask"].to(DEVICE)

        with torch.no_grad():
            urg_logits, ner_logits, fld_logits = model(input_ids, attention_mask)

        # Urgency
        urg_pred = urg_logits.argmax(dim=1).item()
        urg_conf = torch.softmax(urg_logits, dim=1)[0, urg_pred].item()

        # NER: extract entities from predicted BIO tags
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        ner_pred_ids = ner_logits.argmax(dim=2)[0].cpu().tolist()

        entities_found = []
        current_entity = None
        current_tokens = []

        for tok, pred_id, mask in zip(tokens, ner_pred_ids, attention_mask[0].tolist()):
            if mask == 0 or tok in ["[CLS]", "[SEP]", "[PAD]"]:
                if current_entity:
                    entities_found.append((current_entity, current_tokens))
                    current_entity = None
                    current_tokens = []
                continue

            tag = NER_LABELS[pred_id]
            if tag.startswith("B-"):
                if current_entity:
                    entities_found.append((current_entity, current_tokens))
                current_entity = tag[2:]
                current_tokens = [tok]
            elif tag.startswith("I-") and current_entity == tag[2:]:
                current_tokens.append(tok)
            else:
                if current_entity:
                    entities_found.append((current_entity, current_tokens))
                    current_entity = None
                    current_tokens = []

        if current_entity:
            entities_found.append((current_entity, current_tokens))

        # Structured fields
        field_results = {}
        for field, classes in STRUCTURED_FIELDS.items():
            pred_id = fld_logits[field].argmax(dim=1).item()
            field_results[field] = classes[pred_id]

        # Print
        print(f"\nText: \"{text}\"")
        print(f"  Urgency: {urg_pred+1} — {URGENCY_NAMES[urg_pred]} ({urg_conf:.0%})")
        if entities_found:
            print(f"  Entities:")
            for etype, toks in entities_found:
                phrase = tokenizer.convert_tokens_to_string(toks)
                print(f"    [{etype}] {phrase}")
        else:
            print(f"  Entities: none detected")
        print(f"  Structured fields:")
        for field, value in field_results.items():
            print(f"    {field}: {value}")

    print("\nDone.")

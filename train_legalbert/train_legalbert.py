import os
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import f1_score, precision_score, recall_score
from collections import defaultdict


MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
DATA_DIR= os.path.normpath("/preprocess_data/abelation_data/data_full/")
OUTPUT_DIR = os.path.normpath("./models_full/")
RESULTS_DIR = os.path.normpath("./results_full/")
MAX_LENGTH= 512
BATCH_SIZE= 8
LEARNING_RATE = 2e-5
NUM_EPOCHS = 10
WARMUP_STEPS = 100
CATEGORY_LOSS_WEIGHT= 1.0
ATTRIBUTE_LOSS_WEIGHT  = 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CATEGORY_THRESHOLDS = {
    "Data Retention":0.55,
    "Data Security": 0.55,
    "First Party Collection/Use": 0.55,
    "International and Specific Audiences": 0.5,
    "Policy Change": 0.45,
    "Third Party Sharing/Collection":0.6,
    "User Access, Edit and Deletion": 0.6,
    "User Choice/Control": 0.55,
}

class PrivacyPolicyDataset(Dataset):

    def __init__(self, json_path, tokenizer, label_schema, max_length):
        with open(json_path, "r", encoding="utf-8") as f:
            self.examples = json.load(f)

        self.tokenizer = tokenizer
        self.label_schema = label_schema
        self.max_length= max_length

        self.categories = sorted(
            set(info["category"] for info in label_schema.values())
        )
        self.head_names = sorted(label_schema.keys())

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        encoding = self.tokenizer(
            example["text"],
            max_length=self.max_length,
            padding= "max_length",
            truncation=True,
            return_tensors="pt"
        )

        cat_labels = torch.tensor(
            [float(example["category_labels"].get(cat, 0))
             for cat in self.categories],
            dtype=torch.float
        )

        attr_labels = torch.tensor(
            [int(example["attribute_labels"].get(head, -1))
             for head in self.head_names],
            dtype=torch.long
        )

        return {
            "input_ids":encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "category_labels": cat_labels,
            "attribute_labels": attr_labels,
        }

class MultiHeadLegalBERT(nn.Module):

    def __init__(self, model_name, label_schema, categories):
        super().__init__()

        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout= nn.Dropout(0.1)
        self.categories = sorted(categories)
        self.head_names = sorted(label_schema.keys())
        hidden_size= self.bert.config.hidden_size


        self.category_heads = nn.ModuleDict({
            self._safe_key(cat): nn.Linear(hidden_size, 1)
            for cat in self.categories
        })

        self.attribute_heads = nn.ModuleDict({
            self._safe_key(head): nn.Linear(
                hidden_size,
                label_schema[head]["num_classes"]
            )
            for head in self.head_names
        })

    def _safe_key(self, name):
        return name.replace("/", "_").replace(" ", "_").replace(",", "_")

    def forward(self, input_ids, attention_mask):

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls_embedding = self.dropout(outputs.pooler_output)

        category_logits = {}
        for cat in self.categories:
            key = self._safe_key(cat)
            category_logits[cat] = self.category_heads[key](cls_embedding)

        attribute_logits = {}
        for head in self.head_names:
            key = self._safe_key(head)
            attribute_logits[head] = self.attribute_heads[key](cls_embedding)

        return category_logits, attribute_logits


def compute_category_weights(dataset, categories):

    total = len(dataset.examples)
    pos_counts = {cat: 0 for cat in categories}

    for example in dataset.examples:
        for cat in categories:
            if example["category_labels"].get(cat, 0) == 1:
                pos_counts[cat] += 1

    weights = {}

    for cat in categories:
        pos = pos_counts[cat]
        neg = total- pos
        raw_weight = neg / max(pos, 1)
        weight = min(max(raw_weight, 1.0), 10.0)
        weights[cat] = weight

    return weights


def compute_loss(category_logits, attribute_logits,
                 category_labels, attribute_labels,
                 categories, head_names,
                 category_weights=None,
                 cat_weight=1.0, attr_weight=1.0):

    ce_loss = nn.CrossEntropyLoss(ignore_index=-1)

    losses = []
    cat_loss_total= 0.0
    attr_loss_total = 0.0

    for i, cat in enumerate(categories):
        logits = category_logits[cat].squeeze(-1)
        labels = category_labels[:, i]

        if category_weights is not None:
            pos_weight = torch.tensor(
                category_weights[cat],
                dtype=torch.float,
                device=logits.device
            )
            loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(logits, labels)

        else:
            loss = nn.BCEWithLogitsLoss()(logits, labels)

        losses.append(cat_weight * loss)
        cat_loss_total += loss.item()

    for j, head in enumerate(head_names):
        logits = attribute_logits[head]
        labels = attribute_labels[:, j]

        valid_mask = labels != -1
        if valid_mask.sum() == 0:
            continue

        loss = ce_loss(logits, labels)

        if torch.isnan(loss):
            continue

        losses.append(attr_weight * loss)
        attr_loss_total += loss.item()

    if not losses:
        return torch.tensor(
            0.0, requires_grad=True, device=category_labels.device
        ), 0.0, 0.0

    total_loss = sum(losses)
    return total_loss, cat_loss_total, attr_loss_total


def evaluate(model, dataloader, categories, head_names, device,
             category_weights=None,
             category_thresholds=None):

    model.eval()

    total_loss = 0.0
    n_batches= 0

    all_cat_preds = defaultdict(list)
    all_cat_labels = defaultdict(list)
    all_attr_preds= defaultdict(list)
    all_attr_labels = defaultdict(list)

    with torch.no_grad():
        for batch in dataloader:
            input_ids= batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            category_labels = batch["category_labels"].to(device)
            attribute_labels = batch["attribute_labels"].to(device)

            category_logits, attribute_logits = model(input_ids, attention_mask)

            loss, _, _ = compute_loss(
                category_logits, attribute_logits,
                category_labels,attribute_labels,
                categories, head_names,
                category_weights=category_weights
            )
            total_loss += loss.item()
            n_batches += 1

            for i, cat in enumerate(categories):
                probs = torch.sigmoid(category_logits[cat].squeeze(-1))

                threshold = (
                    category_thresholds.get(cat, 0.5)
                    if category_thresholds is not None
                    else 0.5
                )

                preds = (probs> threshold).long().cpu().numpy()

                labels = category_labels[:, i].long().cpu().numpy()
                all_cat_preds[cat].extend(preds)
                all_cat_labels[cat].extend(labels)

            for j, head in enumerate(head_names):
                logits = attribute_logits[head]
                preds  = torch.argmax(logits, dim=-1).cpu().numpy()
                labels = attribute_labels[:, j].cpu().numpy()

                mask   = labels != -1
                if mask.sum() > 0:
                    all_attr_preds[head].extend(preds[mask])
                    all_attr_labels[head].extend(labels[mask])

    avg_loss = total_loss / max(n_batches, 1)

    cat_metrics = {}

    for cat in categories:

        if len(set(all_cat_labels[cat])) > 1:

            f1 = f1_score(
                all_cat_labels[cat], all_cat_preds[cat],
                average="binary", zero_division=0
            )

            precision = precision_score(
                all_cat_labels[cat],all_cat_preds[cat],
                average="binary", zero_division=0
            )

            recall = recall_score(
                all_cat_labels[cat], all_cat_preds[cat],
                average="binary", zero_division=0
            )

        else:
            f1 = precision = recall = 0.0

        cat_metrics[cat] = {
            "f1": f1,
            "precision": precision,
            "recall": recall
        }

    macro_cat_f1 = np.mean([m["f1"] for m in cat_metrics.values()])
    macro_cat_precision = np.mean([m["precision"] for m in cat_metrics.values()])
    macro_cat_recall = np.mean([m["recall"] for m in cat_metrics.values()])

    attr_metrics = {}
    for head in head_names:

        if len(all_attr_preds[head]) > 0:

            f1 = f1_score(
                all_attr_labels[head], all_attr_preds[head],
                average="macro", zero_division=0
            )

            precision = precision_score(
                all_attr_labels[head], all_attr_preds[head],
                average="macro", zero_division=0
            )

            recall = recall_score(
                all_attr_labels[head], all_attr_preds[head],
                average="macro", zero_division=0
            )

        else:
            f1 = precision = recall = 0.0

        attr_metrics[head] = {
            "f1": f1,
            "precision": precision,
            "recall": recall
        }

        macro_attr_f1 = np.mean([m["f1"] for m in attr_metrics.values()])
        macro_attr_precision = np.mean([m["precision"] for m in attr_metrics.values()])
        macro_attr_recall = np.mean([m["recall"] for m in attr_metrics.values()])

    return {
        "loss": avg_loss,
        "macro_cat_f1": macro_cat_f1,
        "macro_cat_precision": macro_cat_precision,
        "macro_cat_recall": macro_cat_recall,
        "macro_attr_f1": macro_attr_f1,
        "macro_attr_precision": macro_attr_precision,
        "macro_attr_recall": macro_attr_recall,
        "cat_f1": {cat: m["f1"] for cat, m in cat_metrics.items()},
        "cat_precision": {cat: m["precision"] for cat, m in cat_metrics.items()},
        "cat_recall": {cat: m["recall"] for cat, m in cat_metrics.items()},
        "attr_f1": {h: m["f1"] for h, m in attr_metrics.items()},
        "attr_precision": {h: m["precision"] for h, m in attr_metrics.items()},
        "attr_recall": {h: m["recall"] for h, m in attr_metrics.items()},
    }

def train(model, train_loader, val_loader, categories, head_names, device, category_weights=None):

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01
    )

    scheduler   = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=WARMUP_STEPS
    )

    best_val_loss = float("inf")
    all_metrics= []

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        epoch_cat_loss = 0.0
        epoch_attr_loss = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            category_labels = batch["category_labels"].to(device)
            attribute_labels= batch["attribute_labels"].to(device)

            optimizer.zero_grad()

            category_logits, attribute_logits = model(input_ids, attention_mask)

            loss, cat_loss, attr_loss = compute_loss(
                category_logits, attribute_logits,
                category_labels, attribute_labels,
                categories, head_names,
                category_weights=category_weights,
                cat_weight =CATEGORY_LOSS_WEIGHT,
                attr_weight= ATTRIBUTE_LOSS_WEIGHT
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)

            optimizer.step()
            scheduler.step()

            epoch_loss+= loss.item()
            epoch_cat_loss += cat_loss
            epoch_attr_loss += attr_loss
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)

        val_metrics = evaluate(
            model, val_loader, categories, head_names, device,
            category_weights=category_weights
        )

        print(f"Train loss: {avg_train_loss:.4f}")
        print(f"Val loss: {val_metrics['loss']:.4f}")
        print(f"Val cat F1: {val_metrics['macro_cat_f1']:.4f}")
        print(f"Val cat Precision: {val_metrics['macro_cat_precision']:.4f}")
        print(f"Val cat Recall: {val_metrics['macro_cat_recall']:.4f}")
        print(f"Val attr F1: {val_metrics['macro_attr_f1']:.4f}")
        print(f"Val attr Precision: {val_metrics['macro_attr_precision']:.4f}")
        print(f"Val attr Recall: {val_metrics['macro_attr_recall']:.4f}")

        print(f"\nPer-category F1:")
        for cat, f1 in sorted(val_metrics["cat_f1"].items()):
            print(f"{cat:<45} {f1:.4f}")

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_path = os.path.join(OUTPUT_DIR, "legalbert_best")

            model.bert.save_pretrained(best_path)

            torch.save(
                {
                    "category_heads": model.category_heads.state_dict(),
                    "attribute_heads": model.attribute_heads.state_dict(),
                },
                os.path.join(best_path, "heads.pt")
            )

        all_metrics.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": val_metrics["loss"],
            "val_cat_f1": val_metrics["macro_cat_f1"],
            "val_attr_f1": val_metrics["macro_attr_f1"],
            "val_cat_f1_per":val_metrics["cat_f1"],
            "val_attr_f1_per":val_metrics["attr_f1"],
        })

    final_path = os.path.join(OUTPUT_DIR, "legalbert_final")
    model.bert.save_pretrained(final_path)

    torch.save(
        {
            "category_heads": model.category_heads.state_dict(),
            "attribute_heads": model.attribute_heads.state_dict(),
        },
        os.path.join(final_path, "heads.pt")
    )

    metrics_path = os.path.join(RESULTS_DIR, "legalbert_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    return all_metrics

def evaluate_test(model, test_loader, categories, head_names, device, category_weights=None, category_thresholds=None):

    metrics = evaluate(
        model, test_loader, categories, head_names, device,
        category_weights=category_weights,
        category_thresholds=category_thresholds
    )

    print(f"\nTest loss:{metrics['loss']:.4f}")
    print(f"Macro category F1: {metrics['macro_cat_f1']:.4f}")
    print(f"Macro attribute F1:{metrics['macro_attr_f1']:.4f}")

    print("\nPer-category F1:")
    for cat, f1 in sorted(metrics["cat_f1"].items()):
        print(f"{cat:<45} {f1:.4f}")

    print("\nPer-attribute F1:")
    for head, f1 in sorted(metrics["attr_f1"].items()):
        print(f"{head:<60} {f1:.4f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    test_path = os.path.join(RESULTS_DIR, "legalbert_test_results.json")
    with open(test_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics

def main():

    schema_path = os.path.join(DATA_DIR, "label_schema.json")
    with open(schema_path) as f:
        label_schema = json.load(f)

    categories = sorted(
        set(info["category"] for info in label_schema.values())
    )
    head_names =sorted(label_schema.keys())

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = PrivacyPolicyDataset(
        os.path.join(DATA_DIR, "train.json"),
        tokenizer, label_schema, MAX_LENGTH
    )
    val_dataset = PrivacyPolicyDataset(
        os.path.join(DATA_DIR, "val.json"),
        tokenizer, label_schema, MAX_LENGTH
    )
    test_dataset = PrivacyPolicyDataset(
        os.path.join(DATA_DIR, "test.json"),
        tokenizer, label_schema, MAX_LENGTH
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )


    model = MultiHeadLegalBERT(MODEL_NAME, label_schema, categories)
    model = model.to(DEVICE)

    category_weights = compute_category_weights(train_dataset, categories)


    metrics = train(
        model, train_loader, val_loader,
        categories, head_names, DEVICE,
        category_weights=category_weights
    )

    best_path = os.path.join(OUTPUT_DIR, "legalbert_best")

    model.bert= AutoModel.from_pretrained(best_path)
    heads = torch.load(
        os.path.join(best_path, "heads.pt"),
        map_location=DEVICE
    )
    model.category_heads.load_state_dict(heads["category_heads"])
    model.attribute_heads.load_state_dict(heads["attribute_heads"])
    model = model.to(DEVICE)

    evaluate_test(model, test_loader, categories, head_names, DEVICE,
                  category_weights=category_weights,
                  category_thresholds=CATEGORY_THRESHOLDS)

if __name__ == "__main__":
    main()
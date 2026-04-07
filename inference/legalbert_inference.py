import os
import sys
import json
import argparse
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from scoring import compute_privacy_score


MODEL_NAME   = "nlpaueb/legal-bert-base-uncased"
BEST_PATH    = "../train_legalbert/models/legalbert_best"
DATA_DIR     = "../preprocess_data/data"
MAX_LENGTH   = 512
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CATEGORY_THRESHOLDS = {
    "Data Retention":                       0.55,
    "Data Security":                        0.55,
    "First Party Collection/Use":           0.55,
    "International and Specific Audiences": 0.45,
    "Policy Change":                        0.45,
    "Third Party Sharing/Collection":       0.60,
    "User Access, Edit and Deletion":       0.60,
    "User Choice/Control":                  0.60,
}

class MultiHeadLegalBERT(nn.Module):
    def __init__(self, model_name, label_schema, categories):


        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.categories = sorted(categories)
        self.head_names = sorted(label_schema.keys())
        hidden_size = self.bert.config.hidden_size

        self.category_heads = nn.ModuleDict({
            self._safe_key(cat): nn.Linear(hidden_size, 1)
            for cat in self.categories
        })

        self.attribute_heads = nn.ModuleDict({
            self._safe_key(head): nn.Linear(
                hidden_size, label_schema[head]["num_classes"]
            )

            for head in self.head_names
        })

    def _safe_key(self, name):
        return name.replace("/", "_").replace(" ", "_").replace(",", "_")

    def forward(self, input_ids, attention_mask):
        outputs      = self.bert(input_ids=input_ids,
                                 attention_mask=attention_mask)
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


def load_model(label_schema, categories):

    model = MultiHeadLegalBERT(MODEL_NAME, label_schema, categories)
    model.bert = AutoModel.from_pretrained(BEST_PATH)

    heads_path = os.path.join(BEST_PATH, "heads.pt")
    heads = torch.load(heads_path, map_location=DEVICE)
    model.category_heads.load_state_dict(heads["category_heads"])
    model.attribute_heads.load_state_dict(heads["attribute_heads"])

    model = model.to(DEVICE)
    model.eval()

    return model


def predict_segment(model, tokenizer, text, label_schema, categories, head_names):

    encoding = tokenizer(
        text,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    input_ids      = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)

    with torch.no_grad():
        category_logits, attribute_logits = model(input_ids, attention_mask)

    category_labels = {}

    for cat in categories:
        prob = torch.sigmoid(
            category_logits[cat].squeeze(-1)
        ).item()
        threshold = CATEGORY_THRESHOLDS.get(cat, 0.5)
        category_labels[cat] = 1 if prob > threshold else 0

    attribute_indices = {}
    attribute_values = {}

    for head in head_names:
        logits = attribute_logits[head]
        idx = torch.argmax(logits, dim=-1).item()
        cat = label_schema[head]["category"]
        values = label_schema[head]["values"]
        value_str = values[idx] if idx < len(values) else values[0]

        attribute_indices[head] = idx

        if category_labels[cat] == 1:

            attribute_values[head] = value_str

    return {
        "text": text,
        "category_labels": category_labels,
        "attribute_labels": attribute_indices,
        "attribute_values": attribute_values,
    }


def predict_policy(model, tokenizer, segments, label_schema, categories, head_names):

    results = []
    for i, segment in enumerate(segments):

        if not segment.strip():
            continue

        result = predict_segment( model, tokenizer, segment, label_schema, categories, head_names)

        results.append(result)

    return results



def main():
    parser = argparse.ArgumentParser(
        description="LegalBERT Privacy Policy Inference"
    )
    parser.add_argument(
        "--text",
        type=str,
    )
    parser.add_argument(
        "--file",
        type=str,
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--score",
        action="store_true",
    )
    args = parser.parse_args()

    if not args.text and not args.file:
        sys.exit(1)

    schema_path = os.path.join(DATA_DIR, "label_schema.json")
    with open(schema_path) as f:
        label_schema= json.load(f)

    categories = sorted(
        set(info["category"] for info in label_schema.values())
    )

    head_names =sorted(label_schema.keys())

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model= load_model(label_schema, categories)

    if args.text:
        result = predict_segment(
            model, tokenizer, args.text,
            label_schema, categories, head_names
        )
        predictions = [result]

        print("\nResults:")
        print(f"Categories detected:")
        for cat, label in result["category_labels"].items():
            if label == 1:
                print(f"{cat}")

        print(f"\nAttribute values:")
        for head, value in result["attribute_values"].items():
            print(f"{head}: {value}")

    elif args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            content = f.read()

        if "|||" in content:
            segments = content.split("|||")
            print(f"  Found {len(segments)} segments")
        else:
            segments = [s.strip() for s in content.split("\n\n") if s.strip()]

            print(f"Split into {len(segments)} paragraphs")

        predictions = predict_policy( model, tokenizer, segments, label_schema, categories, head_names)


    if args.score and len(predictions) > 0:

        score_result = compute_privacy_score(predictions)
        print(f"\nPrivacy-Friendliness Score:")
        print(f"Score: {score_result['score_0_10']}/10")
        print(f"Raw score: {score_result['raw_score']}")
        print(f"Segments: {score_result['n_segments']}")
        print(f"categories: "
              f"{', '.join(score_result['categories_seen'])}")

        for pred in predictions:
            pred["privacy_score"] = score_result

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
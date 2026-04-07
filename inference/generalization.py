import os
import json
import numpy as np
from collections import defaultdict
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import AutoTokenizer

from legalbert_inference import (
    load_model,
    predict_segment
)

MODEL_NAME    = "nlpaueb/legal-bert-base-uncased"
BEST_PATH     = "../train_legalbert/models/legalbert_best"
DATA_DIR      = "../preprocess_data/data"
POLICIES_DIR  = "./modern_policies"
OUTPUT_PATH   = "./results/generalization_results.json"

CATEGORIES = [
    "Data Retention",
    "Data Security",
    "First Party Collection/Use",
    "International and Specific Audiences",
    "Policy Change",
    "Third Party Sharing/Collection",
    "User Access, Edit and Deletion",
    "User Choice/Control",
]

def load_policy_and_annotations(txt_path, annotation_path):

    with open(txt_path, encoding="utf-8") as f:
        segments = [s.strip() for s in f.read().split("|||")
                    if s.strip()]

    ground_truth = defaultdict(
        lambda: {cat: 0 for cat in CATEGORIES}
    )

    with open(annotation_path, encoding="utf-8") as f:

        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split(",", 1)
            if len(parts) != 2:
                continue

            sid = int(parts[0].strip())
            cat = parts[1].strip().strip('"')

            if cat in CATEGORIES:
                ground_truth[sid][cat] = 1

    examples = []
    for i, text in enumerate(segments):
        examples.append({
            "text": text,
            "category_labels": ground_truth[i],
        })

    return examples


def discover_policy_pairs(policies_dir):

    pairs = []
    for filename in sorted(os.listdir(policies_dir)):
        if filename.endswith("_ann.txt"):
            continue
        if not filename.endswith(".txt"):
            continue
        base = filename[:-4]
        policy_path = os.path.join(policies_dir, filename)
        annotation_path = os.path.join(
            policies_dir, f"{base}_ann.txt"
        )
        if os.path.exists(annotation_path):
            pairs.append((base, policy_path, annotation_path))
        else:
            print(f"No ann file found {filename}")

    return pairs


def evaluate_policy(model, tokenizer, label_schema,
                    categories, head_names, examples, policy_name):

    all_cat_preds  = defaultdict(list)
    all_cat_labels = defaultdict(list)

    total = len(examples)

    for i, example in enumerate(examples):
        if not example["text"].strip():
            continue

        result = predict_segment(
            model, tokenizer, example["text"],
            label_schema, categories, head_names
        )

        for cat in categories:
            all_cat_preds[cat].append(
                result["category_labels"].get(cat, 0)
            )
            all_cat_labels[cat].append(
                example["category_labels"].get(cat, 0)
            )

    cat_metrics = {}
    for cat in categories:

        if len(set(all_cat_labels[cat])) > 1:

            f1 = f1_score(
                all_cat_labels[cat], all_cat_preds[cat],
                average="binary", zero_division=0
            )

            p = precision_score(
                all_cat_labels[cat], all_cat_preds[cat],
                average="binary", zero_division=0
            )

            r = recall_score(
                all_cat_labels[cat], all_cat_preds[cat],
                average="binary", zero_division=0
            )

        else:
            f1 =p = r = None
        cat_metrics[cat] = {"f1": f1, "precision": p, "recall": r}

    valid = [m for m in cat_metrics.values() if m["f1"] is not None]
    macro_f1 = np.mean([m["f1"] for m in valid]) if valid else 0.0
    macro_p = np.mean([m["precision"] for m in valid]) if valid else 0.0
    macro_r = np.mean([m["recall"] for m in valid]) if valid else 0.0

    print(f"\nResults for {policy_name}:"
          f"Macro F1: {macro_f1:.4f}"
          f"Precision: {macro_p:.4f}"
          f"Recall: {macro_r:.4f}"
          f"Per-category (P / R / F1):")
    for cat in sorted(categories):

        m = cat_metrics[cat]

        if m["f1"] is None:
            print(f"{cat:<45} — (not present in policy)")
        else:
            print(f"{cat:<45} "
                  f"{m['precision']:.3f} / "
                  f"{m['recall']:.3f} / "
                  f"{m['f1']:.3f}")

    return {
        "policy": policy_name,
        "n_segments": total,
        "macro_f1": round(macro_f1, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),

        "cat_f1": {c: round(m["f1"], 4)
        if m["f1"] is not None else None
        for c, m in cat_metrics.items()},

        "cat_precision": {c: round(m["precision"], 4)
        if m["precision"] is not None else None
        for c, m in cat_metrics.items()},

        "cat_recall": {c: round(m["recall"], 4)
        if m["recall"] is not None else None
        for c, m in cat_metrics.items()},
    }

def aggregate_results(policy_results):

    all_f1 = []
    all_precision = []
    all_recall= []

    cat_f1_all = defaultdict(list)

    for r in policy_results:

        if r["macro_f1"] is not None:
            all_f1.append(r["macro_f1"])
            all_precision.append(r["macro_precision"])
            all_recall.append(r["macro_recall"])

        for cat, f1 in r["cat_f1"].items():
            if f1 is not None:
                cat_f1_all[cat].append(f1)

    overall_f1 =np.mean(all_f1) if all_f1 else 0.0
    overall_precision = np.mean(all_precision) if all_precision else 0.0
    overall_recall = np.mean(all_recall) if all_recall else 0.0

    per_cat_f1 = {
        cat: round(np.mean(vals), 4)
        for cat, vals in cat_f1_all.items()
    }

    return {
        "n_policies": len(policy_results),
        "macro_f1": round(overall_f1, 4),
        "macro_precision": round(overall_precision, 4),
        "macro_recall": round(overall_recall, 4),
        "per_category_f1": per_cat_f1,
    }


def main():
    schema_path = os.path.join(DATA_DIR, "label_schema.json")
    with open(schema_path) as f:
        label_schema = json.load(f)

    categories= sorted(set(info["category"] for info in label_schema.values()))
    head_names = sorted(label_schema.keys())

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = load_model(label_schema, categories)

    pairs = discover_policy_pairs(POLICIES_DIR)
    if not pairs:
        print(f"No policy pairs found in {POLICIES_DIR}")
        return

    for name, _, _ in pairs:
        print(f"{name}")

    policy_results = []
    for name, policy_path, annotation_path in pairs:

        examples = load_policy_and_annotations(
            policy_path, annotation_path
        )
        result = evaluate_policy(
            model, tokenizer,label_schema,
            categories, head_names, examples, name
        )
        policy_results.append(result)

    aggregate = aggregate_results(policy_results)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    output = {"aggregate": aggregate, "per_policy": policy_results}

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()
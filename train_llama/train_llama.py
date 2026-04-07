import os
import re
import json
import torch
import numpy as np
from datasets import Dataset as HFDataset
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from sklearn.metrics import f1_score, precision_score, recall_score


MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
DATA_DIR = "../preprocess_data/data/"
CHECKPOINT_DIR= "./models/llama_checkpoints/"
FINAL_DIR= "./models/llama_final/"
RESULTS_DIR = "./results/"

MAX_LENGTH= 1024
BATCH_SIZE= 4
GRAD_ACCUM = 4
LEARNING_RATE= 5e-5
NUM_EPOCHS= 3
WARMUP_STEPS= 30
SAVE_STEPS = 50
SAVE_TOTAL_LIMIT = 20
LOGGING_STEPS = 10

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT= 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CATEGORY_ATTRIBUTES = {
    "First Party Collection/Use": {
        "Does/Does Not": ["Does", "Does Not"],
        "Personal Information Type": [
            "Unspecified", "Generic personal information",
            "Cookies and tracking elements", "Contact",
            "User online activities", "IP address and device IDs",
            "Computer information", "Location", "Demographic",
            "Financial", "User profile", "Other sensitive"
        ],

        "Purpose": [
            "Unspecified", "Basic service/feature", "Analytics/Research",
            "Additional service/feature", "Personalization/Customization",
            "Advertising", "Marketing", "Service Operation and Security",
            "Other"
        ],
        "Choice Type": [
            "Unspecified", "Opt-in", "Opt-out",
            "Privacy controls", "Dont use service/feature"
        ],
        "Choice Scope": ["Unspecified", "Collection", "Use", "Both"],
    },

    "Third Party Sharing/Collection": {
        "Does/Does Not": ["Does", "Does Not"],
        "Third Party Entity": [
            "Unnamed third party", "Named third party",
            "Affiliate", "Unspecified", "Public", "Other"
        ],
        "Personal Information Type": [
            "Unspecified", "Generic personal information",
            "Cookies and tracking elements", "User online activities",
            "Contact", "IP address and device IDs", "Financial",
            "Health", "Computer information", "User profile",
            "Other sensitive"
        ],
        "Purpose": [
            "Unspecified", "Advertising", "Basic service/feature",
            "Additional service/feature", "Service operation and security",
            "Marketing", "Legal requirement", "Analytics/Research",
            "Merger/Acquisition", "Other"
        ],
        "Choice Type": [
            "Unspecified", "Opt-in", "Opt-out",
            "Privacy controls", "Dont use service/feature"
        ],
        "Choice Scope": ["Unspecified", "Collection", "Use", "Both"],
    },

    "User Choice/Control": {
        "Choice Type": [
            "Opt-out", "Opt-in", "Privacy controls",
            "Dont use service/feature", "Unspecified"
        ],
        "Choice Scope": [
            "First party use", "First party collection",
            "Third party sharing/collection", "Third party use",
            "Unspecified"
        ],
        "Personal Information Type": [
            "Unspecified", "Contact", "Cookies and tracking elements",
            "Generic personal information", "Other"
        ],
        "Purpose": [
            "Unspecified", "Marketing", "Advertising",
            "Basic service/feature", "Additional service/feature", "Other"
        ],
    },
    "User Access, Edit and Deletion": {
        "Access Type": [
            "Edit information", "View", "Delete account",
            "None", "Other", "Unspecified"
        ],
        "Access Scope": ["User account data", "Other data", "Unspecified"],
    },

    "Data Retention": {
        "Retention Period": ["Unspecified", "Limited", "Indefinitely"],
    },
    "Data Security": {
        "Security Measure": ["Generic", "Specific", "Unspecified"],
    },
    "Policy Change": {
        "Change Type": ["Unspecified", "Privacy relevant change", "Other"],
        "Notification Type": [
            "Personal notice", "General notice",
            "No notification", "Unspecified"
        ],
        "User Choice": ["Unspecified", "Has choice", "None"],
    },
    "International and Specific Audiences": {
        "Audience Type": [
            "Children", "Californians", "International", "Other"
        ],
    },
}

CATEGORIES = sorted(CATEGORY_ATTRIBUTES.keys())

CATEGORY_DESCRIPTIONS = {
    "First Party Collection/Use":
        "How and why the service itself collects or uses user information.",
    "Third Party Sharing/Collection":
        "How user information may be shared with or collected by third parties.",
    "User Choice/Control":
        "Choices and control options available to users over their data.",
    "User Access, Edit and Deletion":
        "Whether and how users may access, edit, or delete their information.",
    "Data Retention":
        "How long user information is stored by the service.",
    "Data Security":
        "How user information is protected from unauthorized access.",

    "Policy Change":
        "If and how users will be informed about changes to the policy.",
    "International and Specific Audiences":
        "Practices that apply only to specific groups such as children.",
}

def build_system_prompt():

    lines = []

    lines.append(
        "You are a privacy policy analyst. Your task is to classify "
        "privacy policy text segments into privacy practice categories. "
        "For each segment, identify which categories apply and "
        "classify the relevant attributes according to the defined "
        "schema below.\n"
    )
    lines.append("CATEGORIES AND ATTRIBUTES:")


    for cat in CATEGORIES:
        desc = CATEGORY_DESCRIPTIONS[cat]
        lines.append(f"\n{cat}: {desc}")


        for attr_name, values in CATEGORY_ATTRIBUTES[cat].items():
            vals = ", ".join(f'"{v}"' for v in values)
            lines.append(f"  - {attr_name}: [{vals}]")
    lines.append("\n")

    lines.append(
        "\nRespond ONLY with valid JSON. No explanation, no markdown. "
        "Use exactly the category names and attribute values defined "
        "above — do not use any values outside the listed options. "
        "Format:\n"
        "{\n"
        '  "category_labels": {"<category>": 1 or 0, ...},\n'
        '  "attribute_labels": {"<category>__<attribute>": "<value>", ...}\n'
        "}\n"
        "Include all 8 categories in category_labels. "
        "Include attributes only for categories with label 1."
        "\nOnly include categories that apply to the segment. "
        "If no categories apply, output empty category_labels. "
        "Do not list categories with label 0."
    )
    return "\n".join(lines)


def build_full_prompt(segment_text, response_text=None):
    system = build_system_prompt()
    user   = f"Classify this privacy policy segment:\n\n\"\"\"\n{segment_text}\n\"\"\""

    prompt = (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n"
        f"{system}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    if response_text is not None:
        prompt += f"{response_text}<|eot_id|>"
    return prompt


def build_response(example, label_schema):

    relevant_categories = {
        cat: 1
        for cat, label in example["category_labels"].items()
        if label == 1
    }

    if not relevant_categories:
        return json.dumps({"category_labels": {}, "attribute_labels": {}})

    attribute_labels = {}
    for cat in relevant_categories:


        for attr_name, values in CATEGORY_ATTRIBUTES[cat].items():
            head_name = f"{cat}__{attr_name}"
            idx =example["attribute_labels"].get(head_name, -1)
            value = values[idx] if 0 <= idx < len(values) else values[0]
            attribute_labels[head_name] = value

    return json.dumps({
        "category_labels": relevant_categories,
        "attribute_labels": attribute_labels
    }, indent=2)

class LlamaPrivacyDataset:
    def __init__(self, json_path, label_schema):
        with open(json_path, "r", encoding="utf-8") as f:
            self.examples = json.load(f)
        self.label_schema = label_schema
        print(f"  Loaded {len(self.examples)} examples from {json_path}")

    def to_hf_dataset(self):
        texts = []

        for example in self.examples:
            has_positive = any(
                v == 1 for v in example["category_labels"].values()
            )
            if not has_positive:
                continue

            response = build_response(example, self.label_schema)
            text = build_full_prompt(example["text"], response)
            texts.append({"text": text})

        return HFDataset.from_list(texts)


def load_model_and_tokenizer():
    print(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=True
    )

    tokenizer.pad_token= tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    return model, tokenizer

def find_latest_checkpoint(checkpoint_dir):

    if not os.path.exists(checkpoint_dir):
        return None

    checkpoints = [
        d for d in os.listdir(checkpoint_dir)
        if d.startswith("checkpoint-") and
        os.path.isdir(os.path.join(checkpoint_dir, d))
    ]

    if not checkpoints:
        return None

    checkpoints.sort(key=lambda x: int(x.split("-")[1]))
    latest = os.path.join(checkpoint_dir, checkpoints[-1])
    print(f"Found checkpoint to resume from: {latest}")

    return latest



def train(model, tokenizer, train_hf, val_hf):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    latest_checkpoint = find_latest_checkpoint(CHECKPOINT_DIR)
    if latest_checkpoint:
        print(f"\nResuming from: {latest_checkpoint}")
    else:
        print("\nStarting fresh training run...")

    sft_config = SFTConfig(
        output_dir =CHECKPOINT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        weight_decay= 0.001,
        lr_scheduler_type="cosine",
        fp16=False,
        bf16=True,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        eval_strategy="steps",
        eval_steps=SAVE_STEPS,
        logging_steps=LOGGING_STEPS,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        max_length=MAX_LENGTH,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_hf,
        eval_dataset=val_hf,
        args=sft_config,
        processing_class=tokenizer,
    )

    trainer.train(resume_from_checkpoint=latest_checkpoint)
    return trainer



def parse_llama_output(output_text):

    try:
        match = re.search(r'\{.*\}', output_text, re.DOTALL)

        if match:
            return json.loads(match.group())

    except json.JSONDecodeError:
        pass

    return {
        "category_labels":  {cat: 0 for cat in CATEGORIES},
        "attribute_labels": {}
    }


def predict_segment(model, tokenizer, segment_text, device):

    prompt = build_full_prompt(segment_text)

    inputs = tokenizer(
        prompt, return_tensors="pt",
        truncation=True, max_length=MAX_LENGTH
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]
    new_tokens = outputs[0][prompt_len:]
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return parse_llama_output(response_text), response_text


def convert_to_indices(parsed_output):
    cat_labels = {}
    attr_labels = {}
    raw_cats= parsed_output.get("category_labels", {})
    raw_attrs = parsed_output.get("attribute_labels", {})

    for cat in CATEGORIES:
        cat_labels[cat] = int(raw_cats.get(cat, 0))

    for cat in CATEGORIES:

        if cat_labels[cat] == 0:
            continue

        for attr_name, values in CATEGORY_ATTRIBUTES[cat].items():
            head_name = f"{cat}__{attr_name}"
            raw_value = raw_attrs.get(head_name, values[0])

            try:
                idx = values.index(raw_value)
            except ValueError:
                idx = 0

            attr_labels[head_name] = idx

    return cat_labels, attr_labels


def evaluate_model(model, tokenizer, dataset, device, split_name="test", max_samples=None):

    examples = dataset.examples

    if max_samples is not None:
        examples = examples[:max_samples]

    model.eval()

    all_cat_preds= defaultdict(list)
    all_cat_labels = defaultdict(list)

    all_attr_preds = defaultdict(list)
    all_attr_labels = defaultdict(list)
    parse_failures = 0
    total= len(examples)

    for i, example in enumerate(examples):

        parsed, _ = predict_segment(
            model, tokenizer, example["text"], device
        )

        if not parsed.get("category_labels"):
            parse_failures += 1

        pred_cats, pred_attrs = convert_to_indices(parsed)

        for cat in CATEGORIES:
            all_cat_preds[cat].append(pred_cats.get(cat, 0))
            all_cat_labels[cat].append(
                int(example["category_labels"].get(cat, 0))
            )

        for cat in CATEGORIES:

            if example["category_labels"].get(cat, 0) == 0:
                continue

            for attr_name in CATEGORY_ATTRIBUTES[cat].keys():
                head_name = f"{cat}__{attr_name}"
                true_idx  = example["attribute_labels"].get(head_name, -1)

                if true_idx == -1:
                    continue

                all_attr_preds[head_name].append(pred_attrs.get(head_name, 0))

                all_attr_labels[head_name].append(true_idx)


    cat_metrics = {}
    for cat in CATEGORIES:

        if len(set(all_cat_labels[cat])) > 1:

            f1 = f1_score(all_cat_labels[cat], all_cat_preds[cat], average="binary", zero_division=0)
            p  = precision_score(all_cat_labels[cat], all_cat_preds[cat], average="binary", zero_division=0)
            r  = recall_score(all_cat_labels[cat], all_cat_preds[cat], average="binary", zero_division=0)

        else:
            f1 = p = r = 0.0

        cat_metrics[cat] = {"f1": f1, "precision": p, "recall": r}

    macro_cat_f1  = np.mean([m["f1"] for m in cat_metrics.values()])
    macro_cat_p = np.mean([m["precision"] for m in cat_metrics.values()])
    macro_cat_r= np.mean([m["recall"] for m in cat_metrics.values()])

    attr_metrics = {}
    for head in sorted(all_attr_preds.keys()):

        if all_attr_preds[head]:
            f1 = f1_score(all_attr_labels[head], all_attr_preds[head], average="macro", zero_division=0)
            p  = precision_score(all_attr_labels[head], all_attr_preds[head], average="macro", zero_division=0)
            r  = recall_score(all_attr_labels[head], all_attr_preds[head], average="macro", zero_division=0)

        else:
            f1 = p = r = 0.0
        attr_metrics[head] = {"f1": f1, "precision": p, "recall": r}

    macro_attr_f1 = np.mean([m["f1"] for m in attr_metrics.values()]) if attr_metrics else 0.0
    macro_attr_p  = np.mean([m["precision"] for m in attr_metrics.values()]) if attr_metrics else 0.0
    macro_attr_r  = np.mean([m["recall"] for m in attr_metrics.values()]) if attr_metrics else 0.0

    print(f"Macro cat  F1:        {macro_cat_f1:.4f}")
    print(f"Macro cat  Precision: {macro_cat_p:.4f}")
    print(f"Macro cat  Recall:    {macro_cat_r:.4f}")
    print(f"Macro attr F1:        {macro_attr_f1:.4f}")
    print(f"\nPer-category (Precision /Recall / F1):")

    for cat in CATEGORIES:
        m = cat_metrics[cat]
        print(f"{cat:<45}"
              f"{m['precision']:.3f} / {m['recall']:.3f} / {m['f1']:.3f}")

    results = {
        "split": split_name,
        "parse_failures":parse_failures,
        "total_examples":total,
        "macro_cat_f1":macro_cat_f1,
        "macro_cat_precision":macro_cat_p,
        "macro_cat_recall":macro_cat_r,
        "macro_attr_f1": macro_attr_f1,
        "macro_attr_precision": macro_attr_p,
        "macro_attr_recall": macro_attr_r,
        "cat_f1": {c: m["f1"] for c, m in cat_metrics.items()},
        "cat_precision": {c: m["precision"] for c, m in cat_metrics.items()},
        "cat_recall": {c: m["recall"] for c, m in cat_metrics.items()},
        "attr_f1":{h: m["f1"] for h, m in attr_metrics.items()},
        "attr_precision": {h: m["precision"] for h, m in attr_metrics.items()},
        "attr_recall": {h: m["recall"] for h, m in attr_metrics.items()},
    }
    return results

def main():

    with open(os.path.join(DATA_DIR, "label_schema.json")) as f:
        label_schema = json.load(f)

    train_dataset = LlamaPrivacyDataset(
        os.path.join(DATA_DIR, "train.json"), label_schema
    )
    val_dataset = LlamaPrivacyDataset(
        os.path.join(DATA_DIR, "val.json"), label_schema
    )
    test_dataset = LlamaPrivacyDataset(
        os.path.join(DATA_DIR, "test.json"), label_schema
    )

    train_hf = train_dataset.to_hf_dataset()
    val_hf = val_dataset.to_hf_dataset()

    model, tokenizer = load_model_and_tokenizer()

    trainer = train(model, tokenizer, train_hf, val_hf)

    os.makedirs(FINAL_DIR, exist_ok=True)

    trainer.save_model(FINAL_DIR)
    tokenizer.save_pretrained(FINAL_DIR)

    test_results = evaluate_model(trainer.model, tokenizer, test_dataset, DEVICE, "test")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "llama_test_results.json")
    with open(out_path, "w") as f:
        json.dump(test_results, f, indent=2)


if __name__ == "__main__":
    main()
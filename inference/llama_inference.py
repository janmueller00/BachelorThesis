import os
import re
import sys
import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from scoring import compute_privacy_score

BASE_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
LORA_PATH = "../train_llama/models/checkpoint-100"
MAX_LENGTH = 1024
MAX_NEW_TOKENS = 512
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
    lines.append("=" * 40)
    for cat in CATEGORIES:
        desc = CATEGORY_DESCRIPTIONS[cat]
        lines.append(f"\n{cat}: {desc}")
        for attr_name, values in CATEGORY_ATTRIBUTES[cat].items():
            vals = ", ".join(f'"{v}"' for v in values)
            lines.append(f"  - {attr_name}: [{vals}]")
    lines.append("\n" + "=" * 40)
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
    )
    return "\n".join(lines)


def build_inference_prompt(segment_text):

    system = build_system_prompt()
    user   = (f"Classify this privacy policy segment:\n\n"
               f"\"\"\"\n{segment_text}\n\"\"\"")
    return (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n"
        f"{system}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def load_model_and_tokenizer():

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_NAME, trust_remote_code=True
    )
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.eval()

    return model, tokenizer

def parse_llama_output(output_text):

    try:
        match = re.search(r'\{.*\}', output_text, re.DOTALL)
        if match:
            return json.loads(match.group()), output_text
    except json.JSONDecodeError:
        pass

    fallback = {
        "category_labels": {cat: 0 for cat in CATEGORIES},
        "attribute_labels": {}
    }
    return fallback, output_text


def convert_to_indices(parsed_output):

    cat_labels= {}
    attr_indices = {}
    attr_values = {}

    raw_cats  = parsed_output.get("category_labels", {})
    raw_attrs = parsed_output.get("attribute_labels", {})

    for cat in CATEGORIES:
        cat_labels[cat] =int(raw_cats.get(cat, 0))

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
            attr_indices[head_name] = idx
            attr_values[head_name] = values[idx]

    return cat_labels, attr_indices, attr_values


def predict_segment(model, tokenizer, text):

    prompt = build_inference_prompt(text)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,        # greedy for consistency
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]
    new_tokens = outputs[0][prompt_len:]
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    parsed,raw = parse_llama_output(response_text)
    cat_labels, attr_indices, attr_values = convert_to_indices(parsed)

    return {
        "text": text,
        "category_labels": cat_labels,
        "attribute_labels": attr_indices,
        "attribute_values": attr_values,
        "raw_response": raw,
    }


def predict_policy(model, tokenizer, segments):

    results = []

    for i, segment in enumerate(segments):
        if not segment.strip():
            continue

        result =predict_segment(model, tokenizer, segment)
        results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="LLaMA Privacy Policy Inference"
    )

    parser.add_argument("--text", type=str)
    parser.add_argument("--file", type=str)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--score", action="store_true")

    args = parser.parse_args()

    if not args.text and not args.file:
        print("Provide --text or --file")
        sys.exit(1)

    model, tokenizer =load_model_and_tokenizer()

    if args.text:

        print(f"\nSingle segment")

        result= predict_segment(model, tokenizer, args.text)
        predictions = [result]

        print(f"\nRaw LLaMA response:\n{result['raw_response']}")

    elif args.file:

        print(f"\nPolicy: {args.file}...")

        with open(args.file, "r", encoding="utf-8") as f:
            content = f.read()

        if "|||" in content:
            segments = content.split("|||")
        else:
            segments= [s.strip() for s in content.split("\n\n") if s.strip()]

        predictions = predict_policy(model, tokenizer, segments)


    if args.score and predictions:
        score_result = compute_privacy_score(predictions)
        print(f"\nPrivacy-Friendliness Score:")
        print(f"Score: {score_result['score_0_10']}/10")
        print(f"Raw score: {score_result['raw_score']}")

        for pred in predictions:
            pred["privacy_score"] = score_result

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
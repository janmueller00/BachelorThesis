import os
import re
import json
import random
import pandas as pd
from collections import defaultdict
from html.parser import HTMLParser

CONSOLIDATION_DIR   = "./threshold-0.75-overlap-similarity/"
SANITIZED_DIR       = "./sanitized_policies/"
OUTPUT_DIR          = "./data/"
RANDOM_SEED         = 42
TRAIN_RATIO         = 0.8
VAL_RATIO           = 0.1
TEST_RATIO          = 0.1

EXCLUDE_CATEGORIES = {"Do Not Track", "Other"}

ATTRIBUTES_TO_DROP = {
    "User Type",
    "Collection Mode",
    "Action First-Party",
    "Action Third Party",
    "Identifiability",
    "Retention Purpose",
}

CATEGORY_SPECIFIC_DROPS = {
    "Data Retention": {
        "Personal Information Type",
        "Retention Purpose"
    }
}

ATTRIBUTE_MERGING_RULES = {

    "First Party Collection/Use >> Does/Does Not": {
        "Does":     "Does",
        "Does Not": "Does Not",
    },

    "First Party Collection/Use >> Personal Information Type": {
        "Unspecified":                  "Unspecified",
        "Generic personal information": "Generic personal information",
        "Cookies and tracking elements":"Cookies and tracking elements",
        "Contact":                      "Contact",
        "User online activities":       "User online activities",
        "IP address and device IDs":    "IP address and device IDs",
        "Computer information":         "Computer information",
        "Location":                     "Location",
        "Demographic":                  "Demographic",
        "Financial":                    "Financial",
        "User profile":                 "User profile",
        "Health":                       "Other sensitive",
        "Personal identifier":          "Other sensitive",
        "Social media data":            "Other sensitive",
        "Survey data":                  "Other sensitive",
        "Other":                        "Other sensitive",
    },

    "First Party Collection/Use >> Purpose": {
        "Unspecified":                   "Unspecified",
        "Basic service/feature":         "Basic service/feature",
        "Analytics/Research":            "Analytics/Research",
        "Additional service/feature":    "Additional service/feature",
        "Personalization/Customization": "Personalization/Customization",
        "Advertising":                   "Advertising",
        "Marketing":                     "Marketing",
        "Service Operation and Security":"Service Operation and Security",
        "Other":                         "Other",
        "Legal requirement":             "Other",
        "Merger/Acquisition":            "Other",
    },

    "First Party Collection/Use >> Choice Type": {
        "Unspecified":                   "Unspecified",
        "not-selected":                  "Unspecified",
        "Other":                         "Unspecified",
        "Opt-in":                        "Opt-in",
        "Dont use service/feature":      "Dont use service/feature",
        "Browser/device privacy controls":"Privacy controls",
        "Third-party privacy controls":  "Privacy controls",
        "First-party privacy controls":  "Privacy controls",
        "Opt-out link":                  "Opt-out",
        "Opt-out via contacting company":"Opt-out",
    },

    "First Party Collection/Use >> Choice Scope": {
        "Unspecified":   "Unspecified",
        "not-selected":  "Unspecified",
        "Collection":    "Collection",
        "Use":           "Use",
        "Both":          "Both",
    },

    "Third Party Sharing/Collection >> Does/Does Not": {
        "Does":     "Does",
        "Does Not": "Does Not",
    },

    "Third Party Sharing/Collection >> Third Party Entity": {
        "Unnamed third party":            "Unnamed third party",
        "Named third party":              "Named third party",
        "Other part of company/affiliate":"Affiliate",
        "Unspecified":                    "Unspecified",
        "Public":                         "Public",
        "Other users":                    "Other",
        "Other":                          "Other",
    },

    "Third Party Sharing/Collection >> Personal Information Type": {
        "Unspecified":                  "Unspecified",
        "Generic personal information": "Generic personal information",
        "Cookies and tracking elements":"Cookies and tracking elements",
        "User online activities":       "User online activities",
        "Contact":                      "Contact",
        "IP address and device IDs":    "IP address and device IDs",
        "Financial":                    "Financial",
        "Health":                       "Health",
        "Computer information":         "Computer information",
        "User Profile":                 "User profile",
        "Demographic":                  "Other sensitive",
        "Location":                     "Other sensitive",
        "Personal identifier":          "Other sensitive",
        "Survey data":                  "Other sensitive",
        "Other":                        "Other sensitive",
    },

    "Third Party Sharing/Collection >> Purpose": {
        "Unspecified":                   "Unspecified",
        "Advertising":                   "Advertising",
        "Basic service/feature":         "Basic service/feature",
        "Additional service/feature":    "Additional service/feature",
        "Service operation and security":"Service operation and security",
        "Marketing":                     "Marketing",
        "Legal requirement":             "Legal requirement",
        "Analytics/Research":            "Analytics/Research",
        "Merger/Acquisition":            "Merger/Acquisition",
        "Other":                         "Other",
        "Personalization/Customization": "Other",
    },

    "Third Party Sharing/Collection >> Choice Type": {
        "Unspecified":                   "Unspecified",
        "not-selected":                  "Unspecified",
        "Other":                         "Unspecified",
        "Opt-in":                        "Opt-in",
        "Dont use service/feature":      "Dont use service/feature",
        "Opt-out link":                  "Opt-out",
        "Opt-out via contacting company":"Opt-out",
        "Third-party privacy controls":  "Privacy controls",
        "Browser/device privacy controls":"Privacy controls",
        "First-party privacy controls":  "Privacy controls",
    },

    "Third Party Sharing/Collection >> Choice Scope": {
        "Unspecified":  "Unspecified",
        "not-selected": "Unspecified",
        "Collection":   "Collection",
        "Use":          "Use",
        "Both":         "Both",
    },
    "User Choice/Control >> Choice Type": {
        "Opt-out link":                  "Opt-out",
        "Opt-out via contacting company":"Opt-out",
        "Opt-in":                        "Opt-in",
        "Browser/device privacy controls":"Privacy controls",
        "First-party privacy controls":  "Privacy controls",
        "Third-party privacy controls":  "Privacy controls",
        "Dont use service/feature":      "Dont use service/feature",
        "Unspecified":                   "Unspecified",
        "Other":                         "Unspecified",
    },

    "User Choice/Control >> Choice Scope": {
        "First party use":               "First party use",
        "First party collection":        "First party collection",
        "Third party sharing/collection":"Third party sharing/collection",
        "Third party use":               "Third party use",
        "Unspecified":                   "Unspecified",
    },

    "User Choice/Control >> Personal Information Type": {
        "Unspecified":                  "Unspecified",
        "Contact":                      "Contact",
        "Cookies and tracking elements":"Cookies and tracking elements",
        "Generic personal information": "Generic personal information",
        "User online activities":       "Other",
        "Health":                       "Other",
        "Location":                     "Other",
        "User profile":                 "Other",
        "Demographic":                  "Other",
        "Personal identifier":          "Other",
        "Financial":                    "Other",
        "Social media data":            "Other",
        "Computer information":         "Other",
        "Survey data":                  "Other",
        "IP address and device IDs":    "Other",
        "Other":                        "Other",
    },

    "User Choice/Control >> Purpose": {
        "Unspecified":                   "Unspecified",
        "Marketing":                     "Marketing",
        "Advertising":                   "Advertising",
        "Basic service/feature":         "Basic service/feature",
        "Additional service/feature":    "Additional service/feature",
        "Other":                         "Other",
        "Personalization/Customization": "Other",
        "Service Operation and Security":"Other",
        "Legal requirement":             "Other",
        "Analytics/Research":            "Other",
        "Merger/Acquisition":            "Other",
    },
    "User Access, Edit and Deletion >> Access Type": {
        "Edit information":        "Edit information",
        "View":                    "View",
        "Delete account (partial)":"Delete account",
        "Delete account (full)":   "Delete account",
        "Deactivate account":      "Other",
        "Export":                  "Other",
        "Other":                   "Other",
        "Unspecified":             "Unspecified",
        "None":                    "None",
    },

    "User Access, Edit and Deletion >> Access Scope": {
        "User account data":    "User account data",
        "Unspecified":          "Unspecified",
        "Profile data":         "Other data",
        "Other data about user":"Other data",
        "Transactional data":   "Other data",
        "Other":                "Other data",
    },
    "Data Retention >> Retention Period": {
        "Unspecified":   "Unspecified",
        "Limited":       "Limited",
        "Stated Period": "Limited",
        "Indefinitely":  "Indefinitely",
        "Other":         "Unspecified",
    },

    "Data Security >> Security Measure": {
        "Generic":                  "Generic",
        "Secure data transfer":     "Specific",
        "Secure data storage":      "Specific",
        "Secure user authentication":"Specific",
        "Data access limitation":   "Specific",
        "Privacy/Security program": "Specific",
        "Privacy review/audit":     "Specific",
        "Privacy training":         "Specific",
        "Other":                    "Specific",
        "Unspecified":              "Unspecified",
    },

    "Policy Change >> Change Type": {
        "Unspecified":                   "Unspecified",
        "Privacy relevant change":       "Privacy relevant change",
        "Non-privacy relevant change":   "Other",
        "In case of merger or acquisition":"Other",
        "Other":                         "Other",
    },
    "Policy Change >> Notification Type": {
        "Personal notice":                      "Personal notice",
        "General notice in privacy policy":     "General notice",
        "General notice on website":            "General notice",
        "No notification":                      "No notification",
        "Unspecified":                          "Unspecified",
        "Other":                               "Unspecified",
    },

    "Policy Change >> User Choice": {
        "Unspecified":       "Unspecified",
        "Opt-in":            "Has choice",
        "Opt-out":           "Has choice",
        "User participation":"Has choice",
        "None":              "None",
        "Other":             "Unspecified",
    },

    "International and Specific Audiences >> Audience Type": {
        "Children":                    "Children",
        "Californians":                "Californians",
        "Europeans":                   "International",
        "Citizens from other countries":"International",
        "Other":                       "Other",
    },
}

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
        "Choice Scope": [
            "Unspecified", "Collection", "Use", "Both"
        ],
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
        "Choice Scope": [
            "Unspecified", "Collection", "Use", "Both"
        ],
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
        "Access Scope": [
            "User account data", "Other data", "Unspecified"
        ],
    },
    "Data Retention": {
        "Retention Period": [
            "Unspecified", "Limited", "Indefinitely"
        ],
    },
    "Data Security": {
        "Security Measure": [
            "Generic", "Specific", "Unspecified"
        ],
    },
    "Policy Change": {
        "Change Type": [
            "Unspecified", "Privacy relevant change", "Other"
        ],
        "Notification Type": [
            "Personal notice", "General notice",
            "No notification", "Unspecified"
        ],
        "User Choice": [
            "Unspecified", "Has choice", "None"
        ],
    },
    "International and Specific Audiences": {
        "Audience Type": [
            "Children", "Californians", "International", "Other"
        ],
    },
}

class HTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text_parts = []

    def handle_data(self, data):
        self.text_parts.append(data)

    def get_text(self):
        return " ".join(self.text_parts)


def clean_html(raw_html):
    extractor = HTMLTextExtractor()
    try:
        extractor.feed(raw_html)
        text = extractor.get_text()
    except Exception:
        text = re.sub(r"<[^>]+>", " ", raw_html)

    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_policy_segments(sanitized_dir, consolidation_dir):

    prefix_to_policy_id = {}

    for filename in os.listdir(consolidation_dir):
        if not filename.endswith(".csv"):
            continue

        match = re.match(r"^(\d+)_", filename)
        if not match:
            continue

        prefix = match.group(1)
        filepath = os.path.join(consolidation_dir, filename)

        try:
            df = pd.read_csv(
                filepath,
                header=None,
                names=[
                    "annotation_id", "batch_id", "annotator_id",
                    "policy_id", "segment_id", "category",
                    "attribute_json", "date", "url"
                ],
                nrows=1
            )
            policy_id = int(df["policy_id"].iloc[0])
            prefix_to_policy_id[prefix] = policy_id
        except Exception as e:
            print(f"Error: {filename}: {e}")

    policy_segments = {}

    for filename in os.listdir(sanitized_dir):
        match = re.match(r"^(\d+)_", filename)

        if not match:
            continue

        prefix = match.group(1)


        if prefix not in prefix_to_policy_id:
            continue

        policy_id = prefix_to_policy_id[prefix]
        filepath = os.path.join(sanitized_dir, filename)

        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

            raw_segments = content.split("|||")
            segments = [clean_html(seg) for seg in raw_segments]
            policy_segments[policy_id] = segments

        except Exception as e:
            print(f"Error:{filename}: {e}")

    return policy_segments


def load_annotations(consolidation_dir):

    all_rows = []

    for filename in sorted(os.listdir(consolidation_dir)):

        if not filename.endswith(".csv"):
            continue

        filepath = os.path.join(consolidation_dir, filename)
        try:
            df = pd.read_csv(
                filepath,
                header=None,
                names=[
                    "annotation_id", "batch_id", "annotator_id",
                    "policy_id", "segment_id", "category",
                    "attribute_json", "date", "url"
                ]
            )
            all_rows.append(df)
        except Exception as e:
            print(f"Error:{filename}: {e}")

    df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    df = df[~df["category"].isin(EXCLUDE_CATEGORIES)]

    return df


def should_drop_attribute(category, attr_name):

    if attr_name in ATTRIBUTES_TO_DROP:
        return True

    if category in CATEGORY_SPECIFIC_DROPS:
        if attr_name in CATEGORY_SPECIFIC_DROPS[category]:
            return True
    return False


def apply_merging_rule(category, attr_name, value):

    if should_drop_attribute(category, attr_name):
        return None

    rule_key = f"{category} >> {attr_name}"
    if rule_key in ATTRIBUTE_MERGING_RULES:
        merged = ATTRIBUTE_MERGING_RULES[rule_key].get(value)

        if merged is None:
            return "Unspecified"
        return merged

    return value


def value_to_index(category, attr_name, value):

    if category not in CATEGORY_ATTRIBUTES:
        return None
    if attr_name not in CATEGORY_ATTRIBUTES[category]:
        return None
    value_list = CATEGORY_ATTRIBUTES[category][attr_name]
    if value not in value_list:
        return None
    return value_list.index(value)


def majority_vote_index(votes):

    if not votes:
        return None

    max_count = max(votes.values())

    tied = [idx for idx, count in votes.items() if count == max_count]

    return min(tied)


def build_examples_for_policy(policy_id, segments, annotations_df):

    examples = []

    policy_annotations = annotations_df[
        annotations_df["policy_id"] == policy_id
    ]


    segment_annotations = defaultdict(list)
    for _, row in policy_annotations.iterrows():
        segment_annotations[int(row["segment_id"])].append(row)


    for seg_id, segment_text in enumerate(segments):

        if not segment_text.strip():
            continue

        category_labels = {
            cat: 0 for cat in CATEGORY_ATTRIBUTES.keys()
        }

        attribute_votes = defaultdict(lambda: defaultdict(int))

        seg_rows = segment_annotations.get(seg_id, [])

        for row in seg_rows:
            category = row["category"]

            if category not in CATEGORY_ATTRIBUTES:
                continue

            category_labels[category] = 1

            try:
                attr_dict = json.loads(row["attribute_json"])
            except (json.JSONDecodeError, TypeError):
                continue

            for attr_name, attr_data in attr_dict.items():
                if not isinstance(attr_data, dict):
                    continue
                if "value" not in attr_data:
                    continue

                raw_value = attr_data["value"]

                merged_value = apply_merging_rule(
                    category, attr_name, raw_value
                )

                if merged_value is None:
                    continue

                idx = value_to_index(category, attr_name, merged_value)
                if idx is None:
                    continue

                head_name = f"{category}__{attr_name}"
                attribute_votes[head_name][idx] += 1

        attribute_labels = {}
        for cat, attrs in CATEGORY_ATTRIBUTES.items():
            for attr_name in attrs.keys():
                head_name = f"{cat}__{attr_name}"
                votes = attribute_votes.get(head_name, {})
                if votes:
                    attribute_labels[head_name] = majority_vote_index(votes)
                else:
                    attribute_labels[head_name] = -1

        examples.append({
            "policy_id": policy_id,
            "segment_id": seg_id,
            "text": segment_text,
            "category_labels": category_labels,
            "attribute_labels": attribute_labels,
        })

    return examples


def split_by_policy(all_examples, train_ratio, val_ratio, seed):

    random.seed(seed)

    policy_ids = list(set(ex["policy_id"] for ex in all_examples))
    random.shuffle(policy_ids)

    n = len(policy_ids)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    train_policies = set(policy_ids[:n_train])
    val_policies   = set(policy_ids[n_train:n_train + n_val])
    test_policies  = set(policy_ids[n_train + n_val:])

    train = [ex for ex in all_examples if ex["policy_id"] in train_policies]
    val   = [ex for ex in all_examples if ex["policy_id"] in val_policies]
    test  = [ex for ex in all_examples if ex["policy_id"] in test_policies]

    return train, val, test


def compute_stats(examples, split_name):

    n_examples = len(examples)
    n_policies = len(set(ex["policy_id"] for ex in examples))

    cat_counts = defaultdict(int)
    for ex in examples:
        for cat, label in ex["category_labels"].items():
            if label == 1:
                cat_counts[cat] += 1

    attr_counts = defaultdict(lambda: defaultdict(int))
    for ex in examples:
        for head_name, idx in ex["attribute_labels"].items():
            if idx >= 0:
                attr_counts[head_name][idx] += 1

    for cat, count in sorted(cat_counts.items()):
        pct = count / n_examples * 100
        print(f"    {cat:<45} {count:>5} ({pct:.1f}%)")

    return {
        "split": split_name,
        "n_examples": n_examples,
        "n_policies": n_policies,
        "category_counts": dict(cat_counts),
    }


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    annotations_df = load_annotations(CONSOLIDATION_DIR)
    print(f"Annotations: {len(annotations_df)}")

    policy_segments = load_policy_segments(SANITIZED_DIR, CONSOLIDATION_DIR)
    print(f"Policies loaded: {len(policy_segments)}")

    total_segments = sum(len(segs) for segs in policy_segments.values())
    print(f"Segments: {total_segments}")

    all_examples = []
    policies_processed = 0
    policies_skipped = 0

    for policy_id, segments in sorted(policy_segments.items()):

        policy_annotations = annotations_df[
            annotations_df["policy_id"] == policy_id
        ]

        if len(policy_annotations) == 0:
            policies_skipped += 1
            continue

        examples = build_examples_for_policy(
            policy_id, segments, annotations_df
        )
        all_examples.extend(examples)
        policies_processed += 1


    train, val, test = split_by_policy(
        all_examples, TRAIN_RATIO, VAL_RATIO, RANDOM_SEED
    )

    splits = {"train": train, "val": val, "test": test}
    for split_name, examples in splits.items():
        output_path = os.path.join(OUTPUT_DIR, f"{split_name}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(examples, f, indent=2, ensure_ascii=False)

    label_schema = {}
    for cat, attrs in CATEGORY_ATTRIBUTES.items():
        for attr_name, values in attrs.items():
            head_name = f"{cat}__{attr_name}"
            label_schema[head_name] = {
                "values": values,
                "num_classes": len(values),
                "category": cat,
                "attribute": attr_name,
            }

    schema_path = os.path.join(OUTPUT_DIR, "label_schema.json")
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(label_schema, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
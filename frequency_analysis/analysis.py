import os
import json
import pandas as pd
from collections import defaultdict
from contextlib import redirect_stdout

CONSOLIDATION_DIR = "../preprocess_data/threshold-0.75-overlap-similarity/"
OUTPUT_FILE = "./frequency_analysis.txt"
MIN_FREQUENCY_THRESHOLD = 100


EXCLUDE_CATEGORIES = {"Do Not Track", "Other"}

def load_all_annotations(consolidation_dir):
    all_rows = []
    files_loaded = 0

    for filename in os.listdir(consolidation_dir):

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
            files_loaded += 1

        except Exception as e:
            print(f"Could not load {filename}: {e}")
    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()


def parse_attribute_json(json_str):

    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return {}


def run_frequency_analysis(consolidation_dir, threshold):

    print(f"OPP-115 Frequency Analysis (threshold: 0.75)\n")

    df = load_all_annotations(consolidation_dir)

    if df.empty:
        print("No data loaded.")
        return

    total_annotations = len(df)
    df_filtered = df[~df["category"].isin(EXCLUDE_CATEGORIES)]

    print("\n\nCATEGORY FREQUENCIES:")

    category_counts = df_filtered["category"].value_counts()
    for category, count in category_counts.items():


        flag = "*** RARE ***" if count < threshold else ""
        print(f"{category:<45} {count:>6}{flag}")


    print("\n\nATTRIBUTE VALUE FREQUENCIES:")

    attr_value_counts = defaultdict(lambda: defaultdict(int))
    parse_errors = 0

    for _, row in df_filtered.iterrows():
        category = row["category"]
        attr_dict = parse_attribute_json(row["attribute_json"])

        for attr_name, attr_data in attr_dict.items():
            if isinstance(attr_data, dict) and "value" in attr_data:
                value = attr_data["value"]
                key = f"{category} >> {attr_name}"
                attr_value_counts[key][value] += 1
            else:
                parse_errors += 1

    if parse_errors > 0:
        print(f"{parse_errors} errors")

    flagged_for_reduction = []

    for attr_key in sorted(attr_value_counts.keys()):
        print(f"\n {attr_key}")

        value_counts = attr_value_counts[attr_key]


        for value, count in sorted(value_counts.items(), key=lambda x: x[1], reverse=True):
            flag = ""
            if count <threshold:
                flag ="*** BELOW THRESHOLD ***"
                flagged_for_reduction.append((attr_key, value, count))
            print(f"    {value:<45} {count:>6} {flag}")

    print(f"\n\nSUMMARY: VALUES BELOW THRESHOLD")

    if flagged_for_reduction:

        print(f"\n {len(flagged_for_reduction)} values flagged:")
        current_attr = None

        for attr_key,value, count in sorted(flagged_for_reduction):
            if attr_key != current_attr:
                print(f"\n  {attr_key}")
                current_attr = attr_key
            print(f"    - {value}: {count}")
    else:
        print(f"\n No values below threshold of {threshold}.")

    print("DATASET SIZE OVERVIEW:")

    unique_policies = df["policy_id"].nunique()

    print(f"\nTotal policies: {unique_policies}")
    print(f"Total annotations (all): {total_annotations}")
    print(f"Annotations after filtering: {len(df_filtered)}")

    return attr_value_counts, flagged_for_reduction

if __name__ == "__main__":

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            run_frequency_analysis(CONSOLIDATION_DIR, MIN_FREQUENCY_THRESHOLD)

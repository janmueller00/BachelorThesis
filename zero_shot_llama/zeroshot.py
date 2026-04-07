import os
import re
import sys
import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig



MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
MAX_LENGTH= 4096
MAX_NEW_TOKENS_SEGMENT = 150
MAX_NEW_TOKENS_HOLISTIC = 400
MAX_SEGMENTS_IN_STAGE2= 40
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def load_base_model():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,

    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer



SEGMENT_SYSTEM_PROMPT = """You are a privacy policy analyst. Rate the \
privacy-friendliness of privacy policy text segments on a scale from 1 to 10.

Scoring guidance:
- 1-2: Very privacy-unfriendly. Extensive data collection, sharing with \
unnamed third parties, no user control, indefinite retention, no transparency.
- 3-4: Privacy-unfriendly. Significant data practices with limited user \
control or transparency.
- 5-6: Neutral. Standard data practices with some user control or partial \
transparency.
- 7-8: Privacy-friendly. Limited collection, named third parties, clear \
user choices, or explicit retention limits.
- 9-10: Very privacy-friendly. Minimal collection, opt-in consent, full \
user control, explicit retention limits, strong security.

Special cases:
- If the segment only DESCRIBES what the policy will cover, introduces \
the company, or contains contact information without describing any \
actual data practice — output score 0. Score 0 means excluded.
- A segment must describe a SPECIFIC data practice (what is collected, \
how it is used, who it is shared with, user rights, retention, security) \
to receive a score of 1-10.
- Phrases like 'we protect your privacy' or 'we implement fair practices' \
WITHOUT specifics are boilerplate and should receive score 0. 
- If the segment explicitly states data is NOT collected or NOT shared, \
score higher (7-9).

Respond ONLY with a valid JSON object, nothing else:
{"score": <integer 1-10>, "reasoning": "<one concise sentence explaining the score>"}"""


def build_segment_prompt(segment_text):
    user = f"Rate this privacy policy segment:\n\n\"\"\"\n{segment_text}\n\"\"\""
    return (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n"
        f"{SEGMENT_SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def parse_segment_response(response_text):

    try:
        match = re.search(r'\{.*?\}', response_text, re.DOTALL)

        if match:
            parsed = json.loads(match.group())
            score = int(parsed.get("score", 0))
            reasoning = str(parsed.get("reasoning", ""))
            score = max(0, min(10, score))
            return score, reasoning

    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    numbers = re.findall(r'\b([0-9]|10)\b', response_text)

    if numbers:
        return int(numbers[0]), "score extracted from non-JSON response"

    return 0, "parse failure - excluding from score"


def score_segment(model, tokenizer, segment_text):

    prompt = build_segment_prompt(segment_text)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS_SEGMENT,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]

    new_tokens= outputs[0][prompt_len:]
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    score,reasoning = parse_segment_response(response_text)
    return score, reasoning, response_text

def score_policy_mean(model, tokenizer, segments, verbose=True):

    results = []
    total = len(segments)
    scores= []

    for i, segment in enumerate(segments):

        if not segment.strip():
            continue

        if verbose and ((i + 1) % 10 == 0 or i == 0):
            print(f"Segment {i+1}/{total}...")

        score, reasoning, raw = score_segment(model, tokenizer, segment)
        scores.append(score)
        results.append({
            "segment_id": i,
            "text": segment,
            "score": score,
            "reasoning": reasoning,
            "raw": raw,
        })

    relevant_scores = [r["score"] for r in results if r["score"] > 0]


    if not relevant_scores:
        policy_score = 0.0

    else:
        policy_score = round(sum(relevant_scores) / len(relevant_scores), 2)


    return {
        "method": "mean",
        "policy_score": policy_score,
        "n_segments": len(results),
        "score_min": min(scores) if scores else 5,
        "score_max": max(scores) if scores else 5,
        "segment_scores": results,
    }

HOLISTIC_SYSTEM_PROMPT = """You are a senior privacy policy analyst. \
You have reviewed individual segments of a privacy policy and assigned \
scores to each. Now synthesize these segment-level assessments into a \
comprehensive evaluation of the entire policy.

Consider the following when forming your holistic assessment:
- Overall balance between privacy-friendly and unfriendly practices
- Whether positive practices genuinely offset negative ones
- Consistency and transparency across the policy
- User control and consent mechanisms
- Data sharing and retention practices
- Whether the policy meets reasonable privacy expectations

Respond ONLY with a valid JSON object, nothing else:
{
  "overall_score": <integer 1-10>,
  "reasoning": "<2-3 sentences summarizing the overall privacy posture>",
  "strengths": "<main privacy-friendly aspects of this policy>",
  "weaknesses": "<main privacy concerns or unfriendly practices>"
}"""


def build_holistic_prompt(segment_results):

    results = segment_results

    if len(results) > MAX_SEGMENTS_IN_STAGE2:
        sorted_by_score = sorted(results, key=lambda x: x["score"])

        bottom = sorted_by_score[:10]
        top = sorted_by_score[-10:]
        middle = sorted_by_score[10:-10]

        n_middle = MAX_SEGMENTS_IN_STAGE2 - 20
        step = max(1, len(middle) // n_middle)
        middle = middle[::step][:n_middle]
        results = bottom + middle + top
        results = sorted(results, key=lambda x: x["segment_id"])

    summary_lines = []
    for r in results:
        summary_lines.append(f"Segment {r['segment_id'] + 1} [score {r['score']}/10]: {r['reasoning']}")
    segment_summary = "\n".join(summary_lines)

    user = (
        f"I have analyzed {len(segment_results)} segments of a privacy policy. "
        f"Here are the segment-level assessments"
        f"{'(representative sample)' if len(segment_results) > MAX_SEGMENTS_IN_STAGE2 else ''}:"
        f"\n\n{segment_summary}\n\n"
        f"Based on these assessments, provide an overall privacy-friendliness "
        f"score and comprehensive evaluation of this policy."
    )

    return (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n"
        f"{HOLISTIC_SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def parse_holistic_response(response_text):

    try:
        match = re.search(r'\{.*\}', response_text, re.DOTALL)

        if match:
            parsed = json.loads(match.group())
            score= int(parsed.get("overall_score", 5))
            score = max(1, min(10, score))

            return {
                "overall_score": score,
                "reasoning": str(parsed.get("reasoning", "")),
                "strengths": str(parsed.get("strengths", "")),
                "weaknesses": str(parsed.get("weaknesses", "")),
            }

    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    return {
        "overall_score": 5,
        "reasoning": "parse failure",
        "strengths": "",
        "weaknesses": "",
    }


def score_policy_hierarchical(model, tokenizer, segments, verbose=True):

    mean_result = score_policy_mean(
        model, tokenizer, segments, verbose=verbose
    )

    segment_results = mean_result["segment_scores"]

    if not segment_results:
        return {
            "method": "hierarchical",
            "overall_score": 5,
            "reasoning": "no segments to evaluate",
            "strengths": "",
            "weaknesses": "",
            "mean_score": 5.0,
            "segment_scores": [],
        }

    prompt = build_holistic_prompt(segment_results)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS_HOLISTIC,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]
    new_tokens = outputs[0][prompt_len:]
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    holistic = parse_holistic_response(response_text)

    if verbose:
        print(f"\nOverall score:{holistic['overall_score']}/10")
        print(f"Mean score:{mean_result['policy_score']}/10")
        print(f"Reasoning: {holistic['reasoning']}")
        print(f"Strengths: {holistic['strengths']}")
        print(f"Weaknesses:{holistic['weaknesses']}")

    return {
        "method": "hierarchical",
        "overall_score": holistic["overall_score"],
        "reasoning": holistic["reasoning"],
        "strengths": holistic["strengths"],
        "weaknesses":holistic["weaknesses"],
        "raw_stage2": response_text,
        "mean_score": mean_result["policy_score"],
        "n_segments": len(segment_results),
        "segment_scores": segment_results,
    }


def load_segments(file_path):

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    if "|||" in content:
        segments = [s.strip() for s in content.split("|||")]
    else:
        segments = [s.strip() for s in content.split("\n\n") if s.strip()]

    segments = [s for s in segments if len(s.strip()) > 20]

    return segments


def main():
    parser = argparse.ArgumentParser(
        description="LLaMA Zero-Shot Privacy Policy Scoring"
    )
    parser.add_argument( "--text", type=str)
    parser.add_argument("--file", type=str)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--method", type=str, default="both", choices=["mean", "hierarchical", "both"])
    args = parser.parse_args()

    if not args.text and not args.file:
        sys.exit(1)

    model, tokenizer = load_base_model()

    results = {}

    if args.text:

        score, reasoning, raw = score_segment(model, tokenizer, args.text)

        print(f"\nScore: {score}/10")
        print(f"Reasoning: {reasoning}")
        results = {"segment_score": score, "reasoning": reasoning, "raw": raw}

    elif args.file:
        segments = load_segments(args.file)

        if args.method in ("mean", "both"):
            mean_result = score_policy_mean(model, tokenizer, segments)
            results["mean_approach"] = mean_result
            print(f"\nMean score: {mean_result['policy_score']}/10")

        if args.method in ("hierarchical", "both"):

            if args.method == "hierarchical":

                hier_result = score_policy_hierarchical(
                    model, tokenizer, segments
                )
            else:
                segment_results = results["mean_approach"]["segment_scores"]

                prompt = build_holistic_prompt(segment_results)
                inputs = tokenizer(
                    prompt, return_tensors="pt",
                    truncation=True, max_length=MAX_LENGTH
                ).to(DEVICE)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS_HOLISTIC,
                        do_sample=False,
                        temperature=1.0,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                prompt_len = inputs["input_ids"].shape[1]
                response_text = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)

                holistic = parse_holistic_response(response_text)

                hier_result = {
                    "method": "hierarchical",
                    "overall_score": holistic["overall_score"],
                    "reasoning": holistic["reasoning"],
                    "strengths": holistic["strengths"],
                    "weaknesses":holistic["weaknesses"],
                    "raw_stage2": response_text,
                    "mean_score": results["mean_approach"]["policy_score"],
                    "n_segments":len(segment_results),
                    "segment_scores": segment_results,
                }


            print(f"\nFinal overall score: {hier_result['overall_score']}/10")
            print(f"Mean score: {hier_result['mean_score']}/10")
            print(f"Reasoning: {hier_result['reasoning']}")
            print(f"Strengths: {hier_result['strengths']}")
            print(f"Weaknesses:{hier_result['weaknesses']}")

            results["hierarchical_approach"] = hier_result

        if args.method == "both":

            print(f"Mean score:"
                  f"{results['mean_approach']['policy_score']}/10")

            print(f"Hierarchical score:"
                  f"{results['hierarchical_approach']['overall_score']}/10")


    if args.output and results:

        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
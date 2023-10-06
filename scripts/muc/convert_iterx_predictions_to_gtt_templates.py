import argparse
import json

from iterx.metrics.muc.gtt_eval_utils import (
    jsonlines_to_gtt_templates,
    read_gold_templates,
    convert_docid,
)


def convert_to_gtt_templates(gold_file: str, pred_file: str, output_file: str) -> None:
    gold = {}
    with open(gold_file, "r") as f:
        for line in f:
            parsed_line = json.loads(line.strip())
            gold[parsed_line["docid"]] = parsed_line
    gold_templates = read_gold_templates(
        gold_file, convert_doc_id=False, sanitize_special_chars=True
    )
    pred_templates = jsonlines_to_gtt_templates(
        pred_file, dedup=False, cluster_substr=False, normalize_role=True
    )
    output = {}
    templates_missing_from_predictions = 0
    for gold_key, gold_template in gold_templates.items():
        if gold_key not in pred_templates:
            templates_missing_from_predictions += 1
        pred_template = pred_templates.get(gold_key, [])
        converted_key = convert_docid(gold_key)
        output[converted_key] = {
            "doctext": gold[gold_key]["doctext"],
            "pred_templates": pred_template,
            "gold_templates": gold_template,
        }
    print(
        f"Total templates missing from predictions: {templates_missing_from_predictions}"
    )
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pred_file", type=str, help="path to input file")
    parser.add_argument("gold_file", type=str, help="path to gold file")
    parser.add_argument("output_file", type=str, help="path to output file")
    args = parser.parse_args()
    convert_to_gtt_templates(args.gold_file, args.pred_file, args.output_file)

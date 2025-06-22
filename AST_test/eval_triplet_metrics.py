# eval_triplet_metrics.py
import json
import ast
import re
import sys
import os

sentiment_dict = {"POS": "positive", "NEG": "negative", "NEU": "neutral"}

def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def generate_triples(txt_path):
    label = []
    sentences = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            if '####' not in line:
                continue
            sentence, raw_triplets = line.strip().split('####')
            sentences.append(sentence)
            tokens = sentence.strip().split()
            triplets = ast.literal_eval(raw_triplets.strip())

            output_triplets = []
            for aspect_idx, opinion_idx, sentiment in triplets:
                aspect = " ".join([tokens[i] for i in aspect_idx])
                opinion = " ".join([tokens[i] for i in opinion_idx])
                sentiment_str = sentiment_dict[sentiment]
                output_triplets.append((aspect.lower(), opinion.lower(), sentiment_str))
            label.append(output_triplets)
    return label, sentences

def extract_triplets(data):
    predict_all = []
    for i, item in enumerate(data):
        raw_predict_str = item['predict']
        predict_list = []

        try:
            triplet_strs = raw_predict_str.split('|')
            for triplet in triplet_strs:
                triplet = triplet.strip()
                aspect_match = re.search(r"aspect:\s*(.*?),", triplet, re.IGNORECASE)
                opinion_match = re.search(r"opinion:\s*(.*?),", triplet, re.IGNORECASE)
                sentiment_match = re.search(r"sentiment:\s*(\w+)", triplet, re.IGNORECASE)

                if aspect_match and opinion_match and sentiment_match:
                    aspect_term = aspect_match.group(1).strip().lower()
                    opinion_term = opinion_match.group(1).strip().lower()
                    sentiment = sentiment_match.group(1).strip().lower()
                    predict_list.append((aspect_term, opinion_term, sentiment))
                else:
                    print(f"字段缺失或格式异常 at index {i}: {triplet}")

        except Exception as e:
            print(f"解析失败 at index {i}: {raw_predict_str}")
            predict_list = []

        predict_all.append(predict_list)

    return predict_all

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_path', type=str, required=True)
    parser.add_argument('--predict_path', type=str, required=True)
    parser.add_argument('--output_error_path', type=str, required=True)
    parser.add_argument('--checkpoint', type=int, required=True)
    parser.add_argument('--results_file', type=str, required=True)

    args = parser.parse_args()

    label_all, sentences = generate_triples(args.label_path)
    predict_datas = read_jsonl(args.predict_path)
    predict_all = extract_triplets(predict_datas)

    assert len(label_all) == len(predict_all)

    n_preds, n_labels, n_common = 0, 0, 0
    error_predictions = []

    for i, (pred, label) in enumerate(zip(predict_all, label_all)):
        n_preds += len(pred)
        n_labels += len(label)
        label_dup = label.copy()
        for p in pred:
            if p in label_dup:
                n_common += 1
                label_dup.remove(p)
            else:
                error_info = {
                    "sentence": sentences[i],
                    "label": label,
                    "error_predict": pred
                }
                error_predictions.append(error_info)

    with open(args.output_error_path, 'w', encoding='utf-8') as f:
        json.dump(error_predictions, f, ensure_ascii=False, indent=4)

    precision = n_common / n_preds if n_preds > 0 else 0
    recall = n_common / n_labels if n_labels > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    print("n_labels",n_labels)
    print("n_preds",n_preds)
    print("n_common",n_common)
    print("precision:", precision)
    print("recall:", recall)
    print("f1_score:", f1_score)

    with open(args.results_file, 'a', encoding='utf-8') as f:
        f.write(
            f"Result of checkpoint {args.checkpoint}:\n"
            f"n_labels={n_labels},\n"
            f"n_preds={n_preds},\n"
            f"n_common={n_common},\n"
            f"Precision={precision:.4f},\n"
            f"Recall={recall:.4f},\n" 
            f"F1={f1_score:.4f}\n\n"
    )

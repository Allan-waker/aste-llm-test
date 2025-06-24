import json
import os
from collections import defaultdict
from html import escape

def triplet_to_str(triplet):
    return f"({triplet[0]}, {triplet[1]}, {triplet[2]})"

def generate_html_visualization(error_details, input_path, filename="error_visualization.html"):
    def color_span(text, color):
        return f'<span style="color:{color}; font-weight:bold">{escape(text)}</span>'

    html_lines = ["<html><head><meta charset='utf-8'><style>body{font-family:Arial}</style></head><body>"]
    html_lines.append("<h2>模型预测错误可视化</h2>")

    color_map = {
        'extra_triplet': 'orange',
        'term_error': 'red',
        'sentiment_error': 'blue'
    }
    error_map = {
        'extra_triplet': '正确预测出句子中的三元组，但错误预测出多余的三元组',
        'term_error': '方面词或观点词预测错误',
        'sentiment_error': '情感极性预测错误'
    }
    for err_type, items in error_details.items():
        html_lines.append(f"<h3>错误类型：{error_map[err_type]}（共{len(items)}条）</h3><ul>")
        for item in items:
            sentence = escape(item['sentence'])
            label_set = set(map(triplet_to_str, item['label']))
            pred_set = set(map(triplet_to_str, item['error_predict']))

            html_lines.append(f"<li><strong>句子：</strong> {sentence}<br>")
            html_lines.append("<strong>标签：</strong><br>")
            for l in label_set:
                html_lines.append(color_span("✔ " + l, "green") + "<br>")

            html_lines.append("<strong>预测结果：</strong><br>")
            for p in pred_set:
                if p in label_set:
                    html_lines.append(color_span("✔ " + p, "green") + "<br>")
                else:
                    html_lines.append(color_span("❌ " + p, color_map[err_type]) + "<br>")
            html_lines.append("<hr></li>")
        html_lines.append("</ul>")

    html_lines.append("</body></html>")

    output_dir = os.path.dirname(os.path.abspath(input_path))
    html_path = os.path.join(output_dir, filename)
    with open(html_path, "w", encoding="utf-8") as f:
        f.writelines("\n".join(html_lines))
    print(f"✅ 可视化 HTML 已保存为: {html_path}")

# 加载并去重
def load_and_deduplicate(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)

    seen = set()
    deduplicated_data = []
    for item in data:
        sentence = item['sentence']
        if sentence not in seen:
            seen.add(sentence)
            deduplicated_data.append(item)
    return deduplicated_data

# 标准化三元组
def normalize_triplet(triplet):
    return tuple([x.lower().strip() for x in triplet])

# 错误类型判断与分类输出
def analyze_errors(data):
    stats = {
        'extra_triplet': 0,
        'term_error': 0,
        'sentiment_error': 0
    }
    error_details = defaultdict(list)

    for item in data:
        label_set = set(map(normalize_triplet, item['label']))
        pred_set = set(map(normalize_triplet, item['error_predict']))

        if pred_set == label_set:
            continue  # 预测完全正确

        # 情感极性错误
        label_ap = set((a, p) for a, p, _ in label_set)
        pred_ap = set((a, p) for a, p, _ in pred_set)
        common_ap = label_ap & pred_ap

        polarity_errors = [
            (a, p) for a, p in common_ap
            if (a, p, 'positive') in pred_set and (a, p, 'negative') in label_set or
               (a, p, 'negative') in pred_set and (a, p, 'positive') in label_set or
               (a, p, 'neutral') in pred_set and (a, p, 'positive') in label_set or
               (a, p, 'neutral') in pred_set and (a, p, 'negative') in label_set
        ]
        if polarity_errors:
            stats['sentiment_error'] += 1
            error_details['sentiment_error'].append(item)
            continue

        # 多余三元组错误
        if label_set.issubset(pred_set):
            stats['extra_triplet'] += 1
            error_details['extra_triplet'].append(item)
            continue

        # 方面或观点词错误
        stats['term_error'] += 1
        error_details['term_error'].append(item)

    return stats, error_details

# 保存每类错误到与输入路径相同的文件夹
def save_error_details(error_details, input_path, prefix='errors'):
    output_dir = os.path.dirname(os.path.abspath(input_path))
    for err_type, items in error_details.items():
        save_path = os.path.join(output_dir, f'{prefix}_{err_type}.json')
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(items, f, indent=2, ensure_ascii=False)
        print(f"已保存: {save_path}")

if __name__ == '__main__':
    file_path = 'ASTE_test/laptop14/results/error_laptop14-llama-0615-checkpoint85.json'  # ✅ 替换为你的文件路径
    data = load_and_deduplicate(file_path)
    result, details = analyze_errors(data)

    print("错误类型统计结果：")
    print(f"1. 多余三元组错误：{result['extra_triplet']} 条")
    print(f"2. 方面或观点词错误：{result['term_error']} 条")
    print(f"3. 情感极性错误：{result['sentiment_error']} 条")

    # 保存 JSON
    save_error_details(details, file_path)

    # 保存 HTML 可视化
    generate_html_visualization(details, file_path)
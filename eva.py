import json

# 1. 打开jsonl文件
with open('Reading/LLaMA2-13b/CWQ_Freebase_NQ_lora_epoch10/evaluation/generated_predictions.jsonl', 'r', encoding='utf-8') as f:
    total_lines = 0
    matched_lines = 0

    # 2. 读取每一行
    for line in f:
        total_lines += 1

        # 3. 对于每行，将其解析为字典
        data = json.loads(line)

        # 4. 检查"label"和"predict"的值是否相等
        if data['label'] == data['predict']:
            matched_lines += 1

# 5. 计算相等的行的数量
print(f"Total lines: {total_lines}")
print(f"Matched lines: {matched_lines}")

# 6. 计算相等行的占比
percentage = (matched_lines / total_lines) * 100
print(f"Percentage of matched lines: {percentage:.2f}%")

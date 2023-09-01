from llmtuner import ChatModel
import json
from tqdm import tqdm
import random
import re
import os
from llmtuner.tuner.core import get_infer_args

def main():
    model_args, data_args, _, _ = get_infer_args()
    chat_model = ChatModel()
    output_data = []
    with open(os.path.join(data_args.dataset_dir,data_args.dataset,'examples.json'), 'r', encoding='utf-8') as f:
        json_data = json.load(f)
        # random.shuffle(json_data)
        total_lines = 0
        matched_lines = 0
        will_matched_lines = 0

        # 2. 读取每一行
        for data in tqdm(json_data):
            total_lines += 1
            query = data['instruction']+data['input']
            predict = chat_model.chat_beam(query)
            predict = [p[0] for p in predict]
            output_data.append({'label':data['output'],'predict':predict})
            for p in predict:
                # 4. 检查"label"和"predict"的值是否相等
                if data['output'] == p:
                    matched_lines += 1
                    break
            for p in predict:
                # 4. 检查"label"和"predict"的值是否相等
                if re.sub(r'\[.*?\]', '', data['output']) == re.sub(r'\[.*?\]', '', p):
                    will_matched_lines += 1
                    break
       

    # 5. 计算相等的行的数量
    print(f"Total lines: {total_lines}")
    print(f"Matched lines: {matched_lines}")
    print(f"Will Matched lines: {will_matched_lines}")

    # 6. 计算相等行的占比
    percentage = (matched_lines / total_lines) * 100
    print(f"Percentage of matched lines: {percentage:.2f}%")
    # 6. 计算相等行的占比
    will_percentage = (will_matched_lines / total_lines) * 100
    print(f"Percentage of will matched lines: {will_percentage:.2f}%")
    
    
    output_dir = os.path.join(os.path.dirname(model_args.checkpoint_dir[0]),'evaluation_beam/generated_predictions.jsonl')
    if not os.path.exists(os.path.dirname(output_dir)):
        os.makedirs(os.path.dirname(output_dir))
    with open(output_dir, 'w') as f:
        for item in output_data:
            json_string = json.dumps(item)
            f.write(json_string + '\n')
    
if __name__ == "__main__":
    main()

import os
import json
import argparse
from components.utils import load_json
from tqdm import tqdm
import re

def load_data(split, args):
    if args.dataset_type == "CWQ":
        data_file_name = 'data/CWQ/generation/merged/CWQ_{}_new.json'.format(split)
    elif args.dataset_type == "WebQSP":
        data_file_name = 'data/WebQSP/generation/merged/WebQSP_{}_new.json'.format(split)
    print('Loading data from:',data_file_name)
    data_dict = load_json(data_file_name)
    return data_dict

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_type', default="WebQSP", type=str, help="CWQ | WebQSP")
    parser.add_argument("--retrieval_dir", type=str, default='Retrieval/ChatGLMv2/Retrieval_WebQSP_Freebase_lora_epoch10')
    parser.add_argument("--max_length", type=int, default=1024)
    args = parser.parse_args()
    return args

def prepare_dataloader(args,split):
    assert split in ['train','test','dev','train_sample','dev_sample','test_sample']

    data = load_data(split, args)
    print(f'Origin {split} dataset len: {len(data)}')
    assert type(data)==list
    if 'train' in split or 'dev' in split:
        # for train and dev, filter the examples without sexpr
        examples = []
        for x in data:
            if x['sexpr'].lower()!="null":
                examples.append(x)                
    else:
        examples = [x for x in data]
    print(f'Real {split} dataset len: {len(examples)}')
    
    retrieval_dir = os.path.join(args.retrieval_dir, 'evaluation_{}/generated_predictions.jsonl'.format(split))
    retrieval_data=[]
    with open(retrieval_dir) as file:
        for line in file:
            retrieval_data.append(json.loads(line))

    retrieval_count_dir = 'data/{}/retrieval_count/{}_{}_count.json'.format(args.dataset_type, args.dataset_type, split)
    with open(retrieval_count_dir) as file:
        retrieval_count = json.load(file)
        retrieval_count_list = list(retrieval_count.values())
        assert len(retrieval_count) == len(examples)
        assert sum(retrieval_count_list) == len(retrieval_data)
        
    result = []
    start_idx = 0
    for count in list(retrieval_count_list):
        end_idx = start_idx + count
        result.append(retrieval_data[start_idx:end_idx])
        start_idx = end_idx

    json_data=[]
    instruction='Generate a Logical Form query that retrieves the information corresponding to the given question. \n'
    for cnt, item in tqdm(enumerate(examples)):
        question=item['question']
        input = 'Question: { '+question+' }'       
        output = item['normed_sexpr']
        json_data.append({"instruction":instruction,"input":input,"output":output,"history":[]})
               
    
    output_dir = 'LLMs/data/{}_Freebase_NQ_{}/examples.json'.format(args.dataset_type, split)

    if not os.path.exists(os.path.dirname(output_dir)):
        os.mkdir(os.path.dirname(output_dir))   

    with open(output_dir, "w", encoding="utf-8") as file:
        json.dump(json_data, file)    
    

if __name__=='__main__':
    
    args = _parse_args()
    print(args)
    if args.dataset_type == "CWQ":
        prepare_dataloader(args,'train')
        prepare_dataloader(args,'dev')
        prepare_dataloader(args, 'test')
    elif args.dataset_type == "WebQSP":
        prepare_dataloader(args,'train')
        prepare_dataloader(args, 'test')
    print('Finished')


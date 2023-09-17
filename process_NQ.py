import os
import json
import argparse
from components.utils import load_json
from tqdm import tqdm

def load_data(split, args):
    data_file_name = 'data/{}/generation/merged/{}_{}.json'.format(args.dataset_type,args.dataset_type,split)
    print('Loading data from:',data_file_name)
    data_dict = load_json(data_file_name)
    return data_dict

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_type', default="WebQSP", type=str, help="CWQ | WebQSP")
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
    prepare_dataloader(args,'train')
    prepare_dataloader(args, 'test')
    print('Finished')


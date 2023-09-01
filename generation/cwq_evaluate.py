#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   cwq_evaluate.py
@Time    :   2022/01/10 16:14:58
@Author  :   Xixin Hu 
@Version :   1.0
@Contact :   xixinhu97@foxmail.com
@Desc    :   None
"""

# here put the import lib
import argparse 
from executor import sparql_executor
from components.utils import dump_json, load_json
from tqdm import tqdm
import os


def cwq_evaluate_valid_results(args):
    """Compute P, R and F1 for CWQ"""
    pred_data = load_json(args.pred_file)
    # origin dataset
    dataset_data = load_json(f'data/CWQ/origin/ComplexWebQuestions_{args.split}.json')
    
    dataset_dict = {x["ID"]:x for x in dataset_data}

    p_list = []
    r_list = []
    f_list = []
    p_dict = {}
    r_dict = {}
    f_dict = {}
    acc_num = 0

    pred_dict = {}
    acc_qid_list = [] # Pred Answer ACC
    for pred in pred_data:
        qid = pred['qid']
        pred_answer = set(pred['answer'])
        pred_dict[qid]=pred_answer
    
    for qid,example in tqdm(dataset_dict.items()):
        
        gt_sparql = example['sparql']
        if 'answer' in example:
            gt_answer = set(example['answer'])
        else:
            gt_answer = set(sparql_executor.execute_query(gt_sparql))

        # for dev split
        # gt_answer = set([item["answer_id"] for item in example["answers"]])

        pred_answer = set(pred_dict.get(qid,{}))

        # assert len(pred_answer)>0 and len(gt_answer)>0
        if pred_answer == gt_answer:
            acc_num+=1
            acc_qid_list.append(qid)

        if len(pred_answer)== 0:
            if len(gt_answer)==0:
                p=1
                r=1
                f=1
            else:
                p=0
                r=0
                f=0
        elif len(gt_answer)==0:
            p=0
            r=0
            f=0
        else:
            p = len(pred_answer & gt_answer)/ len(pred_answer)
            r = len(pred_answer & gt_answer)/ len(gt_answer)
            f = 2*(p*r)/(p+r) if p+r>0 else 0
        
        p_list.append(p)
        r_list.append(r)
        f_list.append(f)
        p_dict[qid] = p
        r_dict[qid] = r
        f_dict[qid] = f
    
    p_average = sum(p_list)/len(p_list)
    r_average = sum(r_list)/len(r_list)
    f_average = sum(f_list)/len(f_list)

    res = f'Total: {len(p_list)}, ACC:{acc_num/len(p_list)}, AVGP: {p_average}, AVGR: {r_average}, AVGF: {f_average}'
    print(res)
    dirname = os.path.dirname(args.pred_file)
    filename = os.path.basename(args.pred_file)
    with open (os.path.join(dirname,f'{filename}_final_eval_results.txt'),'w') as f:
        f.write(res)
        f.flush()
    
    # Write answer acc result to prediction file
    for pred in pred_data:
        qid = pred['qid']
        if qid in acc_qid_list:
            pred['answer_acc'] = True
        else:
            pred['answer_acc'] = False
        pred['precision'] = p_dict[qid] if qid in p_dict else None
        pred['recall'] = r_dict[qid] if qid in r_dict else None
        pred['f1'] = f_dict[qid] if qid in f_dict else None
    
    dump_json(pred_data, os.path.join(dirname, f'{filename}_new.json'), indent=4)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        help="split to operate on, can be `test`, `dev` and `train`",
    )
    parser.add_argument(
        "--pred_file", type=str, default=None, help="prediction results file"
    )
    
    args = parser.parse_args()
    
    cwq_evaluate_valid_results(args)
    





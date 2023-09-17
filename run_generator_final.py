import os
import argparse
import json
from components.utils import dump_json

def prepare_dataloader(args):
    print('Loading data from:',args.data_file_name)
    with open(args.data_file_name, 'r', encoding='utf-8') as f:
        # 读取每一行并转换为字典
        data = [json.loads(line) for line in f]
    print(f'Dataset len: {len(data)}')
    return data


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file_name',default='Reading/LLaMA2-13b/WebQSP_Freebase_NQ_lora_epoch100/evaluation_beam/generated_predictions.jsonl')

    args = parser.parse_args()
    return args


def run_prediction(args,dataloader,output_dir,output_predictions=True):
    print()
    print('Start predicting ')
            
    ex_cnt = 0
    contains_ex_cnt = 0
    output_list = []
    real_total = 0
    for i,pred in enumerate(dataloader):
        predictions = pred['predict']
        gen_label = pred['label']

        output_list.append({
            'predictions':predictions,
            'gen_label':gen_label,
        })

        if predictions[0].lower()==gen_label.lower():
            ex_cnt+=1

        if any([x.lower()==gen_label.lower() for x in predictions]):
            contains_ex_cnt+=1
        
        if gen_label.lower()!='null':
            real_total+=1

    
    print(f"""total:{len(output_list)}, 
                    ex_cnt:{ex_cnt}, 
                    ex_rate:{ex_cnt/len(output_list)}, 
                    real_ex_rate:{ex_cnt/real_total}, 
                    contains_ex_cnt:{contains_ex_cnt}, 
                    contains_ex_rate:{contains_ex_cnt/len(output_list)}
                    real_contains_ex_rate:{contains_ex_cnt/real_total}
                    """)

        
    if output_predictions:
        file_path = os.path.join(output_dir,f'beam_test_top_k_predictions.json')
        
        gen_statistics_file_path = os.path.join(output_dir,f'beam_test_gen_statistics.json')
        gen_statistics = {
            'total':len(output_list),
            'exmatch_num': ex_cnt,
            'exmatch_rate': ex_cnt/len(output_list),
            'real_exmatch_rate':ex_cnt/real_total, 
            'contains_ex_num':contains_ex_cnt,
            'contains_ex_rate':contains_ex_cnt/len(output_list),
            'real_contains_ex_rate':contains_ex_cnt/real_total
        }
        dump_json(output_list, file_path, indent=4)
        dump_json(gen_statistics, gen_statistics_file_path,indent=4)


if __name__=='__main__':
    
    args = _parse_args()
    print(args)

    test_dataloader = prepare_dataloader(args)
    run_prediction(args,test_dataloader,output_dir=os.path.dirname(args.data_file_name),output_predictions=True)

    print('Prediction Finished')


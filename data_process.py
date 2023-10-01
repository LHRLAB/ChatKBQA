from collections import defaultdict
import random
from typing import Dict, List
from components.utils import (
    _textualize_relation,
    load_json, 
    dump_json, 
    extract_mentioned_entities_from_sparql, 
    extract_mentioned_relations_from_sparql,
    vanilla_sexpr_linearization_method
)
import argparse
from tqdm import tqdm
import os
import torch
import pandas as pd
from executor.sparql_executor import (
    get_label_with_odbc,    
    get_types_with_odbc,
    get_out_relations_with_odbc,
    get_in_relations_with_odbc,
    get_entity_labels
)


def _parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--action',default='merge_all',help='Action to operate')
    parser.add_argument('--dataset', default='WebQSP', help='dataset to perform entity linking, should be CWQ or WebQSP')
    parser.add_argument('--split', default='train', help='split to operate on') # the split file: ['dev','test','train']
    
    
    return parser.parse_args()


def combine_entities_from_FACC1_and_elq(dataset, split, sample_size=10):
    """ Combine the linking results from FACC1 and ELQ """
    entity_dir = f'data/{dataset}/entity_retrieval/candidate_entities'

    facc1_disamb_res = load_json(f'{entity_dir}/{dataset}_{split}_cand_entities_facc1.json')
    elq_res = load_json(f'{entity_dir}/{dataset}_{split}_cand_entities_elq.json')

    combined_res = dict()

    train_entities_elq = {}
    elq_res_train = load_json(f'{entity_dir}/{dataset}_train_cand_entities_elq.json')
    for qid,cand_ents in elq_res_train.items():
        for ent in cand_ents:
            train_entities_elq[ent['id']] = ent['label']
        
    
    train_entities_elq = [{"id":mid,"label":label} for mid,label in train_entities_elq.items()]


    for qid in tqdm(elq_res,total=len(elq_res),desc=f'Merging candidate entities of {split}'):
        cur = dict() # unique by mid

        elq_result = elq_res[qid]
        facc1_result = facc1_disamb_res.get(qid,[])
        
        # sort by score
        elq_result = sorted(elq_result, key=lambda d: d.get('score', -20.0), reverse=True)
        facc1_result = sorted(facc1_result, key=lambda d: d.get('logit', -20.0), reverse=True)

        # merge the linking results of ELQ and FACC1 one by one
        idx = 0
        while len(cur.keys()) < sample_size:
            if idx < len(elq_result):
                cur[elq_result[idx]["id"]] = elq_result[idx]
            if len(cur.keys()) < sample_size and idx < len(facc1_result):
                cur[facc1_result[idx]["id"]] = facc1_result[idx]
            if idx >= len(elq_result) and idx >= len(facc1_result):
                break
            idx += 1

        if len(cur.keys()) < sample_size:
            # sample some entities to reach the sample size
            diff_entities = list(filter(lambda v: v["id"] not in cur.keys(), train_entities_elq))
            random_entities = random.sample(diff_entities, 10 - len(cur.keys()))
            for ent in random_entities:
                cur[ent["id"]] = ent

        assert len(cur.keys()) == sample_size, print(qid)
        combined_res[qid] = list(cur.values())

    
    merged_file_path = f'{entity_dir}/{dataset}_{split}_merged_cand_entities_elq_facc1.json'
    print(f'Writing merged candidate entities to {merged_file_path}')
    dump_json(combined_res, merged_file_path, indent=4)

    if dataset.lower() == 'cwq':
        update_entity_label(dirname=entity_dir, dataset=dataset)


def make_sorted_relation_dataset_from_logits(dataset, split):

    assert dataset in ['CWQ','WebQSP']
    if dataset == 'WebQSP':
        assert split in ['train', 'test', 'train_2hop', 'test_2hop']
    else:
        assert split in ['test','train','dev']

    output_dir = f'data/{dataset}/relation_retrieval/candidate_relations'
    
    
    if dataset=='CWQ':
        tsv_file = f'data/CWQ/relation_retrieval/cross-encoder/mask_mention_1epoch_question_relation/CWQ.{split}.tsv'
        logits_file = f'data/CWQ/relation_retrieval/cross-encoder/saved_models/mask_mention_1epoch_question_relation/CWQ_ep_1.pt_{split}/logits.pt'
        idmap = load_json(f'data/CWQ/relation_retrieval/cross-encoder/mask_mention_1epoch_question_relation/CWQ_{split}_id_index_map.json')
    elif dataset=='WebQSP':
        tsv_file = f'data/WebQSP/relation_retrieval/cross-encoder/rich_relation_3epochs_question_relation/WebQSP.{split}.tsv'
        logits_file = f'data/WebQSP/relation_retrieval/cross-encoder/saved_models/rich_relation_3epochs_question_relation/WebQSP_ep_3.pt_{split}/logits.pt'
        idmap = load_json(f'data/WebQSP/relation_retrieval/cross-encoder/rich_relation_3epochs_question_relation/WebQSP_{split}_id_index_map.json')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logits = torch.load(logits_file,map_location=torch.device('cpu'))

    logits_list = list(logits.squeeze().numpy())
    print('Logits len:',len(logits_list))
    print('tsv_file: {}'.format(tsv_file))
    tsv_df = pd.read_csv(tsv_file, delimiter='\t',dtype={"id":int, "question":str, "relation":str, 'label':int})
                            

    print('Tsv len:', len(tsv_df))
    print('Question Num:',len(tsv_df['question'].unique()))

    # the length of predicted logits must match the num of input examples
    assert(len(logits_list)==len(tsv_df))

    
    if dataset.lower()=='webqsp':
        if split in ['train_2hop', 'train']:
            split_dataset = load_json(f'data/{dataset}/sexpr/{dataset}.train.expr.json')
        elif split in ['test', 'test_2hop']:
            split_dataset = load_json(f'data/{dataset}/sexpr/{dataset}.test.expr.json')
    else:
        split_dataset = load_json(f'data/{dataset}/sexpr/{dataset}.{split}.expr.json')


    rowid2qid = {} # map rowid to qid
        

    for qid in idmap:
        rowid_start = idmap[qid]['start']
        rowid_end = idmap[qid]['end']
        #rowid2qid[rowid]=qid
        for i in range(rowid_start,rowid_end+1):
           rowid2qid[i]=qid


    # cand_rel_bank = {} # Dict[Question, Dict[Relation:logit]]
    cand_rel_bank = defaultdict(dict)
    rel_info_map = defaultdict(str)
    for idx,logit in tqdm(enumerate(logits_list),total=len(logits_list),desc=f'Reading logits of {split}'):
        logit = float(logit[1])
        row_id = tsv_df.loc[idx]['id']
        question = tsv_df.loc[idx]['question']
        rich_rel = tsv_df.loc[idx]['relation']
        rel = rich_rel.split("|")[0]
        rel_info = " | ".join(rich_rel.split("|")).replace("."," , ").replace("_"," ")
        #cwq_id = question2id.get(question,None)
        qid = rowid2qid[row_id]

        if not qid:
            print(question)
            cand_rel_bank[qid]= {}
        else:
            cand_rel_bank[qid][rel]=logit

        if not rel in rel_info_map:
            rel_info_map[rel] = rel_info
            

    cand_rel_logit_map = {}
    for qid in tqdm(cand_rel_bank,total=len(cand_rel_bank),desc='Sorting rels...'):
        cand_rel_maps = cand_rel_bank[qid]
        cand_rel_list = [(rel,logit,rel_info_map[rel]) for rel,logit in cand_rel_maps.items()]
        cand_rel_list.sort(key=lambda x:x[1],reverse=True)
        
        cand_rel_logit_map[qid]=cand_rel_list

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dump_json(cand_rel_logit_map,os.path.join(output_dir,f'{dataset}_{split}_cand_rel_logits.json'),indent=4)

    final_candRel_map = defaultdict(list) # Dict[Question,List[Rel]]   sorted by logits

    for ori_data in tqdm(split_dataset,total=len(split_dataset),desc=f'{split} Dumping... '):
        if dataset=='CWQ':
            qid = ori_data['ID']
        else:
            qid = ori_data['QuestionId']
        # cand_rel_map = cand_rel_bank.get(qid,None)
        cand_rel_list = cand_rel_logit_map.get(qid,None)
        if not cand_rel_list:
            final_candRel_map[qid]=[]
        else:
            # cand_rel_list = list(cand_rel_map.keys())
            # cand_rel_list.sort(key=lambda x:float(cand_rel_map[x]),reverse=True)
            final_candRel_map[qid]=[x[0] for x in cand_rel_list]

    sorted_cand_rel_name = os.path.join(output_dir,f'{dataset}_{split}_cand_rels_sorted.json')
    dump_json(final_candRel_map,sorted_cand_rel_name,indent=4)   


def get_all_unique_candidate_entities(dataset)->List[str]:
    """Get unique candidate entity ids of {dataset}"""

    ent_dir = f'data/{dataset}/entity_retrieval/candidate_entities'
    unique_entities_file = f'{ent_dir}/{dataset}_candidate_entity_ids_unique.json'
    
    if os.path.exists(unique_entities_file):
        print(f'Loading unique candidate entities from {unique_entities_file}')
        unique_entities = load_json(unique_entities_file)
    else:
        print(f'Processing candidate entities...')

        train_data = load_json(f'{ent_dir}/{dataset}_train_merged_cand_entities_elq_facc1.json')
        test_data = load_json(f'{ent_dir}/{dataset}_test_merged_cand_entities_elq_facc1.json')

        if dataset=='CWQ':
            dev_data = load_json(f'{ent_dir}/{dataset}_dev_merged_cand_entities_elq_facc1.json')
        else:
            dev_data = None

        unique_entities = set()
        for qid in train_data.keys():
            for ent in train_data[qid]:
                unique_entities.add(ent["id"])
        
        
        for qid in test_data.keys():
            for ent in test_data[qid]:
                unique_entities.add(ent["id"])

        if dev_data:
            for qid in dev_data.keys():
                for ent in dev_data[qid]:
                    unique_entities.add(ent["id"])
                
        print(f'Wrinting unique candidate entities to {unique_entities_file}')
        dump_json(list(unique_entities), unique_entities_file ,indent=4)
    
    return unique_entities


def get_entities_in_out_relations(dataset,unique_candidate_entities)->Dict[str,Dict[str,List[str]]]:
    
    ent_dir = f'data/{dataset}/entity_retrieval/candidate_entities'
    in_out_rels_file = f'{ent_dir}/{dataset}_candidate_entities_in_out_relations_new.json'

    if os.path.exists(in_out_rels_file):
        print(f'Loading cached 1hop relations from {in_out_rels_file}')
        in_out_rels = load_json(in_out_rels_file)
    else:

        if unique_candidate_entities:
            entities = unique_candidate_entities
        else:
            unique_entities_file = f'{ent_dir}/{dataset}_candidate_entity_ids_unique.json'
            entities = load_json(unique_entities_file)            
        
        IGONORED_DOMAIN_LIST = ['type', 'common', 'kg', 'dataworld', 'freebase', 'user']
        in_out_rels = dict()
        
        for ent in tqdm(entities,total=len(entities),desc='Fetching 1hop relations of candidate entities'):            
            
            relations_out = get_out_relations_with_odbc(ent)
            relations_out = [x for x in relations_out if x.split('.')[0] not in IGONORED_DOMAIN_LIST]
            relations_in = get_in_relations_with_odbc(ent)
            relations_in = [x for x in relations_in if x.split('.')[0] not in IGONORED_DOMAIN_LIST]
            in_out_rels[ent] = {
                'out_relations': relations_out,
                'in_relations': relations_in
            }
        
        print(f'Writing 1hop relations to {in_out_rels_file}')
        dump_json(in_out_rels, in_out_rels_file)

    return in_out_rels
    

def merge_all_data_for_logical_form_generation(dataset, split):

    dataset_with_sexpr = load_json(f'data/{dataset}/sexpr/{dataset}.{split}.expr.json')
        
    global_ent_label_map = {}
    global_rel_label_map = {}
    global_type_label_map = {}
    
    merged_data_all = []

    for example in tqdm(dataset_with_sexpr, total=len(dataset_with_sexpr), desc=f'Processing {dataset}_{split}'):
        
        new_example = {}
        
        if dataset=='CWQ':
            qid = example["ID"]
        elif dataset=='WebQSP':
            qid = example['QuestionId']
        question = example['question'] if dataset=='CWQ' else example['ProcessedQuestion']
        comp_type = example["compositionality_type"] if dataset=='CWQ' else None                
        
        if dataset=='CWQ':
            sexpr = example['SExpr']
            sparql = example['sparql']
            if split=='test':
                answer = example["answer"]
            else:
                answer = [x['answer_id'] for x in example['answers']]
        elif dataset=='WebQSP':
            # for WebQSP choose 
            # 1. shortest sparql
            # 2. s-expression converted from this sparql should leads to same execution results.
            parses = example['Parses']
            shortest_idx = 0
            shortest_len = 9999
            for i in range(len(parses)):
                if 'SExpr_execute_right' in parses[i] and parses[i]['SExpr_execute_right']:
                    if len(parses[i]['Sparql']) < shortest_len:
                        shortest_idx = i
                        shortest_len = len(parses[i]['Sparql'])
                
            sexpr = parses[shortest_idx]['SExpr']
            sparql = parses[shortest_idx]['Sparql']
            answer = [x['AnswerArgument'] for x in parses[shortest_idx]['Answers']]
                
        gold_ent_label_map = {}
        gold_rel_label_map = {}
        gold_type_label_map = {}
        # normed_sexpr = example['question']
            
        normed_sexpr = vanilla_sexpr_linearization_method(sexpr)
        gold_entities = extract_mentioned_entities_from_sparql(sparql)
        gold_relations = extract_mentioned_relations_from_sparql(sparql)
        
        for entity in gold_entities:
            is_type = False
            entity_types = get_types_with_odbc(entity)
            if "type.type" in entity_types:
                is_type = True

            entity_label = get_label_with_odbc(entity)
            if entity_label is not None:
                gold_ent_label_map[entity] = entity_label
                global_ent_label_map[entity] = entity_label

            if is_type and entity_label is not None:
                gold_type_label_map[entity] = entity_label
                global_type_label_map[entity] = entity_label

            
        for rel in gold_relations:
            linear_rel = _textualize_relation(rel)
            gold_rel_label_map[rel] = linear_rel
            global_rel_label_map[rel] = linear_rel

        
        new_example['ID']=qid
        new_example['question'] = question
        new_example['comp_type'] = comp_type
        new_example['answer'] = answer
        new_example['sparql'] = sparql
        new_example['sexpr'] = sexpr
        new_example['normed_sexpr'] = normed_sexpr
        new_example['gold_entity_map'] = gold_ent_label_map
        new_example['gold_relation_map'] = gold_rel_label_map
        new_example['gold_type_map'] = gold_type_label_map


        merged_data_all.append(new_example)
    
    merged_data_dir = f'data/{dataset}/generation/merged'
    if not os.path.exists(merged_data_dir):
        os.makedirs(merged_data_dir)
    merged_data_file = f'{merged_data_dir}/{dataset}_{split}.json'

    print(f'Wrinting merged data to {merged_data_file}...')
    dump_json(merged_data_all,merged_data_file,indent=4)
    print('Writing finished')
                

def get_merged_disambiguated_entities(dataset, split):
    """Get disambiguated entities by entity retrievers (one entity for one mention)"""
    
    disamb_ent_dir = f"data/{dataset}/entity_retrieval/disamb_entities"
    
    disamb_ent_file = f"{disamb_ent_dir}/{dataset}_merged_{split}_disamb_entities.json"

    if os.path.exists(disamb_ent_file):
        print(f'Loading disamb entities from {disamb_ent_file}')
        disamb_ent_map = load_json(disamb_ent_file)
        return disamb_ent_map
    else:
        cand_ent_dir = f"data/{dataset}/entity_retrieval/candidate_entities"
        elq_cand_ent_file = f"{cand_ent_dir}/{dataset}_{split}_cand_entities_elq.json"
        facc1_cand_ent_file = f"{cand_ent_dir}/{dataset}_{split}_cand_entities_facc1.json"
        
        elq_cand_ents = load_json(elq_cand_ent_file)
        facc1_cand_ents = load_json(facc1_cand_ent_file)

        # entities linked and ranked by elq
        elq_disamb_ents = {}        
        for qid,cand_ents in elq_cand_ents.items():
            mention_cand_map = {}
            for ent in cand_ents:
                if ent['mention'] not in mention_cand_map:
                    mention_cand_map[ent['mention']]=ent
            
            elq_disamb_ents[qid] = [ent for (_,ent) in mention_cand_map.items()]

        # entities linked and ranked by facc1
        facc1_disamb_ents = {}
        for qid,cand_ents in facc1_cand_ents.items():
            mention_cand_map = {}
            for ent in cand_ents:
                if ent['mention'] not in mention_cand_map:
                    mention_cand_map[ent['mention']]=ent            
            facc1_disamb_ents[qid] = [ent for (_,ent) in mention_cand_map.items()]


        disamb_ent_map = {}
        
        # merge the disambed entities        
        for qid in elq_disamb_ents:
            disamb_entities = {}

            facc1_entities = facc1_disamb_ents[qid]
            elq_entities = elq_disamb_ents[qid]

            if dataset.lower() == 'cwq':
                for ent in facc1_entities:
                    disamb_entities[ent['id']]={
                        "id":ent["id"],
                        "label":ent["label"],
                        "mention":ent["mention"],
                        "perfect_match":ent["perfect_match"]
                    }

                elq_entities = [ent for ent in elq_entities if ent['score'] > -1.5]
                for ent in elq_entities:
                    if ent["id"] not in disamb_entities: # different id
                        if ent["label"]:
                            disamb_entities[ent['id']] = {
                                "id":ent["id"],
                                "label":ent["label"],
                                "mention":ent["mention"],
                                "perfect_match":ent["perfect_match"]
                            }

            elif dataset.lower() == 'webqsp':
                for ent in elq_entities:
                    disamb_entities[ent['id']]={
                        "id":ent["id"],
                        "label": get_label_with_odbc(ent['id']),
                        "mention":ent["mention"],
                        "perfect_match":ent["perfect_match"]
                    }
                
                for ent in facc1_entities:
                    if ent['id'] not in disamb_entities: # different id
                        disamb_entities[ent['id']]={
                            "id":ent["id"],
                            "label":get_label_with_odbc(ent['id']),
                            "mention":ent["mention"],
                            "perfect_match":ent["perfect_match"]
                        }
            
            disamb_entities = [ent for (_,ent) in disamb_entities.items()]

            disamb_ent_map[qid] = disamb_entities

        print(f'Writing disamb entities into {disamb_ent_file}')
        if not os.path.exists(disamb_ent_dir):
            os.makedirs(disamb_ent_dir)

        dump_json(disamb_ent_map, disamb_ent_file, indent=4)

        return disamb_ent_map
        



def extract_type_label_from_dataset(dataset, split):
    
    
    train_databank =load_json(f"data/{dataset}/sexpr/{dataset}.{split}.expr.json")

    global_type_label_map = {}

    for data in tqdm(train_databank, total=len(train_databank), desc=f"Processing {split}"):
        qid = data['ID']
        sparql = data['sparql']

        type_label_map = {}

        # extract entity labels
        gt_entities = extract_mentioned_entities_from_sparql(sparql=sparql)
        for entity in gt_entities:
            is_type = False
            entity_types = get_types_with_odbc(entity)
            if "type.type" in entity_types:
                is_type = True

            entity_label = get_label_with_odbc(entity)

            if is_type and entity_label is not None:
                type_label_map[entity] = entity_label
                global_type_label_map[entity] = entity_label

    dir_name = f"data/{dataset}/generation/label_maps"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    dump_json(global_type_label_map, f'{dir_name}/{dataset}_{split}_type_label_map.json',indent=4)

    print("done")


def extract_type_label_from_dataset_webqsp(dataset, split):
    # Each WebQSP question may have more than one "Parse"，get label_map of all "Parse"s
    
    train_databank =load_json(f"data/{dataset}/sexpr/{dataset}.{split}.expr.json")

    global_type_label_map = {}

    for data in tqdm(train_databank, total=len(train_databank), desc=f"Processing {split}"):
        qid = data['QuestionId']
        type_label_map = {}
        
        for parse in data["Parses"]:
            sparql = parse["Sparql"]
            # extract entity labels
            gt_entities = extract_mentioned_entities_from_sparql(sparql=sparql)
            for entity in gt_entities:
                is_type = False
                entity_types = get_types_with_odbc(entity)
                if "type.type" in entity_types:
                    is_type = True

                entity_label = get_label_with_odbc(entity)

                if is_type and entity_label is not None:
                    type_label_map[entity] = entity_label
                    global_type_label_map[entity] = entity_label

    dir_name = f"data/{dataset}/generation/label_maps"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    dump_json(global_type_label_map, f'{dir_name}/{dataset}_{split}_type_label_map.json',indent=4)

    print("done")


def get_all_entity(dirname, dataset):
    all_entity = set()
    for split in ['train', 'dev', 'test']:
        el_res = load_json(f'{dirname}/{dataset}_{split}_merged_cand_entities_elq_facc1.json')
        for qid in el_res:
            values = el_res[qid]
            for item in values:
                all_entity.add(item['id'])
                    
    dump_json(list(all_entity), f'{dirname}/{dataset}_all_entities.json')


def update_entity_label(dirname, dataset):
    """Stardardize all entity labels"""
    if not (
        os.path.exists(f'{dirname}/{dataset}_train_merged_cand_entities_elq_facc1.json') and
        os.path.exists(f'{dirname}/{dataset}_dev_merged_cand_entities_elq_facc1.json') and
        os.path.exists(f'{dirname}/{dataset}_test_merged_cand_entities_elq_facc1.json')
    ):
        return # Update label when all dataset splits are ready

    if not os.path.exists(f'{dirname}/{dataset}_all_entities.json'):
        get_all_entity(dirname, dataset)
    assert os.path.exists(f'{dirname}/{dataset}_all_entities.json')

    if not os.path.exists(f'{dirname}/{dataset}_all_label_map.json'):
        get_entity_labels(
            f'{dirname}/{dataset}_all_entities.json',
            f'{dirname}/{dataset}_all_label_map.json'
        )
    assert os.path.exists(f'{dirname}/{dataset}_all_label_map.json')
    
    for split in ['train', 'dev', 'test']:
        el_res = load_json(f'{dirname}/{dataset}_{split}_merged_cand_entities_elq_facc1.json')
        all_label_map = load_json(f'{dirname}/{dataset}_all_label_map.json')
        
        updated_res = dict()

        for qid in el_res:
            values = el_res[qid]
            for item in values:
                if item["id"] in all_label_map:
                    item['label'] = all_label_map[item['id']]
                else:
                    print(item["id"])
                    
            updated_res[qid] = values
        
        dump_json(updated_res, f'{dirname}/{dataset}_{split}_merged_cand_entities_elq_facc1.json'.format(split))

def substitude_relations_in_merged_file(
    prev_merged_path, 
    output_path, 
    sorted_relations_path,
    addition_relations_path,
    topk=10
):
    """
    replace "cand_relation_list" property in previous merged data
    with new relation logits
    """
    prev_merged = load_json(prev_merged_path)
    sorted_relations = load_json(sorted_relations_path) # inference on 2hop
    additional_relation = load_json(addition_relations_path) # inference on bi-encoder top100
    new_merged = []
    for example in tqdm(prev_merged, total=len(prev_merged)):
        qid = example["ID"]
        if qid not in sorted_relations or len(sorted_relations[qid]) < topk: # need exactly 10 relations
            print(qid)
            cand_relations = additional_relation[qid][:topk]
        else:
            cand_relations = sorted_relations[qid][:topk]
        example["cand_relation_list"] = cand_relations
        new_merged.append(example)
    assert len(prev_merged) == len(new_merged)
    dump_json(new_merged, output_path)

def validation_merged_file(prev_file, new_file):
    prev_data = load_json(prev_file)
    new_data = load_json(new_file)
    assert len(prev_data) == len(new_data), print(len(prev_data), len(new_data))
    for (prev, new) in tqdm(zip(prev_data, new_data), total=len(prev_data)):
        for key in prev.keys():
            if key != 'cand_relation_list':
                assert prev[key] == new[key]
            else:
                assert len(prev[key]) == 10
                assert len(new[key]) == 10, print(len(new[key]))


def substitude_relations_in_merged_file_cwq(
    prev_merged_path, 
    output_path, 
    sorted_logits_path,
    topk=10,
):
    prev_merged = load_json(prev_merged_path)
    sorted_logits = load_json(sorted_logits_path)
    new_merged = []
    for example in tqdm(prev_merged, total=len(prev_merged)):
        qid = example["ID"]
        if qid not in sorted_logits:
            print(qid)
        example["cand_relation_list"] = sorted_logits[qid][:topk]
        new_merged.append(example)
    dump_json(new_merged, output_path)

def get_candidate_unique_entities_cwq():
    folder = 'data/CWQ/entity_retrieval/disamb_entities'
    unique_entities = set()
    for split in ['train', 'dev', 'test']:
        cand_entity_file = load_json(os.path.join(folder, f'CWQ_merged_{split}_disamb_entities.json'))
        for qid in cand_entity_file:
            for item in cand_entity_file[qid]:
                if item["id"] != "":
                    unique_entities.add(item["id"])
    dump_json(list(unique_entities), os.path.join(folder, 'unique_entities.json'))

def serialize_rich_relation(relation, domain_range_dict, seperator="|"):
    if relation not in domain_range_dict:
        return relation
    else:
        res = relation
        if 'label' in domain_range_dict[relation]:
            if relation.lower() != domain_range_dict[relation]['label'].lower().replace(' ', ''):
                res += (seperator + domain_range_dict[relation]['label'])
        if 'domain' in domain_range_dict[relation]:
            res += (seperator + domain_range_dict[relation]['domain'])
        if 'range' in domain_range_dict[relation]:
            res += (seperator + domain_range_dict[relation]['range'])
        return res

def construct_common_data(
    filtered_relations_path,
    domain_range_label_map_path,
    output_relation_rich_map_path,
    output_rich_relation_map_path,
    output_filtered_rich_relation_path,
):
    filtered_relations = load_json(filtered_relations_path)
    domain_range_label_map = load_json(domain_range_label_map_path)
    relation_rich_map = dict()
    rich_relation_map = defaultdict(list)
    filtered_rich_relations = []
    for rel in filtered_relations:
        richRelation = serialize_rich_relation(rel, domain_range_label_map).replace('\n', '')
        relation_rich_map[rel] = richRelation
        rich_relation_map[richRelation].append(rel)
        filtered_rich_relations.append(richRelation)
    dump_json(relation_rich_map, output_relation_rich_map_path)
    dump_json(rich_relation_map, output_rich_relation_map_path)
    dump_json(filtered_rich_relations, output_filtered_rich_relation_path)


if __name__=='__main__':
    
    
    args = _parse_args()
    action = args.action

    if action.lower()=='merge_entity':
        combine_entities_from_FACC1_and_elq(dataset=args.dataset, split=args.split)
    elif action.lower()=='merge_relation':
        make_sorted_relation_dataset_from_logits(dataset=args.dataset, split=args.split)
    elif action.lower()=='merge_all':
        merge_all_data_for_logical_form_generation(dataset=args.dataset, split=args.split)
    elif action.lower()=='get_type_label_map':
        if args.dataset == "CWQ":
            extract_type_label_from_dataset(dataset=args.dataset, split=args.split)
        elif args.dataset == "WebQSP":
            extract_type_label_from_dataset_webqsp(dataset=args.dataset, split=args.split)
    else:
        print('usage: data_process.py action [--dataset DATASET] --split SPLIT ')

    # construct_common_data(
    #     'data/common_data/freebase_relations_filtered.json',
    #     'data/common_data/fb_relations_domain_range_label.json',
    #     'data/common_data/fb_relation_rich_map.json',
    #     'data/common_data/fb_rich_relation_map.json',
    #     'data/common_data/freebase_richRelations_filtered.json',
    # )

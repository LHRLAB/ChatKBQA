#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# here put the import lib
import argparse
from generation.cwq_evaluate import cwq_evaluate_valid_results
from generation.webqsp_evaluate_offcial import webqsp_evaluate_valid_results
from components.utils import dump_json, load_json
from tqdm import tqdm
from executor.sparql_executor import execute_query_with_odbc, get_2hop_relations_with_odbc_wo_filter
from utils.logic_form_util import lisp_to_sparql
import re
import os
from entity_retrieval import surface_index_memory
import difflib
import itertools
from simcse import SimCSE
import shutil
model = SimCSE("princeton-nlp/unsup-simcse-roberta-large")

def is_number(t):
    t = t.replace(" , ",".")
    t = t.replace(", ",".")
    t = t.replace(" ,",".")
    try:
        float(t)
        return True
    except ValueError:
        pass
    try:
        import unicodedata  # handle ascii
        unicodedata.numeric(t)  # string of number --> float
        return True
    except (TypeError, ValueError):
        pass
    return False


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='test', help='split to operate on, can be `test`, `dev` and `train`')
    parser.add_argument('--pred_file', default='Reading/LLaMA2-13b/GrailQA_Freebase_NQ_lora_epoch10/evaluation_beam/beam_test_top_k_predictions.json', help='topk prediction file')
    parser.add_argument('--server_ip', default=None, help='server ip for debugging')
    parser.add_argument('--server_port', default=None, help='server port for debugging')
    parser.add_argument('--qid',default=None,type=str, help='single qid for debug, None by default' )
    parser.add_argument('--test_batch_size', default=2)
    parser.add_argument('--dataset', default='GrailQA', type=str)
    parser.add_argument('--beam_size', default=50, type=int)
    parser.add_argument('--golden_ent', default=False, action='store_true')

    args = parser.parse_args()

    print(f'split:{args.split}, topk_file:{args.pred_file}')
    return args

def type_checker(token:str):
    """Check the type of a token, e.g. Integer, Float or date.
       Return original token if no type is detected."""
    
    pattern_year = r"^\d{4}$"
    pattern_year_month = r"^\d{4}-\d{2}$"
    pattern_year_month_date = r"^\d{4}-\d{2}-\d{2}$"
    if re.match(pattern_year, token):
        if int(token) < 3000: # >= 3000: low possibility to be a year
            token = token+"^^http://www.w3.org/2001/XMLSchema#dateTime"
    elif re.match(pattern_year_month, token):
        token = token+"^^http://www.w3.org/2001/XMLSchema#dateTime"
    elif re.match(pattern_year_month_date, token):
        token = token+"^^http://www.w3.org/2001/XMLSchema#dateTime"
    else:
        return token

    return token


def date_post_process(date_string):
    """
    When quering KB, (our) KB tends to autoComplete a date
    e.g.
        - 1996 --> 1996-01-01
        - 1906-04-18 --> 1906-04-18 05:12:00
    """
    pattern_year_month_date = r"^\d{4}-\d{2}-\d{2}$"
    pattern_year_month_date_moment = r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$"

    if re.match(pattern_year_month_date_moment, date_string):
        if date_string.endswith('05:12:00'):
            date_string = date_string.replace('05:12:00', '').strip()
    elif re.match(pattern_year_month_date, date_string):
        if date_string.endswith('-01-01'):
            date_string = date_string.replace('-01-01', '').strip()
    return date_string
        


def denormalize_s_expr_new(normed_expr, 
                            entity_label_map,
                            type_label_map,
                            surface_index):
    
    
    expr = normed_expr

    convert_map ={
        '( greater equal': '( ge',
        '( greater than':'( gt',
        '( less equal':'( le',
        '( less than':'( lt'
    }

    for k in convert_map:
        expr = expr.replace(k,convert_map[k])
        expr = expr.replace(k.upper(),convert_map[k])

    # expr = expr.replace(', ',' , ')
    tokens = expr.split(' ')

    segments = []
    prev_left_bracket = False
    prev_left_par = False
    cur_seg = ''

    for t in tokens:
        
        if t=='[':
            prev_left_bracket=True
            if cur_seg:
                segments.append(cur_seg)
        elif t==']':
            prev_left_bracket=False
            cur_seg = cur_seg.strip()
            
            # find in linear origin map
            processed = False

            if not processed:
                if cur_seg.lower() in type_label_map: # type
                    cur_seg = type_label_map[cur_seg.lower()]
                    processed = True
                else: # relation or unlinked entity
                    if ' , ' in cur_seg: 
                        if is_number(cur_seg):
                            # check if it is a number
                            cur_seg = cur_seg.replace(" , ",".")
                            cur_seg = cur_seg.replace(" ,",".")
                            cur_seg = cur_seg.replace(", ",".")
                        else:
                            # view as relation
                            cur_seg = cur_seg.replace(' , ',',')
                            cur_seg = cur_seg.replace(',','.')
                            cur_seg = cur_seg.replace(' ', '_')
                        processed = True
                    else:
                        search = True
                        if is_number(cur_seg):
                            search = False
                            cur_seg = cur_seg.replace(" , ",".")
                            cur_seg = cur_seg.replace(" ,",".")
                            cur_seg = cur_seg.replace(", ",".")
                            cur_seg = cur_seg.replace(",","")
                        elif len(entity_label_map.keys()) != 0:
                            search = False
                            if cur_seg.lower() in entity_label_map:
                                cur_seg = entity_label_map[cur_seg.lower()]     
                            else:
                                similarities = model.similarity([cur_seg.lower()], list(entity_label_map.keys()))  
                                merged_list = list(zip([v for _,v in entity_label_map.items()], similarities[0]))
                                sorted_list = sorted(merged_list, key=lambda x: x[1], reverse=True)[0]
                                if sorted_list[1] > 0.5:
                                    cur_seg = sorted_list[0]
                                else:       
                                    search = True                         
                        if search:
                            facc1_cand_entities = surface_index.get_indexrange_entity_el_pro_one_mention(cur_seg,top_k=50)
                            if facc1_cand_entities:
                                temp = []
                                for key in list(facc1_cand_entities.keys())[1:]:
                                    if facc1_cand_entities[key] >= 0.001:
                                        temp.append(key)
                                if len(temp) > 0:
                                    cur_seg = [list(facc1_cand_entities.keys())[0]]+temp
                                else:
                                    cur_seg = list(facc1_cand_entities.keys())[0]
                                
            segments.append(cur_seg)
            cur_seg = ''
        else:
            if prev_left_bracket:
                # in a bracket
                cur_seg = cur_seg + ' '+t
            else:
                if t=='(':
                    prev_left_par = True
                    segments.append(t)
                else:
                    if prev_left_par:
                        if t in ['ge', 'gt', 'le', 'lt']: # [ge, gt, le, lt] lowercase
                            segments.append(t)
                        else:                
                            segments.append(t.upper()) # [and, join, r, argmax, count] upper case
                        prev_left_par = False 
                    else:
                        if t != ')':
                            if t.lower() in entity_label_map:
                                t = entity_label_map[t.lower()]
                            else:
                                t = type_checker(t) # number
                        segments.append(t)

    combinations = [list(comb) for comb in itertools.product(*[item if isinstance(item, list) else [item] for item in segments])]
    
    exprs = [" ".join(s) for s in combinations]
                
    return exprs


def execute_normed_s_expr_from_label_maps(normed_expr, 
                                        entity_label_map,
                                        type_label_map,
                                        surface_index
                                        ):
    try:
        denorm_sexprs = denormalize_s_expr_new(normed_expr, 
                                        entity_label_map, 
                                        type_label_map,
                                        surface_index
                                        )
    except:
        return 'null', []
    
    query_exprs = [d.replace('( ','(').replace(' )', ')') for d in denorm_sexprs]
    for query_expr in query_exprs[:500]:
        try:
            # invalid sexprs, may leads to infinite loops
            if 'OR' in query_expr or 'WITH' in query_expr or 'PLUS' in query_expr:
                denotation = []
            else:
                sparql_query = lisp_to_sparql(query_expr)
                denotation = execute_query_with_odbc(sparql_query)
                denotation = [res.replace("http://rdf.freebase.com/ns/",'') for res in denotation]
                if len(denotation) == 0 :
                    
                    ents = set ()
                    
                    for item in sparql_query.replace('(', ' ( ').replace(')', ' ) ').split(' '):
                        if item.startswith("ns:m."):
                            ents.add(item)
                    addline = []
                    for i, ent in enumerate(list(ents)):
                        addline.append(f'{ent} rdfs:label ?en{i} . ')
                        addline.append(f'?ei{i} rdfs:label ?en{i} . ')
                        addline.append(f'FILTER (langMatches( lang(?en{i}), "EN" ) )')
                        sparql_query = sparql_query.replace(ent, f'?ei{i}')
                    clauses = sparql_query.split('\n')
                    for i, line in enumerate(clauses):
                        if line == "FILTER (!isLiteral(?x) OR lang(?x) = '' OR langMatches(lang(?x), 'en'))":
                            clauses = clauses[:i+1] + addline + clauses[i+1:]
                            break
                    sparql_query = '\n'.join(clauses)
                    denotation = execute_query_with_odbc(sparql_query)
                    denotation = [res.replace("http://rdf.freebase.com/ns/",'') for res in denotation]                    
        except:
            denotation = []
        if len(denotation) != 0 :
            break  
    if len(denotation) == 0 :
        query_expr = query_exprs[0]
    
    return query_expr, denotation


def execute_normed_s_expr_from_label_maps_rel(normed_expr, 
                                        entity_label_map,
                                        type_label_map,
                                        surface_index
                                        ):
    try:
        denorm_sexprs = denormalize_s_expr_new(normed_expr, 
                                        entity_label_map,
                                        type_label_map,
                                        surface_index
                                        )
    except:
        return 'null', []
    
    query_exprs = [d.replace('( ','(').replace(' )', ')') for d in denorm_sexprs]

    for d in denorm_sexprs[:5]:
        query_expr, denotation = try_relation(d)
        if len(denotation) != 0 :
            break          
        
    if len(denotation) == 0 :
        query_expr = query_exprs[0]
    
    return query_expr, denotation

def try_relation(d):
    ent_list = set()
    rel_list = set()
    denorm_sexpr = d.split(' ')
    for item in denorm_sexpr:
        if item.startswith('m.'):
            ent_list.add(item)
        elif '.' in item:
            rel_list.add(item)
    ent_list = list(ent_list)
    rel_list = list(rel_list)
    cand_rels = set()
    for ent in ent_list:
        in_rels, out_rels, _ = get_2hop_relations_with_odbc_wo_filter(ent)
        cand_rels = cand_rels | set(in_rels) | set(out_rels)
    cand_rels = list(cand_rels)
    if len(cand_rels) == 0:
        return d.replace('( ','(').replace(' )', ')'), []
    similarities = model.similarity(rel_list, cand_rels)
    change = dict()
    for i, rel in enumerate(rel_list):
        merged_list = list(zip(cand_rels, similarities[i]))
        sorted_list = sorted(merged_list, key=lambda x: x[1], reverse=True)
        change_rel = []
        for s in sorted_list:
            if s[1] > 0.01:
                change_rel.append(s[0])
        change[rel] = change_rel[:3]
    for i, item in enumerate(denorm_sexpr):
        if item in rel_list:
            denorm_sexpr[i] = change[item]
    combinations = [list(comb) for comb in itertools.product(*[item if isinstance(item, list) else [item] for item in denorm_sexpr])]
    exprs = [" ".join(s) for s in combinations][:30]
    query_exprs = [d.replace('( ','(').replace(' )', ')') for d in exprs]
    for query_expr in query_exprs:
        try:
            # invalid sexprs, may leads to infinite loops
            if 'OR' in query_expr or 'WITH' in query_expr or 'PLUS' in query_expr:
                denotation = []
            else:
                sparql_query = lisp_to_sparql(query_expr)
                denotation = execute_query_with_odbc(sparql_query)
                denotation = [res.replace("http://rdf.freebase.com/ns/",'') for res in denotation]
                if len(denotation) == 0 :
                    
                    ents = set ()
                    
                    for item in sparql_query.replace('(', ' ( ').replace(')', ' ) ').split(' '):
                        if item.startswith("ns:m."):
                            ents.add(item)
                    addline = []
                    for i, ent in enumerate(list(ents)):
                        addline.append(f'{ent} rdfs:label ?en{i} . ')
                        addline.append(f'?ei{i} rdfs:label ?en{i} . ')
                        addline.append(f'FILTER (langMatches( lang(?en{i}), "EN" ) )')
                        sparql_query = sparql_query.replace(ent, f'?ei{i}')
                    clauses = sparql_query.split('\n')
                    for i, line in enumerate(clauses):
                        if line == "FILTER (!isLiteral(?x) OR lang(?x) = '' OR langMatches(lang(?x), 'en'))":
                            clauses = clauses[:i+1] + addline + clauses[i+1:]
                            break
                    sparql_query = '\n'.join(clauses)
                    denotation = execute_query_with_odbc(sparql_query)
                    denotation = [res.replace("http://rdf.freebase.com/ns/",'') for res in denotation]
        except:
            denotation = []
        if len(denotation) != 0 :
            break              
    if len(denotation) == 0 :
        query_expr = query_exprs[0]      
    return query_expr, denotation  

def aggressive_top_k_eval_new(split, predict_file, dataset):
    """Run top k predictions, using linear origin map"""
    if dataset == "GrailQA":
        train_gen_dataset = load_json('data/GrailQA/generation/merged/GrailQA_train.json')
        test_gen_dataset = load_json('data/GrailQA/generation/merged/GrailQA_test.json')
        dev_gen_dataset = None
    
    predictions = load_json(predict_file)

    print(os.path.dirname(predict_file))
    dirname = os.path.dirname(predict_file)
    filename = os.path.basename(predict_file)

    if split=='dev':
        gen_dataset = dev_gen_dataset
    elif split=='train':
        gen_dataset = train_gen_dataset
    else:
        gen_dataset = test_gen_dataset

    if dataset == "GrailQA":
        train_type_map = load_json(f"data/GrailQA/generation/label_maps/GrailQA_train_type_label_map.json")
        train_type_map = {l.lower():t for t,l in train_type_map.items()}
    
    
    surface_index = surface_index_memory.EntitySurfaceIndexMemory(
        "data/common_data/facc1/entity_list_file_freebase_complete_all_mention", "data/common_data/facc1/surface_map_file_freebase_complete_all_mention",
        "data/common_data/facc1/freebase_complete_all_mention")
    
    top_hit = 0
    official_lines = dict()
    failed_preds = []

    gen_executable_cnt = 0
    final_executable_cnt = 0
    processed = 0
    for (pred,gen_feat) in tqdm(zip(predictions,gen_dataset), total=len(gen_dataset), desc=f'Evaluating {split}'):
        
        denormed_pred = []
        qid = gen_feat['ID']
            
        if args.golden_ent:
            entity_label_map = {v.lower(): k for k, v in list(gen_feat['gold_entity_map'].items())}
        else:
            entity_label_map = {}

        executable_index = None # index of LF being finally executed

        # find the first executable lf
        for rank, p in enumerate(pred['predictions']):
            lf, answers = execute_normed_s_expr_from_label_maps(
                                                p, 
                                                entity_label_map,
                                                train_type_map,
                                                surface_index)

            answers = [date_post_process(ans) for ans in list(answers)]
            
            denormed_pred.append(lf)

            
            if answers:
                executable_index = rank
                official_lines[qid]={
                    'logical_form':lf,
                    'answer':answers
                }
               
                if rank==0:
                    top_hit +=1
                break
            elif p.lower() ==gen_feat['normed_sexpr'].lower():
                print(p.lower())
                print(lf.lower())
                print(gen_feat['sexpr'].lower())

        
        if executable_index is not None:
            # found executable query from generated model
            gen_executable_cnt +=1
        else:
            denormed_pred = []
            
            # find the first executable lf
            for rank, p in enumerate(pred['predictions']):
                lf, answers = execute_normed_s_expr_from_label_maps_rel(
                                                    p, 
                                                    entity_label_map,
                                                    train_type_map,
                                                    surface_index)

                answers = [date_post_process(ans) for ans in list(answers)]
                
                denormed_pred.append(lf)
                
                if answers:
                    executable_index = rank
                    official_lines[qid]={
                        'logical_form':lf,
                        'answer':answers
                    }
                    if rank==0:
                        top_hit +=1
                    break
                    
            if executable_index is not None:
                # found executable query from generated model
                gen_executable_cnt +=1
                
            else:
            
                failed_preds.append({'qid':qid, 
                                'gt_sexpr': gen_feat['sexpr'], 
                                'gt_normed_sexpr': pred['gen_label'],
                                'pred': pred, 
                                'denormed_pred':denormed_pred})
        
            
        if executable_index is not None:
            final_executable_cnt+=1
        
        processed+=1
        if processed%100==0:
            print(f'Processed:{processed}, gen_executable_cnt:{gen_executable_cnt}')
        # if processed==5:
        #     break


    
    print('TOP 1 Executable', top_hit/ len(predictions))
    print('Gen Executable', gen_executable_cnt/ len(predictions))
    print('Final Executable', final_executable_cnt/ len(predictions))

    official_results_file = os.path.join(dirname,f'{filename}_output.json')
    dump_json(official_lines, official_results_file)

    # write failed predictions
    dump_json(failed_preds,os.path.join(dirname,f'{filename}_gen_failed_results.json'),indent=4)
    dump_json({
        'TOP 1 Executable': top_hit/ len(predictions),
        'Gen Executable': gen_executable_cnt/ len(predictions),
        'Final Executable': final_executable_cnt/ len(predictions)
    }, os.path.join(dirname,f'{filename}_statistics.json'),indent=4)

    # # evaluate
    # if dataset == "GrailQA":
    #     args.pred_file = result_file
    #     cwq_evaluate_valid_results(args)
    # else:
    #     args.pred_file = official_results_file
    #     webqsp_evaluate_valid_results(args)
        


if __name__=='__main__':
    """go down the top-k list to get the first executable locial form"""
    
    args = _parse_args()

    if args.server_ip and args.server_port:
        import ptvsd
        print("Waiting for debugger attach...",flush=True)
        ptvsd.enable_attach(address=(args.server_ip, args.server_port))
        ptvsd.wait_for_attach()

    if args.qid:
        pass
    else:
        if args.golden_ent:
            new_dir_path = os.path.join(os.path.dirname(args.pred_file),'golden_ent_predict')
            if not os.path.exists(new_dir_path):
                os.makedirs(new_dir_path)
            new_dir_name = os.path.join(new_dir_path,args.pred_file.split('/')[-1])
            shutil.copyfile(args.pred_file, new_dir_name)
            args.pred_file = new_dir_name
        aggressive_top_k_eval_new(args.split, args.pred_file, args.dataset)
        
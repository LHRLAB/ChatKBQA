import pickle
import json
import os
import shutil
import re
from typing import List
from executor.sparql_executor import get_label_with_odbc


def dump_to_bin(obj, fname):
    with open(fname, "wb") as f:
        pickle.dump(obj, f)


def load_bin(fname):
    with open(fname, "rb") as f:
        return pickle.load(f)


def load_json(fname, mode="r", encoding="utf8"):
    if "b" in mode:
        encoding = None
    with open(fname, mode=mode, encoding=encoding) as f:
        return json.load(f)


def dump_json(obj, fname, indent=4, mode='w' ,encoding="utf8", ensure_ascii=False):
    if "b" in mode:
        encoding = None
    with open(fname, "w", encoding=encoding) as f:
        return json.dump(obj, f, indent=indent, ensure_ascii=ensure_ascii)


def mkdir_f(prefix):
    if os.path.exists(prefix):
        shutil.rmtree(prefix)
    os.makedirs(prefix)


def mkdir_p(prefix):
    if not os.path.exists(prefix):
        os.makedirs(prefix)


illegal_xml_re = re.compile(u'[\x00-\x08\x0b-\x1f\x7f-\x84\x86-\x9f\ud800-\udfff\ufdd0-\ufddf\ufffe-\uffff]')
def clean_str(s: str) -> str:
    """remove illegal unicode characters"""
    return illegal_xml_re.sub('',s)



def tokenize_s_expr(expr):
    expr = expr.replace('(', ' ( ')
    expr = expr.replace(')', ' ) ')
    toks = expr.split(' ')
    toks = [x for x in toks if len(x)]
    return toks

def extract_mentioned_entities_from_sexpr(expr:str) -> List[str]:
    expr = expr.replace('(', ' ( ')
    expr = expr.replace(')', ' ) ')
    toks = expr.split(' ')
    toks = [x for x in toks if len(x)]
    entitiy_tokens = []
    for t in toks:
        # normalize entity
        if t.startswith('m.') or t.startswith('g.'):
            entitiy_tokens.append(t)
    return entitiy_tokens

def extract_mentioned_entities_from_sparql(sparql:str) -> List[str]:
    """extract entity from sparql"""
    sparql = sparql.replace('(',' ( ').replace(')',' ) ')
    toks = sparql.split(' ')
    toks = [x.replace('\t.','') for x in toks if len(x)]
    entity_tokens = []
    for t in toks:
        if t.startswith('ns:m.') or t.startswith('ns:g.'):
            entity_tokens.append(t[3:])
        
    entity_tokens = list(set(entity_tokens))
    return entity_tokens

def extract_mentioned_relations_from_sparql(sparql:str):
    """extract relation from sparql"""
    sparql = sparql.replace('(',' ( ').replace(')',' ) ')
    toks = sparql.split(' ')
    toks = [x for x in toks if len(x)]
    relation_tokens = []
    for t in toks:
        if (re.match("ns:[a-zA-Z_0-9]*\.[a-zA-Z_0-9]*\.[a-zA-Z_0-9]*",t.strip()) 
            or re.match("ns:[a-zA-Z_0-9]*\.[a-zA-Z_0-9]*\.[a-zA-Z_0-9]*\.[a-zA-Z_0-9]*",t.strip())):
            relation_tokens.append(t[3:])
    
    relation_tokens = list(set(relation_tokens))
    return relation_tokens


def extract_mentioned_relations_from_sexpr(sexpr:str)->List[str]:
    sexpr = sexpr.replace('(',' ( ').replace(')',' ) ')
    toks = sexpr.split(' ')
    toks = [x for x in toks if len(x)]
    relation_tokens = []

    for t in toks:
        if (re.match("[a-zA-Z_]*\.[a-zA-Z_]*\.[a-zA-z_]*",t.strip()) 
            or re.match("[a-zA-Z_]*\.[a-zA-Z_]*\.[a-zA-Z_]*\.[a-zA-Z_]*",t.strip())):
            relation_tokens.append(t)
    relation_tokens = list(set(relation_tokens))
    return relation_tokens

def vanilla_sexpr_linearization_method(expr, entity_label_map={}, relation_label_map={}, linear_origin_map={}):
    """
    textualize a logical form, replace mids with labels

    Returns:
        (str): normalized s_expr
    """
    expr = expr.replace("(", " ( ") # add space for parantheses
    expr = expr.replace(")", " ) ")
    toks = expr.split(" ") # split by space
    toks = [x for x in toks if len(x)]

    norm_toks = []
    for t in toks:

        # original token
        origin_t = t

        if t.startswith("m.") or t.startswith("g."): # replace entity with its name
            if t in entity_label_map:
                t = entity_label_map[t]
            else:
                # name = get_label(t)
                name = get_label_with_odbc(t)
                if name is not None:
                    entity_label_map[t] = name
                    t = name
            t = '[ '+t+' ]'
        elif "XMLSchema" in t: # remove xml type
            format_pos = t.find("^^")
            t = t[:format_pos]
        elif t == "ge": # replace ge/gt/le/lt
            t = "GREATER EQUAL"
        elif t == "gt":
            t = "GREATER THAN"
        elif t == "le":
            t = "LESS EQUAL"
        elif t == "lt":
            t = "LESS THAN"
        else:
            t = t.replace("_", " ") # replace "_" with " "
            t = t.replace(".", " , ") # replace "." with " , "
            
            if "." in origin_t: # relation
                t = "[ "+t+" ]"
                relation_label_map[origin_t]=t
        
        norm_toks.append(t)
        linear_origin_map[t] = origin_t # for reverse transduction
        
    return " ".join(norm_toks)

def _textualize_relation(r):
    """return a relation string with '_' and '.' replaced"""
    if "_" in r: # replace "_" with " "
        r = r.replace("_", " ")
    if "." in r: # replace "." with " , "
        r = r.replace(".", " , ")
    return r
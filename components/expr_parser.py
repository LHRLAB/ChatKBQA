"""
 Copyright (c) 2021, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""


from components.utils import *

def tokenize_s_expr(expr):
    expr = expr.replace('(', ' ( ')
    expr = expr.replace(')', ' ) ')
    toks = expr.split(' ')
    toks = [x for x in toks if len(x)]
    return toks

def extract_entities(expr):
    toks = tokenize_s_expr(expr)
    return [x for x in toks if x.startswith('m.')]


def extract_relations(expr):
    toks = tokenize_s_expr(expr)
    return [x for x in toks if ('.' in x) and (not x.startswith('m.')) and (not '^^' in x)]

class ASTNode:
    UNARY = 'unary'
    BINARY = 'binary'
    def __init__(self, construction, val, data_type, fields):
        self.construction = construction
        self.val = val
        # unary or binary
        self.data_type = data_type
        self.fields = fields

        # determined after construction
        self.depth = -1
        self.level = -1
    
    def assign_depth_and_level(self, level=0):
        self.level = level
        if self.fields:
            max_depth = max([x.assign_depth_and_level(level + 1) for x in self.fields])
            self.depth = max_depth + 1
        else:
            self.depth = 0
        return self.depth

    @classmethod
    def build(cls, tok, data_type, fields):
        if tok == 'AND':
            return AndNode(data_type, fields)
        elif tok == 'R':
            return RNode(data_type, fields)
        elif tok == 'COUNT':
            return CountNode(data_type, fields)
        elif tok == 'JOIN':
            return JoinNode(data_type, fields)
        elif tok in ['le', 'lt', 'ge', 'gt']:
            return CompNode(tok, data_type, fields)
        elif tok in ['ARGMIN', 'ARGMAX']:
            return ArgNode(tok, data_type, fields)
        elif tok.startswith('m.'):
            return EntityNode(tok, data_type, fields)
        elif '^^http://www.w3.org/2001/XMLSchema' in tok:
            return ValNode(tok, data_type, fields)
        else:
            return SchemaNode(tok, data_type, fields)
    
    def logical_form(self):
        if self.depth == 0:
            return self.val
        else:
            fields_str = [x.logical_form() for x in self.fields]
            return ' '.join(['(', self.val] + fields_str +  [')'])

    # nothing special. just fit legacy code input syle
    def compact_logical_form(self):
        lf = self.logical_form()
        return lf.replace('( ', '(').replace(' )', ')')

    def skeleton_form(self):
        if self.depth == 0:
            return self.construction
        else:
            fields_str = [x.skeleton_form() for x in self.fields]
            return ' '.join(['(', self.construction] + fields_str +  [')'])

    def logical_form_with_type(self):
        if self.depth == 0:
            return '{}[{}]'.format(self.val, self.data_type)
        else:
            fields_str = [x.logical_form_with_type() for x in self.fields]
            return ' '.join(['(', '{}[{}]'.format(self.val, self.data_type)] + fields_str +  [')'])

    def __str__(self):
        return self.logical_form()

    def __repr__(self):
        return self.logical_form()
    
    def textual_form_core(self):
        raise NotImplementedError('Textual form not implemented for abstract ast node')

    def textual_form(self):
        core_form = self.textual_form_core()
        if self.depth == 0 or self.level == 0:
            return core_form
        else:
            return '( ' + core_form + ' )'

class AndNode(ASTNode):
    def __init__(self, data_type, fields):
        super().__init__('AND', 'AND', data_type, fields)
    
    def textual_form_core(self):
        # the xxx that 
        if self.fields[0].depth == 0:
            return ' '.join([self.fields[0].textual_form() ,'that', self.fields[1].textual_form()])
        # xxx and xxxx
        else:
            return ' '.join([self.fields[0].textual_form() ,'and', self.fields[1].textual_form()]) 

class RNode(ASTNode):
    def __init__(self, data_type, fields):
        super().__init__('R', 'R', data_type, fields)

    def textual_form(self):
        # only surface relation is reserved
        assert self.depth == 1
        return self.textual_form_core()
    
    def textual_form_core(self):
        return self.fields[0].textual_form() + ' by'

class CountNode(ASTNode):
    def __init__(self, data_type, fields):
        super().__init__('COUNT', 'COUNT', data_type, fields)

    def textual_form_core(self):
        return 'how many ' + self.fields[0].textual_form()

class JoinNode(ASTNode):
    def __init__(self, data_type, fields):
        super().__init__('JOIN', 'JOIN', data_type, fields)

    def textual_form_core(self):
        return ' '.join([self.fields[0].textual_form(), self.fields[1].textual_form()])

# argmin argmax
class ArgNode(ASTNode):
    def __init__(self, val, data_type, fields):
        super().__init__('ARG', val, data_type, fields)

    def textual_form_core(self):
        prompt = 'with most' if self.val == 'ARGMAX' else 'with least'
        return ' '.join([self.fields[0].textual_form(), prompt, self.fields[1].textual_form()])

# lt le gt ge
class CompNode(ASTNode):
    PROMPT_DICT = {
        'gt': 'greater than',
        'ge': 'greater equal',
        'lt': 'less than',
        'le': 'less equal',
    }

    def __init__(self, val, data_type, fields):
        super().__init__('COMP', val, data_type, fields)
    
    def textual_form_core(self):
        prompt = CompNode.PROMPT_DICT[self.val]
        return ' '.join([self.fields[0].textual_form(), prompt, self.fields[1].textual_form()])

class EntityNode(ASTNode):
    def __init__(self, val, data_type, fields):
        super().__init__('ENTITY', val, data_type, fields)

    def textual_form_core(self):
        return self.val

class SchemaNode(ASTNode):
    def __init__(self, val, data_type, fields):
        super().__init__('SCHEMA', val, data_type, fields)

    def textual_form_core(self):
        return self.val

class ValNode(ASTNode):
    def __init__(self, val, data_type, fields):
        super().__init__('VAL', val, data_type, fields)

    def textual_form_core(self):
        return self.val

def _consume_a_node(tokens, cursor, data_type):
    is_root = cursor == 0
    cur_tok = tokens[cursor]
    cursor += 1
    if cur_tok == '(':
        node, cursor = _consume_a_node(tokens, cursor, data_type)
        assert tokens[cursor] == ')'
        cursor += 1
    elif cur_tok == 'AND':
        # left, right, all unary
        left, cursor = _consume_a_node(tokens, cursor, ASTNode.UNARY)
        right, cursor = _consume_a_node(tokens, cursor, ASTNode.UNARY)
        node = ASTNode.build(cur_tok, data_type, [left, right])
    elif cur_tok == 'JOIN':
        # if cur is unary, right unary, else right binary
        left, cursor = _consume_a_node(tokens, cursor, ASTNode.BINARY)
        right, cursor = _consume_a_node(tokens, cursor, data_type)
        node = ASTNode.build(cur_tok, data_type, [left, right])
    elif cur_tok == 'ARGMIN' or cur_tok == 'ARGMAX':
        left, cursor = _consume_a_node(tokens, cursor, ASTNode.UNARY)
        right, cursor = _consume_a_node(tokens, cursor, ASTNode.BINARY)
        node = ASTNode.build(cur_tok, data_type, [left, right])
    elif cur_tok == 'le' or cur_tok == 'lt' or cur_tok == 'ge' or cur_tok == 'gt':
        left, cursor = _consume_a_node(tokens, cursor, ASTNode.UNARY)
        right, cursor = _consume_a_node(tokens, cursor, ASTNode.UNARY)
        node = ASTNode.build(cur_tok, data_type, [left, right])
    elif cur_tok == 'R':
        child, cursor = _consume_a_node(tokens, cursor, ASTNode.BINARY)
        node =  ASTNode.build(cur_tok, data_type, [child])
    elif cur_tok == 'COUNT':
        child, cursor = _consume_a_node(tokens, cursor, ASTNode.UNARY)
        node =  ASTNode.build(cur_tok, data_type, [child])
    else:
        # symbol
        # class relation
        # value
        # entity
        node =  ASTNode.build(cur_tok, ASTNode.UNARY, [])
    if is_root:
        node.assign_depth_and_level()

    return node, cursor

# top lvel: and, arg, count, cant be JOIN, LE, R
def parse_s_expr(expr):
    tokens = tokenize_s_expr(expr)
    # assert tokens[0] == '(' and tokens[-1] == ')'
    # tokens = tokens[1:-1]
    ast, cursor = _consume_a_node(tokens, 0, 'unary')
    assert cursor == len(tokens)
    assert ' '.join(tokens) == ast.logical_form()
    return ast   

def textualize_s_expr(expr):
    ast = parse_s_expr(expr)
    return ast.textual_form()
    # print(ast.logical_form_with_type())

def simplify_textual_form(expr):
    toks = expr.split(' ')

    norm_toks = []
    for t in toks:
        # normalize entity
        if t.startswith('m.'):
            pass
        elif 'XMLSchema' in t:
            pass
        elif '.' in t:
            meta_relations = t = t.split('.')
            t = meta_relations[-1]
            if '.' in t:
                t = t.replace('.', ' , ')
            if '_' in t:
                t = t.replace('.', ' , ')
        # normalize type
        norm_toks.append(t)
    return ' '.join(norm_toks)

def test_text_converter():
    # dataset = load_json('outputs/grailqa_v1.0_dev.json')
    dataset = load_json('outputs/grailqa_v1.0_train.json')
    import random
    random.seed(123)
    random.shuffle(dataset)

    templates = set()
    for i, data in enumerate(dataset[:100]):
        s_expr = data['s_expression']
        question = data['question']
        ast = parse_s_expr(data['s_expression'])
        skeleton = ast.skeleton_form()
        templates.add(skeleton)
        textual_expr = ast.textual_form()

        simplified_expr = simplify_textual_form(textual_expr)
        # if ('AND' in skeleton):
        #     sim_skeleton = skeleton.replace('AND SCHEMA', '')
        #     if 'AND' in sim_skeleton:
        #         print('------------------------------')
        #         print(question)
        #         print(s_expr)
        # if ('COMP' in skeleton):
        print(f'----------------{i}--------------')
        print(question)
        print(s_expr)
        print(textual_expr)
        print(simplified_expr)

    # print(len(templates))
    # for t in templates:
    #     if 'COMP' in t:
    #         print(t)
    #     # if 'ARG' in t:
    #     #     print(t)


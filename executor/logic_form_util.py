import networkx as nx
from typing import List, Union
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from executor.sparql_executor import execute_query
import re
import json


REVERSE = True  # if REVERSE, then reverse relations are also taken into account for semantic EM

path = str(Path(__file__).parent.absolute())

reverse_properties = {}
with open(path + '/../ontology/reverse_properties', 'r') as f:
    for line in f:
        reverse_properties[line.split('\t')[0]] = line.split('\t')[1].replace('\n', '')

with open(path + '/../ontology/fb_roles', 'r') as f:
    content = f.readlines()

relation_dr = {}
relations = set()
for line in content:
    fields = line.split()
    relation_dr[fields[1]] = (fields[0], fields[2])
    relations.add(fields[1])

with open(path + '/../ontology/fb_types', 'r') as f:
    content = f.readlines()

upper_types = defaultdict(lambda: set())

types = set()
for line in content:
    fields = line.split()
    upper_types[fields[0]].add(fields[2])
    types.add(fields[0])
    types.add(fields[2])

function_map = {'le': '<=', 'ge': '>=', 'lt': '<', 'gt': '>'}

def lisp_to_nested_expression(lisp_string):
    """
    Takes a logical form as a lisp string and returns a nested list representation of the lisp.
    For example, "(count (division first))" would get mapped to ['count', ['division', 'first']].
    """
    stack: List = []
    current_expression: List = []
    tokens = lisp_string.split()
    for token in tokens:
        while token[0] == '(':
            nested_expression: List = []
            current_expression.append(nested_expression)
            stack.append(current_expression)
            current_expression = nested_expression
            token = token[1:]
        current_expression.append(token.replace(')', ''))
        while token[-1] == ')':
            current_expression = stack.pop()
            token = token[:-1]
    return current_expression[0]

def get_symbol_type(symbol: str) -> int:
    if symbol.__contains__('^^'):
        return 2
    elif symbol in types:
        return 3
    elif symbol in relations:
        return 4
    elif symbol:
        return 1


def same_logical_form(form1: str, form2: str) -> bool:
    if form1.__contains__("@@UNKNOWN@@") or form2.__contains__("@@UNKNOWN@@"):
        return False
    try:
        G1 = logical_form_to_graph(lisp_to_nested_expression(form1))
    except Exception:
        return False
    try:
        G2 = logical_form_to_graph(lisp_to_nested_expression(form2))
    except Exception:
        return False

    def node_match(n1, n2):
        if n1['id'] == n2['id'] and n1['type'] == n2['type']:
            func1 = n1.pop('function', 'none')
            func2 = n2.pop('function', 'none')
            tc1 = n1.pop('tc', 'none')
            tc2 = n2.pop('tc', 'none')

            if func1 == func2 and tc1 == tc2:
                return True
            else:
                return False
            # if 'function' in n1 and 'function' in n2 and n1['function'] == n2['function']:
            #     return True
            # elif 'function' not in n1 and 'function' not in n2:
            #     return True
            # else:
            #     return False
        else:
            return False

    def multi_edge_match(e1, e2):
        if len(e1) != len(e2):
            return False
        values1 = []
        values2 = []
        for v in e1.values():
            values1.append(v['relation'])
        for v in e2.values():
            values2.append(v['relation'])
        return sorted(values1) == sorted(values2)

    return nx.is_isomorphic(G1, G2, node_match=node_match, edge_match=multi_edge_match)


def logical_form_to_graph(expression: List) -> nx.MultiGraph:
    G = _get_graph(expression)
    G.nodes[len(G.nodes())]['question_node'] = 1
    return G


def _get_graph(
        expression: List) -> nx.MultiGraph:  # The id of question node is always the same as the size of the graph
    if isinstance(expression, str):
        G = nx.MultiDiGraph()
        if get_symbol_type(expression) == 1:
            G.add_node(1, id=expression, type='entity')
        elif get_symbol_type(expression) == 2:
            G.add_node(1, id=expression, type='literal')
        elif get_symbol_type(expression) == 3:
            G.add_node(1, id=expression, type='class')
            # G.add_node(1, id="common.topic", type='class')
        elif get_symbol_type(expression) == 4:  # relation or attribute
            domain, rang = relation_dr[expression]
            G.add_node(1, id=rang, type='class')  # if it's an attribute, the type will be changed to literal in arg
            G.add_node(2, id=domain, type='class')
            G.add_edge(2, 1, relation=expression)

            if REVERSE:
                if expression in reverse_properties:
                    G.add_edge(1, 2, relation=reverse_properties[expression])

        return G

    if expression[0] == 'R':
        G = _get_graph(expression[1])
        size = len(G.nodes())
        mapping = {}
        for n in G.nodes():
            mapping[n] = size - n + 1
        G = nx.relabel_nodes(G, mapping)
        return G

    elif expression[0] in ['JOIN', 'le', 'ge', 'lt', 'gt']:
        G1 = _get_graph(expression=expression[1])
        G2 = _get_graph(expression=expression[2])

        size = len(G2.nodes())
        qn_id = size
        if G1.nodes[1]['type'] == G2.nodes[qn_id]['type'] == 'class':
            if G2.nodes[qn_id]['id'] in upper_types[G1.nodes[1]['id']]:
                G2.nodes[qn_id]['id'] = G1.nodes[1]['id']
            # G2.nodes[qn_id]['id'] = G1.nodes[1]['id']
        mapping = {}
        for n in G1.nodes():
            mapping[n] = n + size - 1
        G1 = nx.relabel_nodes(G1, mapping)
        G = nx.compose(G1, G2)

        if expression[0] != 'JOIN':
            G.nodes[1]['function'] = function_map[expression[0]]

        return G

    elif expression[0] == 'AND':
        G1 = _get_graph(expression[1])
        G2 = _get_graph(expression[2])

        size1 = len(G1.nodes())
        size2 = len(G2.nodes())
        if G1.nodes[size1]['type'] == G2.nodes[size2]['type'] == 'class':
            # if G2.nodes[size2]['id'] in upper_types[G1.nodes[size1]['id']]:
            G2.nodes[size2]['id'] = G1.nodes[size1]['id']
            # IIRC, in nx.compose, for the same node, its information can be overwritten by its info in the second graph
            # So here for the AND function we force it to choose the type explicitly provided in the logical form
        mapping = {}
        for n in G1.nodes():
            mapping[n] = n + size2 - 1
        G1 = nx.relabel_nodes(G1, mapping)
        G2 = nx.relabel_nodes(G2, {size2: size1 + size2 - 1})
        G = nx.compose(G1, G2)

        return G

    elif expression[0] == 'COUNT':
        G = _get_graph(expression[1])
        size = len(G.nodes())
        G.nodes[size]['function'] = 'count'

        return G

    elif expression[0].__contains__('ARG'):
        G1 = _get_graph(expression[1])
        size1 = len(G1.nodes())
        G2 = _get_graph(expression[2])
        size2 = len(G2.nodes())
        # G2.nodes[1]['class'] = G2.nodes[1]['id']   # not sure whether this is needed for sparql
        G2.nodes[1]['id'] = 0
        G2.nodes[1]['type'] = 'literal'
        G2.nodes[1]['function'] = expression[0].lower()
        if G1.nodes[size1]['type'] == G2.nodes[size2]['type'] == 'class':
            # if G2.nodes[size2]['id'] in upper_types[G1.nodes[size1]['id']]:
            G2.nodes[size2]['id'] = G1.nodes[size1]['id']

        mapping = {}
        for n in G1.nodes():
            mapping[n] = n + size2 - 1
        G1 = nx.relabel_nodes(G1, mapping)
        G2 = nx.relabel_nodes(G2, {size2: size1 + size2 - 1})
        G = nx.compose(G1, G2)

        return G

    elif expression[0] == 'TC':
        G = _get_graph(expression[1])
        size = len(G.nodes())
        G.nodes[size]['tc'] = (expression[2], expression[3])

        return G


def graph_to_logical_form(G, start, count: bool = False):
    if count:
        return '(COUNT ' + none_function(G, start) + ')'
    else:
        return none_function(G, start)


def get_end_num(G, s):
    end_num = defaultdict(lambda: 0)
    for edge in list(G.edges(s)):  # for directed graph G.edges is the same as G.out_edges, not including G.in_edges
        end_num[list(edge)[1]] += 1
    return end_num


def set_visited(G, s, e, relation):
    end_num = get_end_num(G, s)
    for i in range(0, end_num[e]):
        if G.edges[s, e, i]['relation'] == relation:
            G.edges[s, e, i]['visited'] = True


def binary_nesting(function: str, elements: List[str], types_along_path=None) -> str:
    if len(elements) < 2:
        print("error: binary function should have 2 parameters!")
    if not types_along_path:
        if len(elements) == 2:
            return '(' + function + ' ' + elements[0] + ' ' + elements[1] + ')'
        else:
            return '(' + function + ' ' + elements[0] + ' ' + binary_nesting(function, elements[1:]) + ')'
    else:
        if len(elements) == 2:
            return '(' + function + ' ' + types_along_path[0] + ' ' + elements[0] + ' ' + elements[1] + ')'
        else:
            return '(' + function + ' ' + types_along_path[0] + ' ' + elements[0] + ' ' \
                   + binary_nesting(function, elements[1:], types_along_path[1:]) + ')'


def count_function(G, start):
    return '(COUNT ' + none_function(G, start) + ')'


def none_function(G, start, arg_node=None, type_constraint=True):
    if arg_node is not None:
        arg = G.nodes[arg_node]['function']
        path = list(nx.all_simple_paths(G, start, arg_node))
        assert len(path) == 1
        arg_clause = []
        for i in range(0, len(path[0]) - 1):
            edge = G.edges[path[0][i], path[0][i + 1], 0]
            if edge['reverse']:
                relation = '(R ' + edge['relation'] + ')'
            else:
                relation = edge['relation']
            arg_clause.append(relation)

        # Deleting edges until the first node with out degree > 2 is meet
        # (conceptually it should be 1, but remember that add edges is both directions)
        while i >= 0:
            flag = False
            if G.out_degree[path[0][i]] > 2:
                flag = True
            G.remove_edge(path[0][i], path[0][i + 1], 0)
            i -= 1
            if flag:
                break

        if len(arg_clause) > 1:
            arg_clause = binary_nesting(function='JOIN', elements=arg_clause)
            # arg_clause = ' '.join(arg_clause)
        else:
            arg_clause = arg_clause[0]

        return '(' + arg.upper() + ' ' + none_function(G, start) + ' ' + arg_clause + ')'

    # arg = -1
    # for nei in G[start]:
    #     if G.nodes[nei]['function'].__contains__('arg'):
    #         arg = nei
    #         arg_function = G.nodes[nei]['function']
    # if arg != -1:
    #     edge = G.edges[start, arg, 0]
    #     if edge['reverse']:
    #         relation = '(R ' + edge['relation'] + ')'
    #     else:
    #         relation = edge['relation']
    #     G.remove_edge(start, arg, 0)
    #     return '(' + arg_function.upper() + ' ' + none_function(G, start) + ' ' + relation + ')'

    if G.nodes[start]['type'] != 'class':
        return G.nodes[start]['id']

    end_num = get_end_num(G, start)
    clauses = []

    if G.nodes[start]['question'] and type_constraint:
        clauses.append(G.nodes[start]['id'])
    for key in end_num.keys():
        for i in range(0, end_num[key]):
            if not G.edges[start, key, i]['visited']:
                relation = G.edges[start, key, i]['relation']
                G.edges[start, key, i]['visited'] = True
                set_visited(G, key, start, relation)
                if G.edges[start, key, i]['reverse']:
                    relation = '(R ' + relation + ')'
                if G.nodes[key]['function'].__contains__('<') or G.nodes[key]['function'].__contains__('>'):
                    if G.nodes[key]['function'] == '>':
                        clauses.append('(gt ' + relation + ' ' + none_function(G, key) + ')')
                    if G.nodes[key]['function'] == '>=':
                        clauses.append('(ge ' + relation + ' ' + none_function(G, key) + ')')
                    if G.nodes[key]['function'] == '<':
                        clauses.append('(lt ' + relation + ' ' + none_function(G, key) + ')')
                    if G.nodes[key]['function'] == '<=':
                        clauses.append('(le ' + relation + ' ' + none_function(G, key) + ')')
                else:
                    clauses.append('(JOIN ' + relation + ' ' + none_function(G, key) + ')')

    if len(clauses) == 0:
        return G.nodes[start]['id']

    if len(clauses) == 1:
        return clauses[0]
    else:
        return binary_nesting(function='AND', elements=clauses)


def get_lisp_from_graph_query(graph_query):
    G = nx.MultiDiGraph()
    aggregation = 'none'
    arg_node = None
    for node in graph_query['nodes']:
        #         G.add_node(node['nid'], id=node['id'].replace('.', '/'), type=node['node_type'], question=node['question_node'], function=node['function'])
        G.add_node(node['nid'], id=node['id'], type=node['node_type'], question=node['question_node'],
                   function=node['function'], cla=node['class'])
        if node['question_node'] == 1:
            qid = node['nid']
        if node['function'] != 'none':
            aggregation = node['function']
            if node['function'].__contains__('arg'):
                arg_node = node['nid']
    for edge in graph_query['edges']:
        G.add_edge(edge['start'], edge['end'], relation=edge['relation'], reverse=False, visited=False)
        G.add_edge(edge['end'], edge['start'], relation=edge['relation'], reverse=True, visited=False)
    if 'count' == aggregation:
        # print(count_function(G, qid))
        return count_function(G, qid)
    else:
        # print(none_function(G, qid))
        return none_function(G, qid, arg_node=arg_node)


def lisp_to_sparql(lisp_program: str):
    clauses = []
    order_clauses = []
    entities = set()  # collect entites for filtering
    # identical_variables = {}   # key should be smaller than value, we will use small variable to replace large variable
    identical_variables_r = {}  # key should be larger than value
    expression = lisp_to_nested_expression(lisp_program)
    superlative = False
    # check SUPERLATIVE
    if expression[0] in ['ARGMAX', 'ARGMIN']:
        superlative = True
        # remove all joins in relation chain of an arg function. In another word, we will not use arg function as
        # binary function here, instead, the arity depends on the number of relations in the second argument in the
        # original function
        if isinstance(expression[2], list): # n-hop relations
            # TODO: in denormalization, ARGMAX and JOIN may wrongly concat two relations to one
            def retrieve_relations(exp: list):
                rtn = []
                for element in exp:
                    if element == 'JOIN':
                        continue
                    elif isinstance(element, str):
                        rtn.append(element)
                    elif isinstance(element, list) and element[0] == 'R':
                        rtn.append(element)
                    elif isinstance(element, list) and element[0] == 'JOIN':
                        rtn.extend(retrieve_relations(element))
                return rtn

            relations = retrieve_relations(expression[2])
            expression = expression[:2]
            expression.extend(relations)

    sub_programs = _linearize_lisp_expression(expression, [0])
    question_var = len(sub_programs) - 1 # get the question_var (last sub_formula_id)
    count = False

    def get_root(var: int):
        while var in identical_variables_r:
            var = identical_variables_r[var]

        return var

    for i, subp in enumerate(sub_programs):
        i = str(i)
        if subp[0] == 'JOIN':
            if isinstance(subp[1], list):  # R relation
                if subp[2][:2] in ["m.", "g."]:  # entity
                    clauses.append("ns:" + subp[2] + " ns:" + subp[1][1] + " ?x" + i + " .")
                    entities.add(subp[2])
                elif subp[2][0] == '#':  # variable
                    clauses.append("?x" + subp[2][1:] + " ns:" + subp[1][1] + " ?x" + i + " .")
                else:  # literal   (actually I think literal can only be object)
                    if subp[2].__contains__('^^'):
                        data_type = subp[2].split("^^")[1].split("#")[1]
                        if data_type not in ['integer', 'float', 'dateTime']:
                            subp[2] = f'"{subp[2].split("^^")[0] + "-08:00"}"^^<{subp[2].split("^^")[1]}>'
                            # subp[2] = subp[2].split("^^")[0] + '-08:00^^' + subp[2].split("^^")[1]
                        else:
                            subp[2] = f'"{subp[2].split("^^")[0]}"^^<{subp[2].split("^^")[1]}>'
                    clauses.append(subp[2] + " ns:" + subp[1][1] + " ?x" + i + " .")
            else:
                if subp[2][:2] in ["m.", "g."]:  # entity
                    clauses.append("?x" + i + " ns:" + subp[1] + " ns:" + subp[2] + " .")
                    entities.add(subp[2])
                elif subp[2][0] == '#':  # variable
                    clauses.append("?x" + i + " ns:" + subp[1] + " ?x" + subp[2][1:] + " .")
                else:  # literal or  2 hop relation (JOIN r1 r2) 
                    if re.match(r'[\w_]*\.[\w_]*\.[\w_]*',subp[2]):
                        # 2-hop relation
                        pass
                    else:
                        # literal or number or type
                        if subp[2].__contains__('^^'): # literal with datatype
                            data_type_string = subp[2].split("^^")[1]
                            if '#' in data_type_string:
                                data_type = data_type_string.split("#")[1]
                            elif 'xsd:' in data_type_string:
                                data_type = data_type_string.split('xsd:')[1]
                            else:
                                data_type = 'dateTime'    
                            if data_type not in ['integer', 'float', 'dateTime','date']:
                                subp[2] = f'"{subp[2].split("^^")[0] + "-08:00"}"^^<{subp[2].split("^^")[1]}>'
                            else:
                                subp[2] = f'"{subp[2].split("^^")[0]}"^^<{subp[2].split("^^")[1]}>'
                        elif re.match("[a-zA-Z_]*\.[a-zA-Z_]*",subp[2]): # type e.g. education.university
                            subp[2]='ns:'+subp[2]
                        elif len(subp)>3: # error splitting, e.g. "2100 Woodward Avenue"@en
                            subp[2]=" ".join(subp[2:])

                        clauses.append("?x" + i + " ns:" + subp[1] + " " + subp[2] + " .")
        elif subp[0] == 'AND':
            var1 = int(subp[2][1:])
            rooti = get_root(int(i))
            root1 = get_root(var1)
            if rooti > root1:
                identical_variables_r[rooti] = root1
            else:
                identical_variables_r[root1] = rooti
                root1 = rooti
            # identical_variables[var1] = int(i)
            if subp[1][0] == "#":
                var2 = int(subp[1][1:])
                root2 = get_root(var2)
                # identical_variables[var2] = int(i)
                if root1 > root2:
                    # identical_variables[var2] = var1
                    identical_variables_r[root1] = root2
                else:
                    # identical_variables[var1] = var2
                    identical_variables_r[root2] = root1
            else:  # 2nd argument is a class
                clauses.append("?x" + i + " ns:type.object.type ns:" + subp[1] + " .")
        elif subp[0] in ['le', 'lt', 'ge', 'gt']:  # the 2nd can only be numerical value
            if subp[1].startswith('#'): # 2-hop constraint
                line_num = int(subp[1].replace('#',''))
                first_relation = sub_programs[line_num][1]
                second_relation = sub_programs[line_num][2]
                
                if isinstance(first_relation,list): # first relation is reversed
                    clauses.append("?cvt" + " ns:" + first_relation[1] + " ?x"+i+" .")
                else:
                    clauses.append("?x"+i + " ns:"+ first_relation+ " ?cvt .")
                
                if isinstance(second_relation,list): #second relation is reversed
                    clauses.append("?y"+ i + " ns:"+ second_relation[1] + " ?cvt .")
                else:
                    clauses.append("?cvt"+ " ns:" + second_relation+ " ?y"+i +" .")
            else:
                clauses.append("?x" + i + " ns:" + subp[1] + " ?y" + i + " .")
            if subp[0] == 'le':
                op = "<="
            elif subp[0] == 'lt':
                op = "<"
            elif subp[0] == 'ge':
                op = ">="
            else:
                op = ">"
            if subp[2].__contains__('^^'):
                data_type = subp[2].split("^^")[1].split("#")[1]
                if data_type not in ['integer', 'float', 'dateTime']:
                    subp[2] = f'"{subp[2].split("^^")[0] + "-08:00"}"^^<{subp[2].split("^^")[1]}>'
                else:
                    subp[2] = f'"{subp[2].split("^^")[0]}"^^<{subp[2].split("^^")[1]}>'

            if re.match(r'\d+', subp[2]) or re.match(r'"\d+"^^xsd:integer', subp[2]): # integer
                clauses.append(f"FILTER (xsd:integer(?y{i}) {op} {subp[2]})")
            else: # others
                clauses.append(f"FILTER (?y{i} {op} {subp[2]})")
        elif subp[0] == 'TC':
            var = int(subp[1][1:])
            # identical_variables[var] = int(i)
            rooti = get_root(int(i))
            root_var = get_root(var)
            if rooti > root_var:
                identical_variables_r[rooti] = root_var
            else:
                identical_variables_r[root_var] = rooti

            year = subp[3]
            if year == 'NOW' or year == 'now':
                from_para = '"2015-08-10"^^xsd:dateTime'
                to_para = '"2015-08-10"^^xsd:dateTime'
            else:
                if "^^" in year:
                    year = year.split("^^")[0]
                from_para = f'"{year}-12-31"^^xsd:dateTime'
                to_para = f'"{year}-01-01"^^xsd:dateTime'

            # get the last relation token
            rel_from_property = subp[2].split('.')[-1]
            if rel_from_property == 'from':
                rel_to_property = 'to'
            elif rel_from_property =='end_date':
                # swap end_date and start_date
                subp[2] = subp[2].replace('end_date','start_date')
                rel_from_property = 'start_date'
                rel_to_property = 'end_date'
            else: # from_date -> to_date
                rel_to_property = 'to_date'
            opposite_rel = subp[2].replace(rel_from_property,rel_to_property)

            # add <= constraint
            clauses.append(f'FILTER(NOT EXISTS {{?x{i} ns:{subp[2]} ?sk0}} || ')
            clauses.append(f'EXISTS {{?x{i} ns:{subp[2]} ?sk1 . ')
            clauses.append(f'FILTER(xsd:datetime(?sk1) <= {from_para}) }})')
            
            # add >= constraint
            clauses.append(f'FILTER(NOT EXISTS {{?x{i} ns:{opposite_rel} ?sk2}} || ')
            clauses.append(f'EXISTS {{?x{i} ns:{opposite_rel} ?sk3 . ')
            clauses.append(f'FILTER(xsd:datetime(?sk3) >= {to_para}) }})')

            # if subp[2][-4:] == "from":
            #     clauses.append(f'FILTER(NOT EXISTS {{?x{i} ns:{subp[2][:-4] + "to"} ?sk2}} || ')
            #     clauses.append(f'EXISTS {{?x{i} ns:{subp[2][:-4] + "to"} ?sk3 . ')
            # elif subp[2][-8:] =='end_date': # end_date -> start_date
            #     clauses.append(f'FILTER(NOT EXISTS {{?x{i} ns:{subp[2][:-8] + "start_date"} ?sk2}} || ')
            #     clauses.append(f'EXISTS {{?x{i} ns:{subp[2][:-8] + "start_date"} ?sk3 . ')
            # else:  # from_date -> to_date
            #     clauses.append(f'FILTER(NOT EXISTS {{?x{i} ns:{subp[2][:-9] + "to_date"} ?sk2}} || ')
            #     clauses.append(f'EXISTS {{?x{i} ns:{subp[2][:-9] + "to_date"} ?sk3 . ')

            

        elif subp[0] in ["ARGMIN", "ARGMAX"]:
            superlative = True
            if subp[1][0] == '#':
                var = int(subp[1][1:])
                rooti = get_root(int(i))
                root_var = get_root(var)
                # identical_variables[var] = int(i)
                if rooti > root_var:
                    identical_variables_r[rooti] = root_var
                else:
                    identical_variables_r[root_var] = rooti
            else:  # arg1 is class
                clauses.append(f'?x{i} ns:type.object.type ns:{subp[1]} .')

            if len(subp) == 3: # 1-hop relations
                clauses.append(f'?x{i} ns:{subp[2]} ?arg0 .')
            elif len(subp) > 3: # multi-hop relations, containing cvt
                for j, relation in enumerate(subp[2:-1]):
                    if j == 0:
                        var0 = f'x{i}'
                    else:
                        var0 = f'c{j - 1}'
                    var1 = f'c{j}'
                    if isinstance(relation, list) and relation[0] == 'R':
                        clauses.append(f'?{var1} ns:{relation[1]} ?{var0} .')
                    else:
                        clauses.append(f'?{var0} ns:{relation} ?{var1} .')

                clauses.append(f'?c{j} ns:{subp[-1]} ?arg0 .')

            if subp[0] == 'ARGMIN':
                order_clauses.append("ORDER BY ?arg0")
            elif subp[0] == 'ARGMAX':
                order_clauses.append("ORDER BY DESC(?arg0)")
            order_clauses.append("LIMIT 1")


        elif subp[0] == 'COUNT':  # this is easy, since it can only be applied to the quesiton node
            var = int(subp[1][1:])
            root_var = get_root(var)
            identical_variables_r[int(i)] = root_var  # COUNT can only be the outtermost
            count = True
    #  Merge identical variables
    for i in range(len(clauses)):
        for k in identical_variables_r:
            clauses[i] = clauses[i].replace(f'?x{k} ', f'?x{get_root(k)} ')

    question_var = get_root(question_var)

    for i in range(len(clauses)):
        clauses[i] = clauses[i].replace(f'?x{question_var} ', f'?x ')

    if superlative:
        arg_clauses = clauses[:]

    # add entity filters




    # add variable filters
    filter_variables = []
    for clause in clauses:
        variables = re.findall(r"\?\w*",clause)
        if variables:
            for var in variables:
                var = var.strip()
                if var not in filter_variables and var != '?x' and not var.startswith('?sk'):
                    filter_variables.append(var)
                    
    
    for entity in entities:
        clauses.append(f'FILTER (?x != ns:{entity})')

    for var in filter_variables:
        clauses.append(f"FILTER (?x != {var})")
        
    num = 0
    sentences = [s for s in clauses]
    for c , sentence in enumerate(sentences):
        if len(sentence.split(' '))==4 and sentence.split(' ')[-1]=='.':
            if sentence.split(' ')[-2].startswith('"') and sentence.split(' ')[-2].endswith('"'):
                name = sentence.split(' ')[-2]
                clauses[c] = clauses[c].replace(name,f'?st{num}')
                clauses.append(f"FILTER (SUBSTR(STR(?st{num}), 1, STRLEN({name})) = {name})")
                num += 1
            elif sentence.split(' ')[-2].endswith('"^^<http://www.w3.org/2001/XMLSchema#dateTime>'):
                name = sentence.split(' ')[-2].replace("^^<http://www.w3.org/2001/XMLSchema#dateTime>","")
                clauses[c] = clauses[c].replace(sentence.split(' ')[-2],f'?st{num}')
                clauses.append(f"FILTER (SUBSTR(STR(?st{num}), 1, STRLEN({name})) = {name})")
                num += 1
    
    
    # num = 0        
    # sentences = [s for s in clauses]
    # for c , sentence in enumerate(sentences):
    #     sentence = sentence.replace('(',' ( ').replace(')',' ) ')
    #     for sent in sentence.split(' '):
    #         if '"' in sent:
    #             name = re.findall(r'(".*?")', sent)[0]
    #             clauses[c] = clauses[c].replace(sent,f'?st{num}')
    #             clauses.append(f"FILTER (SUBSTR(STR(?st{num}), 1, STRLEN({name})) = {name})")
    #             num += 1

    clauses.insert(0,
                   f"FILTER (!isLiteral(?x) OR lang(?x) = '' OR langMatches(lang(?x), 'en'))")
    clauses.insert(0, "WHERE {")
    if count:
        clauses.insert(0, f"SELECT COUNT DISTINCT ?x")
    elif superlative:
        # clauses.insert(0, "{SELECT ?arg0")
        # clauses = arg_clauses + clauses
        # clauses.insert(0, "WHERE {")
        clauses.insert(0, f"SELECT DISTINCT ?x")
    else:
        clauses.insert(0, f"SELECT DISTINCT ?x")
    clauses.insert(0, "PREFIX ns: <http://rdf.freebase.com/ns/>")

    clauses.append('}')
    clauses.extend(order_clauses)
    
    # if superlative:
    #     clauses.append('}')
    #     clauses.append('}')

    # for clause in clauses:
    #     print(clause)

    return '\n'.join(clauses)



# linearize nested lisp exressions
def _linearize_lisp_expression(expression: list, sub_formula_id):
    sub_formulas = []
    for i, e in enumerate(expression):
        if isinstance(e, list) and e[0] != 'R':
            sub_formulas.extend(_linearize_lisp_expression(e, sub_formula_id))
            expression[i] = '#' + str(sub_formula_id[0] - 1)

    sub_formulas.append(expression)
    sub_formula_id[0] += 1
    return sub_formulas


#  I don't think this is ever gonna be implemented
def lisp_to_lambda(expressions: Union[List[str], str]):  # from lisp-grammar formula to lambda DCS
    # expressions = lisp_to_nested_expression(source_formula)
    if not isinstance(expressions, list):
        return expressions
    if expressions[0] == 'AND':
        return lisp_to_lambda(expressions[1]) + ' AND ' + lisp_to_lambda(expressions[2])
    elif expressions[0] == 'JOIN':
        return lisp_to_lambda(expressions[1]) + '*' + lisp_to_lambda(expressions[2])


if __name__=='__main__':
    
    # gt_sexpr = '(JOIN (R government.government_position_held.office_holder) (TC (AND (JOIN government.government_position_held.office_position_or_title m.0j5wjnc) (JOIN (R government.governmental_jurisdiction.governing_officials) (JOIN location.country.national_anthem (JOIN government.national_anthem_of_a_country.anthem m.0gg95zf)))) government.government_position_held.from NOW))'

    sexpr = '(JOIN (R government.government_position_held.office_holder) (TC (JOIN (R government.governmental_jurisdiction.governing_officials) (JOIN location.country.national_anthem (JOIN government.national_anthem_of_a_country.anthem m.0gg95zf))) government.government_position_held.from now))'    
    sparql = lisp_to_sparql(sexpr)
    print(sparql)
    res = execute_query(sparql)
    print(res)




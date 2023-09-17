import re
import os
from tqdm import tqdm
from components.utils import *
from components.expr_parser import parse_s_expr
from executor.sparql_executor import execute_query, execute_query_with_odbc
from executor.logic_form_util import lisp_to_sparql

class ParseError(Exception):
    pass


class Parser:
    def __init__(self):
        pass

    def parse_query_webqsp(self, query, mid_list):
        """parse a sparql query into a s-expression

        @param query: sparql query
        @param mid_list: all mids appeared in the sparql query
        """
        # print('QUERY', query)
        lines = query.split('\n')
        lines = [x for x in lines if x]

        assert lines[0] != '#MANUAL SPARQL'

        prefix_stmts = []
        line_num = 0
        while True:
            l = lines[line_num]
            if l.startswith('PREFIX'):
                prefix_stmts.append(l)
            else:
                break
            line_num = line_num + 1

        next_line = lines[line_num]
        assert next_line.startswith('SELECT DISTINCT ?x')
        line_num = line_num + 1
        next_line = lines[line_num]
        assert next_line == 'WHERE {'

        if re.match(r'ORDER BY .*\?\w*.* LIMIT 1', lines[-1]):
            lines[-1] = lines[-1].replace('LIMIT 1', '').strip()
            lines.append('LIMIT 1')
        
        if re.match(r'LIMIT \d*', lines[-1]): # TODO LIMIT n 
            lines[-1]='LIMIT 1' # transform to LIMIT 1, temporally
        
        if lines[-1].startswith('OFFSET'): # TODO LITMIT 1 \n OFFSET 1 ('the second ...')
            lines.pop(-1) # transform to LIMIT 1, temporally

        assert lines[-1] in ['}', 'LIMIT 1']

        lines = lines[line_num:]

        filter_string_flag = not all(['FILTER (str' not in x for x in lines])

        # assert all(['FILTER (str' not in x for x in lines])

        # normalize body lines
        body_lines, spec_condition, filter_lines = self.normalize_body_lines(
            lines, filter_string_flag)
        body_lines = [x.strip() for x in body_lines]  # strip spaces
        # assert all([x.startswith('?') or x.startswith('ns') or x.startswith('FILTER') for x in body_lines])
        # we only parse query following this format
        if body_lines[0].startswith('FILTER'):
            predefined_filter0 = body_lines[0]
            predefined_filter1 = body_lines[1]

            # filter_0_line validation
            filter_0_valid = (predefined_filter0 == f'FILTER (?x != ?c)')
            if not filter_0_valid:
                for mid in mid_list:
                    filter_0_valid = filter_0_valid or (
                        predefined_filter0 == f'FILTER (?x != {mid})')

            assert filter_0_valid

            # filter_1_line validation
            assert predefined_filter1 == "FILTER (!isLiteral(?x) OR lang(?x) = '' OR langMatches(lang(?x), 'en'))"
            # if predefined_filter0 != f'FILTER (?x != ns:{topic_mid})':
            #     print('QUERY', query)
            #     print('First Filter')
            # if predefined_filter1 != "FILTER (!isLiteral(?x) OR lang(?x) = '' OR langMatches(lang(?x), 'en'))":
            #     print('QUERY', query)
            #     print('Second Filter')
            # if any([not (x.startswith('?') or x.startswith('ns:')) for x in body_lines]):
            #     print('Unprincipled Filter')
            #     print('QUERY', query)
            body_lines = body_lines[2:]

        # body line form assertion
        assert all([(x.startswith('?') or x.startswith('ns:'))
                   for x in body_lines])
        # print(body_lines)

        var_dep_list = self.parse_naive_body(
            body_lines, filter_lines, '?x', spec_condition)
        s_expr = self.dep_graph_to_s_expr(var_dep_list, '?x', spec_condition)
        return s_expr

    def normalize_body_lines(self, lines, filter_string_flag=False):
        """return normalized body lines of sparql, specially return filter lines starting with `FILTER (str(`        

        @param lines: sparql lines list
        @param filter_string_flag: flag indicates existence of filter lines


        @return: (body_lines,
                    spec_condition,
                    # [
                    #     ['SUPERLATIVE', argmax/argmin, arg_var, arg_r], 
                    #     ['COMPARATIVE', gt/lt/ge/le, compare_var, compare_value, compare_rel],
                    #     ['RANGE', range_relation, range_var, range_year],
                    # ]
                    filter_lines
                  )
        """

        spec_condition = []

        # 1. get literal filter_lines
        # ?x ns:base.biblioness.bibs_location.loc_type ?sk0 .
        # FILTER (str(?sk0) = "Country")
        if filter_string_flag:
            filter_lines = [x.strip() for x in lines if 'FILTER (str' in x]
            lines = [x.strip() for x in lines if 'FILTER (str' not in x]
        else:
            lines = [x.strip() for x in lines]
            filter_lines = None
        
        # 2. get compare lines
        # 2.1 FILTER (?num > "2009-01-02"^^xsd:dateTime) .
        # 2.2 FILTER (xsd:integer(?num) < 33351310952) . 
        if re.match(r'FILTER \(\?\w* (>|<|>=|<=) .*',lines[-2]) \
            or re.match(r'FILTER \(xsd:integer\(\?\w*\) (>|<|>=|<=) .*',lines[-2]):
            
            compare_line = lines.pop(-2)
            compare_var = re.findall(r'\?\w*',compare_line)[0]
            compare_operator = re.findall(r'(>|>=|<|<=)',compare_line)[0]
            operator_mapper = {'<':'lt','<=':'le','>':'gt',">=":"ge"}
            if "^^xsd:dateTime" in compare_line: # dateTime
                compare_value = re.findall(r'".*"\^\^xsd:dateTime',compare_line)[0]
            else: # number
                compare_value = compare_line.replace(") .","").split(" ")[-1]

            compare_value = compare_value.replace('"','') # remove \" in compare value
            # print(variable,compare_operator,compare_value)
            compare_condition = ['COMPARATIVE', operator_mapper[compare_operator],compare_var,compare_value]
            spec_condition.append(compare_condition)
            
        # 3. get range lines, move to the end of where clause
        # WHERE {
            # ns:m.04f_xd8 ns:government.government_office_or_title.office_holders ?y .
            # ?y ns:government.government_position_held.office_holder ?x .
            # FILTER(NOT EXISTS {?y ns:government.government_position_held.from ?sk0} ||
            # EXISTS {?y ns:government.government_position_held.from ?sk1 .
            # FILTER(xsd:datetime(?sk1) <= "2009-12-31"^^xsd:dateTime) })
            # FILTER(NOT EXISTS {?y ns:government.government_position_held.to ?sk2} ||
            # EXISTS {?y ns:government.government_position_held.to ?sk3 .
            # FILTER(xsd:datetime(?sk3) >= "2009-01-01"^^xsd:dateTime) })
            # }
        start_line = -1
        right_parantheses_line = -1
        not_exists_num = 0
        for i, line in enumerate(lines):
            if line.startswith("FILTER(NOT EXISTS"):
                not_exists_num +=1
                if start_line == -1:
                    start_line = i
            # if line.startswith("FILTER(") and "2015-08-10" in line and start_line != -1:
            #     meaningless_time_flag = True
            if line == '}':
                right_parantheses_line = i

        if start_line != -1:
            
            if not_exists_num==4: # redundant range filters
                end_line = start_line+12
            else:
                end_line = start_line+6
            
            assert end_line <= right_parantheses_line
            
            if end_line==start_line+12: # discard redundant range filters
                lines = lines[:start_line]+lines[end_line:right_parantheses_line] + \
                        lines[start_line:end_line-6]+lines[right_parantheses_line:]
            else:
                lines = lines[:start_line]+lines[end_line:right_parantheses_line] + \
                        lines[start_line:end_line]+lines[right_parantheses_line:]
                    
           
        # 4. get SUPERLATIVE lines
        body_lines = []
        if lines[-1] == 'LIMIT 1':
            # spec_condition = argmax
            # who did jackie robinson first play for?
            # WHERE {
            # ns:m.0443c ns:sports.pro_athlete.teams ?y .
            # ?y ns:sports.sports_team_roster.team ?x .
            # ?y ns:sports.sports_team_roster.from ?sk0 .
            # }
            # ORDER BY DESC(xsd:datetime(?sk0))
            # LIMIT 1
            order_line = lines[-2]
            direction = 'argmax' if 'DESC(' in order_line else 'argmin'
            compare_var = re.findall(r'\?\w*', order_line)[0]
            # assert ('?sk0' in order_line) # variable in order_line
            assert(compare_var in order_line)

            _tmp_body_lines = lines[1:-3]
            
            hit = False
            for l in _tmp_body_lines:
                if compare_var in l :
                    if 'FILTER' in l: # the return var is also the argmax var, not covered by S-Expression
                        assert 1==2 # raise AssertionError
                    # self.parse_assert(l.endswith('?sk0 .') and not hit)
                    self.parse_assert(l.endswith(compare_var+" .")
                                      and not hit)  # appear only once
                    hit = True
                    arg_var, arg_r = l.split(' ')[0], l.split(' ')[1]
                    arg_r = arg_r[3:]  # rm ns:
                else:
                    body_lines.append(l)

            superlative_cond = ['SUPERLATIVE',direction,arg_var,arg_r]
            spec_condition.append(superlative_cond)
        
            # if not lines[-4].startswith('FILTER(NOT EXISTS {?'):
            #     if filter_string_flag:
            #         return body_lines, [direction, arg_var, arg_r], filter_lines
            #     else:
            #         return body_lines, [direction, arg_var, arg_r], None
            # else:
            #     # contains range constraints FILTER
            #     pass

        # 4. process range lines
        if body_lines: # already processed by superlative extraction
            lines = body_lines
            range_line_num = -6
        else:
            range_line_num = -7
        if len(lines)>= abs(range_line_num) and lines[range_line_num].startswith('FILTER(NOT EXISTS {?'):
            # WHERE {
            # ns:m.04f_xd8 ns:government.government_office_or_title.office_holders ?y .
            # ?y ns:government.government_position_held.office_holder ?x .
            # FILTER(NOT EXISTS {?y ns:government.government_position_held.from ?sk0} ||
            # EXISTS {?y ns:government.government_position_held.from ?sk1 .
            # FILTER(xsd:datetime(?sk1) <= "2009-12-31"^^xsd:dateTime) })
            # FILTER(NOT EXISTS {?y ns:government.government_position_held.to ?sk2} ||
            # EXISTS {?y ns:government.government_position_held.to ?sk3 .
            # FILTER(xsd:datetime(?sk3) >= "2009-01-01"^^xsd:dateTime) })
            # }
            if not body_lines:
                body_lines = lines[1:-7]
                range_lines = lines[-7:-1]
            else:
                body_lines = lines[:-6]
                range_lines = lines[-6:]
            range_prompt = range_lines[0]
            range_prompt = range_prompt[range_prompt.index(
                '{') + 1:range_prompt.index('}')]
            range_var = range_prompt.split(' ')[0]
            range_relation = range_prompt.split(' ')[1]
            # range_relation = '.'.join(
            #     range_relation.split('.')[:2]) + '.time_macro'
            range_relation = range_relation[3:]  # rm ns:
            range_start_time = re.findall(f'".*"\^\^',range_lines[2])[0].split("^^")[0].strip('"')
            if range_start_time =='2015-08-10':
                range_start_time = 'NOW'
            range_start = range_lines[2].split(' ')[2]
            range_start = range_start[1:]
            range_start = range_start[:range_start.index('"')]
            
            range_end = range_lines[5].split(' ')[2]
            range_end = range_end[1:]
            range_end = range_end[:range_end.index('"')]

            # assert range_start[:4] == range_end[:4]
            # to fit parsable
            # range_year = range_start[:4] + \
            #     '^^http://www.w3.org/2001/XMLSchema#dateTime' if range_start_time != 'NOW' else 'NOW'
            range_year = range_start[:4] if range_start_time != 'NOW' else 'NOW'
            range_start_cond = ['RANGE', range_relation, range_var, range_year]
            spec_condition.append(range_start_cond)
            
            # if filter_string_flag:
            #     return body_lines, ['range', range_var, range_relation, range_year], filter_lines
            # else:
            #     return body_lines, ['range', range_var, range_relation, range_year], None
        
        # body_lines not extracted yet
        if not body_lines: 
            body_lines = lines[1:-1]
            # if filter_string_flag:
            #     return body_lines, None, filter_lines
            # else:
            #     return body_lines, None, None
        
        return body_lines, spec_condition, filter_lines
        

    def dep_graph_to_s_expr(self, var_dep_list, ret_var, spec_condition=None):
        """Convert dependancy graph to s_expression
        @param var_dep_list: varialbe dependancy list
        @param ret_var: return var
        @param spec_condition: special condition

        @return s_expression
        """
        self.parse_assert(var_dep_list[0][0] == ret_var)
        var_dep_list.reverse() # reverse the var_dep_list
        parsed_dict = {}  # dict for parsed variables

        # spec_condition,
        #             # [
        #             #     ['SUPERLATIVE', argmax/argmin, arg_var, arg_r], 
        #             #     ['COMPARATIVE', gt/lt/ge/le, compare_var, compare_value],
        #             #     ['RANGE', range_relation, range_var, range_year],
        #             # ]

        # specical condition var map {spec_var:idx in spec_condition}
        spec_var_map = {cond[2]:i for i,cond in enumerate(spec_condition)} if spec_condition else None
        # spec_var = spec_condition[1] if spec_condition is not None else None

        for var_name, dep_relations in var_dep_list:
            # expr = ''
            dep_relations[0]
            clause = self.triplet_to_clause(
                var_name,  dep_relations[0], parsed_dict)
            for tri in dep_relations[1:]:
                n_clause = self.triplet_to_clause(var_name, tri, parsed_dict)
                clause = 'AND ({}) ({})'.format(n_clause, clause)
            # if var_name == spec_var:
            if spec_var_map and var_name in spec_var_map: # spec_condition
                cond = spec_condition[spec_var_map[var_name]]
                # if cond[0] == 'argmax' or cond[0] == 'argmin': # superlative
                if cond[0]=='SUPERLATIVE':
                    #relation = spec_condition[2]
                    relation = cond[3]
                    clause = '{} ({}) {}'.format(
                        cond[1].upper(), clause, relation)
                elif cond[0] == 'RANGE':
                    relation, time_point = cond[1], cond[3]
                    clause = 'TC ({}) {} {}'.format(clause, relation, time_point)
                    # n_clause = 'TC {} {}'.format(relation, time_point)
                    # clause = 'AND ({}) ({})'.format(n_clause, clause)
                elif cond[0] == 'COMPARATIVE':
                    op = cond[1]
                    value = cond[3]
                    rel = cond[4]
                    n_clause = '{} {} {}'.format(op, rel, value)
                    clause = 'AND ({}) ({})'.format(n_clause, clause)
                    # pass
            parsed_dict[var_name] = clause
        
        res = '(' + parsed_dict[ret_var] + ')'
        res = res.replace('xsd:','http://www.w3.org/2001/XMLSchema#')
        return res

    def triplet_to_clause(self, tgt_var, triplet, parsed_dict):
        """Convert a triplet to S_expression clause
        @param tgt_var: target variable
        @param triplet: triplet in sparql
        @param parsed_dict: dict for variables already parsed
        """
        if triplet[0] == tgt_var:
            this = triplet[0]
            other = triplet[-1]
            if other in parsed_dict:
                other = '(' + parsed_dict[other] + ')'
            return 'JOIN {} {}'.format(triplet[1], other)
        elif triplet[-1] == tgt_var:
            this = triplet[-1]
            other = triplet[0]
            if other in parsed_dict:
                other = '(' + parsed_dict[other] + ')'
            return 'JOIN (R {}) {}'.format(triplet[1], other)
        else:
            raise ParseError()

    def parse_assert(self, eval):
        if not eval:
            raise ParseError()

    def parse_naive_body(self, body_lines, filter_lines, ret_var, spec_condition=None):
        """Parse body lines
        @param body_lines: list of sparql body lines
        @param filter_lines: lines that start with `FILTER (str(?`
        @param ret_var: return var, default `?x`
        @param spec_condition: spec_condition like
                    # [
                    #     ['SUPERLATIVE', argmax/argmin, arg_var, arg_r], 
                    #     ['COMPARATIVE', gt/lt/ge/le, compare_var, compare_value, compare_rel],
                    #     ['RANGE', range_relation, range_var, range_year],
                    # ]

        @return: variable dependancy list
        """
        # ret_variable
        # body_lines
        assert all([x[-1] == '.' for x in body_lines])
        # filter lines assertion
        if filter_lines:
            assert all(['FILTER (str' in x for x in filter_lines])


        triplets = [x.replace('"','') if "^^xsd:" in x else x for x in body_lines]
        triplets = [x.split(' ') for x in triplets]  # split by '
                
        triplets = [x[:2] + [" ".join(x[2:-1]), x[-1]] if len(x)>4 else x for x in triplets] # avoid error splitting like "2100 Woodward Avenue"@en
        triplets = [x[:-1] if x[-1] == '.' else x for x in triplets]  # remove '.'
        
        

        # remove ns
        triplets = [[x[3:] if x.startswith(
            'ns:') else x for x in tri] for tri in triplets]
        # dependancy graph
        triplets_pool = triplets
        # while True:
        # varaible dependancy list, in the form like [(?x,[['?x','ns:aaa.aaa.aaa','?y'],['ns:m.xx','ns:bbb.bbb.bbb','?x''])]
        var_dep_list = []
        successors = []

        # firstly solve the return variable
        dep_triplets, triplets_pool = self.resolve_dependancy(
            triplets_pool, filter_lines, ret_var, successors)
        var_dep_list.append((ret_var, dep_triplets))
        # vars_pool = []
        # go over un resolved vars
        # for tri in triplets_pool:
        #     if tri[0].startswith('?') and tri[0] not in vars_pool and tri[0] != ret_var:
        #         vars_pool.append(tri[0])
        #     if tri[-1].startswith('?') and tri[-1] not in vars_pool and tri[-1] != ret_var:
        #         vars_pool.append(tri[-1])

        # for tgt_var in vars_pool:
        #     dep_triplets, triplets_pool = self.resolve_dependancy(triplets_pool, tgt_var)
        #     self.parse_assert(len(dep_triplets) > 0)
        #     var_dep_list.append((tgt_var, dep_triplets))

        # handle all the successor variables
        while len(successors):
            tgt_var = successors[0]
            successors = successors[1:]
            dep_triplets, triplets_pool = self.resolve_dependancy(
                triplets_pool, filter_lines, tgt_var, successors)

            # if (len(dep_triplets)==0):
            #     # no triplet for tgt_var
            #     # ?x ns:xxx ?c
            #     # ?c ns:xxx ?num
            #     # ORDER BY ?num LIMIT 1
            #     print(dep_triplets)

            # assert len(dep_triplets) > 0 # at least one dependancy triplets
            if len(dep_triplets) == 0:
                # zero dep_triples, can be a 2-hop constraint
                # e.g.
                # 'ns:m.0d0x8 ns:government.political_district.representatives ?y .'
                # '?y ns:government.government_position_held.office_holder ?x .'
                # '?y ns:government.government_position_held.governmental_body ns:m.07t58 .'
                # '?x ns:government.politician.government_positions_held ?c .'
                
                if spec_condition and any([tgt_var in x for x in spec_condition]):
                    cond = []
                    for x in spec_condition:
                        if tgt_var in x:
                            cond = x
                            break
                    
                    repeat = True
                    while repeat:        
                        # tgt_var is a var in spec_condition
                        for (var, triplets) in var_dep_list:
                            if any([tgt_var in trip for trip in triplets]):
                                head_var = var  # find the real constrained var
                                _temp_triplets = triplets[:]
                                triplets.clear()
                                for trip in _temp_triplets:
                                    if tgt_var not in trip:
                                        triplets.append(trip)
                                    else:
                                        # find the constraint relation
                                        cons_rel = trip[1]
                                        if trip[0] == head_var:
                                            reversed_direction = False
                                        else:
                                            reversed_direction = True
                                        cons_rel = f'(R {cons_rel})' if reversed_direction else cons_rel

                                # modify spec_condition
                                # spec_condition[1] = head_var
                                if cond[0]=='COMPARATIVE':
                                    cond[2] = head_var
                                    if len(cond)<5:
                                        cond.append(cons_rel)
                                    else:
                                        cond[4] = "(JOIN " + cons_rel+" "+ cond[4]+")"
                                else: # SUPERLATIVE
                                    cond[2] = head_var
                                    cond[3] = "(JOIN "+ cons_rel+" "+cond[3]+")"
                                tgt_var = head_var
                        
                        # check whether need to repeat
                        remove_idx=-1
                        for i,(var,triplets) in enumerate(var_dep_list):
                            if var == head_var:
                                if len(triplets)==0:
                                    repeat = True
                                    remove_idx = i
                                else:
                                    repeat = False
                                break
                        
                        if remove_idx>=0:
                            var_dep_list.pop(remove_idx)
                        else:
                            repeat=False
         
                else:
                    # uncovered situation
                    assert 1 == 2
            else:
                """dep_triplets not None"""
                self.parse_assert(len(dep_triplets) > 0)  # at least one dependancy triplets
                var_dep_list.append((tgt_var, dep_triplets))

        if(len(triplets_pool) != 0):
            print(triplets_pool)

        self.parse_assert(len(triplets_pool) == 0)
        return var_dep_list

    def resolve_dependancy(self, triplets, filter_lines, target_var, successors):
        """resolve dependancy of variables
        @param triplets: all sparql triplet lines
        @param filter_lines: filter lines that start with `Filter (str(`
        @param target_var: target variable
        @param successors: successor variables of target variable

        @return: dependancy triplets of target_var, left triplets (independant of target_var)
        """
        dep = []
        left = []
        if not triplets:  # empty triplets, target_var constrained by filter

            # ns:m.0f9wd ns:influence.influence_node.influenced ?x .
            # ?x ns:government.politician.government_positions_held ?c .
            # ?c ns:government.government_position_held.from ?num .
            # ORDER BY ?num LIMIT 1
            pass
        else:
            for tri in triplets:
                if tri[0] == target_var:  # head is target variable
                    dep.append(tri)  # add to dependancy triplets
                    # tail is variable
                    if tri[-1].startswith('?') and tri[-1] not in successors:
                        successor_var = tri[-1]
                        if filter_lines:  # check filter variable `?sk0`
                            new_filter_lines = []
                            found_filter_variable = False
                            for line in filter_lines:
                                if successor_var in line:
                                    found_filter_variable = True
                                    line = line.replace(
                                        'FILTER (str(', '').replace(')', '')
                                    tuple_list = line.split('=')
                                    var = tuple_list[0].strip()
                                    value = tuple_list[1].strip()

                                    assert successor_var == var
                                    if value.isalpha():
                                        tri[-1] = value+'@en'
                                    else:
                                        tri[-1] = value
                                    # tri[-1] = value+'@en'
                                else:
                                    new_filter_lines.append(line)

                            # remove corresponding filter_lines
                            if not found_filter_variable:  # no filter variable found
                                # add to successor variable
                                successors.append(successor_var)

                            filter_lines = new_filter_lines

                        else:
                            # add to successor variable
                            successors.append(successor_var)
                elif tri[-1] == target_var:  # tail is target variable
                    dep.append(tri)  # add to dependancy triplets
                    # head is variable
                    if tri[0].startswith('?') and tri[0] not in successors:
                        successors.append(tri[0])  # add to successor variable
                else:
                    left.append(tri)  # left triplets
        return dep, left


def convert_parse_instance(parse):
    """convert a webqsp parse instance to a s_expr"""
    sparql = parse['Sparql']
    # print(parse.keys())
    # print(parse['PotentialTopicEntityMention'])
    # print(parse['TopicEntityMid'], parse['TopicEntityName'])
    try:
        s_expr = parser.parse_query(sparql, parse['TopicEntityMid'])
        # print('---GOOD------')
        # print(sparql)
        # print(s_expr)
    except AssertionError:
        s_expr = 'null'
    # print(parse[''])
    parse['SExpr'] = s_expr
    return parse, s_expr != 'null'


def webq_s_expr_to_sparql_query(s_expr):
    ast = parse_s_expr(s_expr)


def execute_webq_s_expr(s_expr):
    try:
        sparql_query = lisp_to_sparql(s_expr)
        print(f'Transformed sparql:\n{sparql_query}')
        denotation = execute_query(sparql_query)
    except:
        denotation = []
    return denotation


def augment_with_s_expr_webqsp(split, check_execute_accuracy=False):
    """augment original webqsp datasets with s-expression"""
    #dataset = load_json(f'data/origin/ComplexWebQuestions_{split}.json')
    dataset = load_json(f'data/WebQSP/origin/WebQSP.{split}.json')
    dataset = dataset['Questions']

    total_num = 0
    hit_num = 0
    execute_hit_num = 0
    failed_instances = []
    for i,data in tqdm(enumerate(dataset), total=len(dataset)):
        
        # sparql = data['sparql']  # sparql string
        parses = data['Parses']
        for parse in parses:
            total_num += 1
            sparql = parse['Sparql']
        
            instance, flag_success = convert_webqsp_sparql_instance(sparql, parse)
            if flag_success:
                hit_num += 1
                if check_execute_accuracy:
                    execute_right_flag = False
                    try:
                        execute_ans = execute_query_with_odbc(lisp_to_sparql(instance['SExpr']))
                        execute_ans = [res.replace("http://rdf.freebase.com/ns/",'') for res in execute_ans]
                        if 'Answers' in parse:
                            gold_ans = [ans['AnswerArgument'] for ans in parse['Answers']]
                        else:
                            gold_ans = execute_query_with_odbc(parse['Sparql'])
                            gold_ans = [res.replace("http://rdf.freebase.com/ns/",'') for res in gold_ans]
                        # if split=='test':
                        #     gold_ans = execute_query(parse['Sparql'])
                        # else:
                        #     gold_ans = [x['answer_id'] for x in data['answers']]

                        if set(execute_ans) == set(gold_ans):
                            execute_hit_num +=1
                            execute_right_flag = True
                            # print(f'{i}: SExpr generation:{flag_success}, Execute right:{execute_right_flag}')
                        else:
                            temp = execute_query_with_odbc(lisp_to_sparql(instance['SExpr']))
                        instance['SExpr_execute_right'] = execute_right_flag
                    except Exception:
                        temp = execute_query_with_odbc(lisp_to_sparql(instance['SExpr']))
                        # instance['SExpr_executed_succeed']=False
                        instance['SExpr_execute_right'] = execute_right_flag
                    if not execute_right_flag:
                        pass
                        # print(f'ID:{instance["ID"]},\nExpected Ansewr:{gold_ans},\nGot Answer:{execute_ans}')
            else:
                # if check_execute_accuracy:
                #     instance['SExpr_execute_right'] = False
                failed_instances.append(instance)
    # print(hit_num, total_num, hit_num/total_num, len(dataset))
        if (i+1)%100==0:
            print(f'In the First {i+1} questions, S-Expression Gen rate [{split}]: {hit_num}, {total_num}, {hit_num/total_num}, {i+1}')
            if check_execute_accuracy:            
                    print(f'In the First {i+1} questions, Execute right rate [{split}]: {execute_hit_num}, {total_num}, {execute_hit_num/total_num}, {i+1}', )

    print(f'S-Expression Gen rate [{split}]: {hit_num}, {total_num}, {hit_num/total_num}, {len(dataset)}')
    print(f'Execute right rate [{split}]: {execute_hit_num}, {total_num}, {execute_hit_num/total_num}, {len(dataset)}', )
    

    sexpr_dir = 'data/WebQSP/sexpr'
    if not os.path.exists(sexpr_dir):
        os.makedirs(sexpr_dir)

    print(f'Writing S_Expression Results into {sexpr_dir}/WebQSP.{split}.expr.json')

    dump_json(dataset, f'{sexpr_dir}/WebQSP.{split}.expr.json', indent=4)
    # dump_json(failed_instances, f'data/WEBQSP/sexpr/WebQSP.{split}.failed.json', indent=4)


def convert_webqsp_sparql_instance(sparql, origin_data):
    """convert a webqsp sparql to a s_expr"""
    # mid_list = []
    # pattern_str = r'ns:m\.0\w*'
    # pattern = re.compile(pattern_str)
    # mid_list = list(set([mid.strip()
    #                 for mid in re.findall(pattern_str, sparql)]))
    
    mid_list = [f'ns:{origin_data["TopicEntityMid"]}']
    
    # for debug
    # if origin_data['TopicEntityMid'] in ['m.05bz_j','m.0166b']:
    #     print('for debug')

    try:
        s_expr = parser.parse_query_webqsp(sparql, mid_list)
    except AssertionError:
        if '#MANUAL SPARQL' not in sparql:
            print(f'Error processing sparql: {sparql}')
        s_expr = 'null'

    origin_data['SExpr'] = s_expr
    return origin_data, s_expr != 'null'


def find_macro_template_from_query(query, topic_mid):
    # print('QUERY', query)
    lines = query.split('\n')
    lines = [x for x in lines if x]

    assert lines[0] != '#MANUAL SPARQL'

    prefix_stmts = []
    line_num = 0
    while True:
        l = lines[line_num]
        if l.startswith('PREFIX'):
            prefix_stmts.append(l)
        else:
            break
        line_num = line_num + 1

    next_line = lines[line_num]
    assert next_line.startswith('SELECT DISTINCT ?x')
    line_num = line_num + 1
    next_line = lines[line_num]
    assert next_line == 'WHERE {'
    assert lines[-1] in ['}', 'LIMIT 1']

    lines = lines[line_num:]
    assert all(['FILTER (str' not in x for x in lines])
    # normalize body lines
    # return_val = check_time_macro_from_body_lines(lines)
    # if return_val:

    # relation_prefix, suffix_pair = c
    return check_time_macro_from_body_lines(lines)


def check_time_macro_from_body_lines(lines):
    # check if xxx
    if lines[-4].startswith('FILTER(NOT EXISTS {?'):
        # WHERE {
        # ns:m.04f_xd8 ns:government.government_office_or_title.office_holders ?y .
        # ?y ns:government.government_position_held.office_holder ?x .
        # FILTER(NOT EXISTS {?y ns:government.government_position_held.from ?sk0} ||
        # EXISTS {?y ns:government.government_position_held.from ?sk1 .
        # FILTER(xsd:datetime(?sk1) <= "2009-12-31"^^xsd:dateTime) })
        # FILTER(NOT EXISTS {?y ns:government.government_position_held.to ?sk2} ||
        # EXISTS {?y ns:government.government_position_held.to ?sk3 .
        # FILTER(xsd:datetime(?sk3) >= "2009-01-01"^^xsd:dateTime) })
        # }
        body_lines = lines[1:-7]
        range_lines = lines[-7:-1]
        range_prompt_start = range_lines[0]
        range_prompt_start = range_prompt_start[range_prompt_start.index(
            '{') + 1:range_prompt_start.index('}')]
        range_relation_start = range_prompt_start.split(' ')[1]

        # range_relation = '.'.join(range_relation.split('.')[:2]) + '.time_macro'
        # range_relation = range_relation[3:] # rm ns:

        range_prompt_end = range_lines[3]
        range_prompt_end = range_prompt_end[range_prompt_end.index(
            '{') + 1:range_prompt_end.index('}')]
        range_relation_end = range_prompt_end.split(' ')[1]

        assert range_relation_start.split(
            '.')[:2] == range_relation_end.split('.')[:2]
        start_suffix = range_relation_start.split('.')[-1]
        end_suffix = range_relation_end.split('.')[-1]
        prefix = '.'.join(range_relation_start.split('.')[:2])[3:]
        return prefix, start_suffix, end_suffix
    else:
        return None


def extract_macro_template_from_instance(parse):
    sparql = parse['Sparql']
    # print(parse.keys())
    # print(parse['PotentialTopicEntityMention'])
    # print(parse['TopicEntityMid'], parse['TopicEntityName'])
    try:
        return find_macro_template_from_query(sparql, parse['TopicEntityMid'])
    except AssertionError:
        return None



def parse_webqsp_sparql(check_execute_accuracy=False):
    """Parse WebQSP sparqls into s-expressions"""
    augment_with_s_expr_webqsp('train',check_execute_accuracy)
    # augment_with_s_expr_webqsp('dev',check_execute_accuracy)
    augment_with_s_expr_webqsp('test',check_execute_accuracy)    
    


if __name__ == '__main__':
    
    parser = Parser()
    """
    Since WebQSP may provide multiple `Parses` for each question
    Execution accuracy of generated S-Expression will be verified.
    It will later be used as an filtering condition in step (5).1
    """
    parse_webqsp_sparql(check_execute_accuracy=False)

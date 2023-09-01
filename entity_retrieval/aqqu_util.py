"""
Copyright 2015, University of Freiburg.

Elmar Haussmann <haussmann@cs.uni-freiburg.de>
"""
import re
from nltk import word_tokenize


def normalize_entity_name(name):
    name = name.lower()
    # name = name.replace('!', '')
    # name = name.replace('.', '')
    # name = name.replace(',', '')
    # name = name.replace('-', '')
    # name = name.replace('_', '')
    # name = name.replace(' ', '')
    # name = name.replace('\'', '')
    # name = name.replace('"', '')
    # name = name.replace('\\', '')


    # the following is only for freebase_complete_all_mention
    name = ' '.join(word_tokenize(name))
    # word_tokenize from nltk will change the left " to ``, which is pretty weird. Fix it here
    name = name.replace('``', '"').replace("''", '"')

    return name


def read_abbreviations(abbreviations_file):
    '''
    Return a set of abbreviations.
    :param abbreviations_file:
    :return:
    '''
    abbreviations = set()
    with open(abbreviations_file, 'r') as f:
        for line in f:
            abbreviations.add(line.strip().decode('utf-8').lower())
    return abbreviations


def remove_abbreviations_from_entity_name(entity_name,
                                          abbreviations):
    tokens = entity_name.lower().split(' ')
    non_abbr_tokens = [t for t in tokens if t not in abbreviations]
    return ' '.join(non_abbr_tokens)


def remove_prefixes_from_name(name):
    if name.startswith('the'):
        name = name[3:]
    return name


def remove_suffixes_from_name(name):
    if '#' in name or '(' in name:
        name = remove_number_suffix(name)
        name = remove_bracket_suffix(name)
    return name


def remove_number_suffix(name):
    res = re.match(r'.*( #[0-9]+)$', name)
    if res:
        name = name[:res.start(1)]
        return name
    else:
        return name


def remove_bracket_suffix(name):
    res = re.match(r'.*( \([^\(\)]+\))$', name)
    if res:
        name = name[:res.start(1)]
        return name
    else:
        return name
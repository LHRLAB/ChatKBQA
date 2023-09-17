import logging
import re
import time
# import stanfordnlp
from entity_retrieval import surface_index_memory
from entity_retrieval.aqqu_util import normalize_entity_name, remove_prefixes_from_name, remove_suffixes_from_name

logger = logging.getLogger(__name__)


class Entity(object):
    """An entity.

    There are different types of entities inheriting from this class, e.g.,
    knowledge base entities and values.
    """

    def __init__(self, name):
        self.name = name

    def sparql_name(self):
        """Returns an id w/o sparql prefix."""
        pass

    def prefixed_sparql_name(self, prefix):
        """Returns an id with sparql prefix."""
        pass


class KBEntity(Entity):
    """A KB entity."""

    def __init__(self, name, identifier, score, aliases):
        Entity.__init__(self, name)
        # The unique identifier used in the knowledge base.
        self.id = identifier
        # A popularity score.
        self.score = score
        # The entity's aliases.
        self.aliases = aliases

    def sparql_name(self):
        return self.id

    def prefixed_sparql_name(self, prefix):
        return "%s:%s" % (prefix, self.id)

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name


class Value(Entity):
    """A value.

     Also has a name identical to its value."""

    def __init__(self, name, value):
        Entity.__init__(self, name)
        # The unique identifier used in the knowledge base.
        self.value = value

    def sparql_name(self):
        return self.value

    def prefixed_sparql_name(self, prefix):
        return "%s:%s" % (prefix, self.value)

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        return self.value == other.value


class DateValue(Value):
    """A date.

    It returns a different sparql name from a value or normal entity.
    """

    def __init__(self, name, date):
        Value.__init__(self, name, date)

    def sparql_name(self):
        return self.value

    def prefixed_sparql_name(self, prefix):
        # Old version uses lowercase t in dateTime
        #return '"%s"^^xsd:dateTime' % self.value
        return '"%s"^^xsd:datetime' % self.value

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        return self.value == other.value


class IdentifiedEntity():
    """An entity identified in some text."""

    def __init__(self, mention,
                 name, entity,
                 score=0, surface_score=0,
                 perfect_match=False):
        # A readable name to be displayed to the user.
        self.name = name
        # The tokens that matched this entity.
        self.mention = mention
        # A score for the match of those tokens.
        self.surface_score = surface_score
        # A popularity score of the entity.
        self.score = score
        # The identified entity object.
        self.entity = entity
        # A flag indicating whether the entity perfectly
        # matched the tokens.
        self.perfect_match = perfect_match

    def as_string(self):
        t = ','.join(["%s" % t.text
                      for t in self.mention])
        return "%s: tokens:%s prob:%.3f score:%s perfect_match:%s" % \
               (self.name, t,
                self.surface_score,
                self.score,
                self.perfect_match)

    def overlaps(self, other):
        """Check whether the other identified entity overlaps this one."""
        return set(self.mention) & set(other.mention)

    def sparql_name(self):
        return self.entity.sparql_name()

    def prefixed_sparql_name(self, prefix):
        return self.entity.prefixed_sparql_name(prefix)


def get_value_for_year(year):
    """Return the correct value representation for a year."""
    # Older Freebase versions do not have the long suffix.
    #return "%s-01-01T00:00:00+01:00" % (year)
    return "%s" % year


class EntityLinker:

    def __init__(self, surface_index,
                 # Better name it max_entities_per_surface
                 max_entities_per_tokens=4):
        self.surface_index = surface_index
        # The max number of entities to keep for a mention according to the surface score
        self.max_entities_per_tokens = max_entities_per_tokens
        # Entities are a mix of nouns, adjectives and numbers and
        # a LOT of other stuff as it turns out:
        # UH, . for: hey arnold!
        # MD for: ben may library
        # PRP for: henry i
        # FW for: ?
        # self.valid_entity_tag = re.compile(r'^(UH|\.|TO|PRP.?|#|FW|IN|VB.?|'
        #                                    r'RB|CC|NNP.?|NN.?|JJ.?|CD|DT|MD|'
        #                                    r'POS)+$')
        # The original patterns used by AQQU.
        self.valid_entity_tag = re.compile(r'^(UH|\.|TO|PRP.?|#|FW|IN|VB.?|'
                                           r'RB|CC|HYPH|WP|XX|NNP.?|NN.?|JJ.?|CD|DT|MD|'
                                           r'POS)+$')
        self.ignore_lemmas = {'be', 'of', 'the', 'and', 'or', 'a'}
        self.year_re = re.compile(r'[0-9]{4}')

    def get_entity_for_mid(self, mid):
        '''
        Returns the entity object for the MID or None
         if the MID is unknown. Forwards to surface index.
        :param mid:
        :return:
        '''
        return self.surface_index.get_entity_for_mid(mid)
    '''
    @staticmethod
    def init_from_config():
        """
        Return an instance with options parsed by a config parser.
        :param config_options:
        :return:
        """
        config_options = globals.config
        surface_index = EntitySurfaceIndexMemory.init_from_config()
        max_entities_p_token = int(config_options.get('EntityLinker',
                                                      'max-entites-per-tokens'))
        return EntityLinker(surface_index,
                            max_entities_per_tokens=max_entities_p_token)
    '''

    def _text_matches_main_name(self, entity, text):

        """
        Check if the entity name is a perfect match on the text.
        :param entity:
        :param text:
        :return:
        """
        text = normalize_entity_name(text)
        text = remove_prefixes_from_name(text)
        name = remove_suffixes_from_name(entity.name)
        name = normalize_entity_name(name)
        name = remove_prefixes_from_name(name)
        if name == text:
            return True
        return False

    # The original rules in AQQU
    def is_entity_occurrence(self, tokens, start, end):
        '''
        Return true if the tokens marked by start and end indices
        are a valid entity occurrence.
        :param tokens:
        :param start:
        :param end:
        :return:
        '''

        token_list = tokens[start:end]
        # Entity mentions cannot be empty
        if len(token_list) < 1:
            return False
        # Concatenate POS-tags
        pos_list = [t.pos for t in token_list]
        pos_str = ''.join(pos_list)
        # Check if all tokens are in the ignore list.
        if all([t.lemma in self.ignore_lemmas for t in token_list]):
            return False

        # Entity mentions cannot start with an ignored lemma
        if token_list[0].lemma in self.ignore_lemmas and \
                token_list[0].lemma != 'the':
            return False

        # For length 1 only allows nouns and foreign and unknown word types
        elif len(pos_list) == 1 and (pos_list[0].startswith('N') or
                                     pos_list[0].startswith('J') or
                                     pos_list[0] == 'FW' or
                                     pos_list[0] == 'XX') or \
                (len(pos_list) > 1 and self.valid_entity_tag.match(pos_str)):
            # It is not allowed to split a consecutive NNP
            # if it is a single token.
            if len(pos_list) == 1:
                if pos_list[0].startswith('NNP') and start > 0 \
                        and tokens[start - 1].pos.startswith('NNP'):
                    return False
                elif pos_list[-1].startswith('NNP') and end < len(tokens) \
                        and tokens[end].pos.startswith('NNP'):
                    return False
            return True
        return False

    # def is_entity_occurrence(self, tokens, start, end):
    #     '''
    #     Return true if the tokens marked by start and end indices
    #     are a valid entity occurrence.
    #     :param tokens:
    #     :param start:
    #     :param end:
    #     :return:
    #     '''
    #     # Concatenate POS-tags
    #     token_list = tokens[start:end]
    #     pos_list = [t.pos for t in token_list]
    #     pos_str = ''.join(pos_list)
    #     # Check if all tokens are in the ignore list.
    #     if all((t.lemma in self.ignore_lemmas for t in token_list)):
    #         return False
    #     # For length 1 only allows nouns
    #     elif len(pos_list) == 1 and pos_list[0].startswith('N') or \
    #                             len(pos_list) > 1 and \
    #                     self.valid_entity_tag.match(pos_str):
    #         # It is not allowed to split a consecutive NNP
    #         # if it is a single token.
    #         if len(pos_list) == 1:
    #             if pos_list[0].startswith('NNP') and start > 0 \
    #                     and tokens[start - 1].pos.startswith('NNP'):
    #                 return False
    #             elif pos_list[-1].startswith('NNP') and end < len(tokens) \
    #                     and tokens[end].pos.startswith('NNP'):
    #                 return False
    #         return True
    #     return False

    def identify_dates(self, tokens):
        '''
        Identify entities representing dates in the
        tokens.
        :param tokens:
        :return:
        '''
        # Very simplistic for now.
        identified_dates = []
        for t in tokens:
            if t.pos == 'CD':
                # A simple match for years.
                if re.match(self.year_re, t.text):
                    year = t.text
                    e = DateValue(year, get_value_for_year(year))
                    ie = IdentifiedEntity([t], e.name, e, perfect_match=True)
                    identified_dates.append(ie)
        return identified_dates

    def identify_entities_in_tokens(self, tokens, min_surface_score=0.3):
        '''
        Identify instances in the tokens.
        :param tokens: A list of string tokens.
        :return: A list of tuples (i, j, e, score) for an identified entity e,
                 at token index i (inclusive) to j (exclusive)
        '''

        n_tokens = len(tokens)
        logger.info("Starting entity identification.")
        start_time = time.time()
        # First find all candidates.
        identified_entities = []
        for start in range(n_tokens):
            for end in range(start + 1, n_tokens + 1):
                entity_tokens = tokens[start:end]
                if not self.is_entity_occurrence(tokens, start, end):
                    continue
                entity_str = ' '.join([t.text for t in entity_tokens])
                logger.debug(u"Checking if '{0}' is an entity.".format(entity_str))
                entities = self.surface_index.get_entities_for_surface(entity_str)
                # No suggestions.
                if len(entities) == 0:
                    continue
                for e, surface_score in entities:
                    # Ignore entities with low surface score.
                    if surface_score < min_surface_score:
                        continue
                    perfect_match = False
                    # Check if the main name of the entity exactly matches the text.
                    # I only use the label as surface, so the perfect match is always True
                    if self._text_matches_main_name(e, entity_str):
                        perfect_match = True
                    ie = IdentifiedEntity(tokens[start:end],
                                          e.name, e, e.score, surface_score,
                                          perfect_match)
                    # self.boost_entity_score(ie)
                    identified_entities.append(ie)
        # Turn Off the date identifier for now. TODO: take care of value identification later
        # identified_entities.extend(self.identify_dates(tokens))
        duration = (time.time() - start_time) * 1000
        identified_entities = self._filter_identical_entities(identified_entities)
        identified_entities = EntityLinker.prune_entities(identified_entities,
                                                          max_threshold=self.max_entities_per_tokens)
        # Sort by quality
        identified_entities = sorted(identified_entities, key=lambda x: (len(x.mention),
                                                                         x.surface_score),
                                     reverse=True)
        logging.info("Entity identification took %.2f ms. Identified %s entities." % (duration,
                                                                                      len(identified_entities)))
        return identified_entities



    def _filter_identical_entities(self, identified_entities):
        '''
        Some entities are identified twice, once with a prefix/suffix
          and once without.
        :param identified_entities:
        :return:
        '''
        entity_map = {}
        filtered_identifications = []
        for e in identified_entities:
            if e.entity not in entity_map:
                entity_map[e.entity] = []
            entity_map[e.entity].append(e)
        for entity, identifications in entity_map.items():
            if len(identifications) > 1:
                # A list of (token_set, score) for each identification.
                token_sets = [(set(i.mention), i.surface_score)
                              for i in identifications]
                # Remove identification if its tokens
                # are a subset of another identification
                # with higher surface_score
                while identifications:
                    ident = identifications.pop()
                    tokens = set(ident.mention)
                    score = ident.surface_score
                    if any([tokens.issubset(x) and score < s
                            for (x, s) in token_sets if x != tokens]):
                        continue
                    filtered_identifications.append(ident)
            else:
                filtered_identifications.append(identifications[0])
        return filtered_identifications

    @staticmethod
    def prune_entities(identified_entities, max_threshold=7):
        token_map = {}
        for e in identified_entities:
            tokens = tuple(e.mention)
            if tokens not in token_map:
                    token_map[tokens] = []
            token_map[tokens].append(e)
        remove_entities = set()
        for tokens, entities in token_map.items():
            if len(entities) > max_threshold:
                sorted_entities = sorted(entities, key=lambda x: x.surface_score, reverse=True)
                # Ignore the entity if it is not in the top candidates, except, when
                # it is a perfect match.
                #for e in sorted_entities[max_threshold:]:
                #    if not e.perfect_match or e.score <= 3:
                #        remove_entities.add(e)
                remove_entities.update(sorted_entities[max_threshold:])
        filtered_entities = [e for e in identified_entities if e not in remove_entities]
        return filtered_entities

    def boost_entity_score(self, entity):
        if entity.perfect_match:
            entity.score *= 60

    @staticmethod
    def create_consistent_identification_sets(identified_entities):
        logger.info("Computing consistent entity identification sets for %s entities." % len(identified_entities))
        # For each identified entity, the ones it overlaps with
        overlapping_sets = []
        for i, e in enumerate(identified_entities):
            overlapping = set()
            for j, other in enumerate(identified_entities):
                if i == j:
                    continue
                if any([t in other.mention for t in e.mention]):
                    overlapping.add(j)
            overlapping_sets.append((i, overlapping))
        maximal_sets = []
        logger.info(overlapping_sets)
        EntityLinker.get_maximal_sets(0, set(), overlapping_sets, maximal_sets)
        #logger.info((maximal_sets))
        result = {frozenset(x) for x in maximal_sets}
        consistent_sets = []
        for s in result:
            consistent_set = set()
            for e_index in s:
                consistent_set.add(identified_entities[e_index])
            consistent_sets.append(consistent_set)
        logger.info("Finished computing %s consistent entity identification sets." % len(consistent_sets))
        return consistent_sets

    @staticmethod
    def get_maximal_sets(i, maximal_set, overlapping_sets, maximal_sets):
        #logger.info("i: %s" % i)
        # if i == len(overlapping_sets):
        #     return
        maximal = True
        # Try to extend the maximal set
        # for j, (e, overlapping) in enumerate(overlapping_sets[i:]):
        for j, (e, overlapping) in enumerate(overlapping_sets):
            # The two do not overlap.
            if len(overlapping.intersection(maximal_set)) == 0 and not e in maximal_set:
                new_max_set = set(maximal_set)
                new_max_set.add(e)
                # maximal_set.add(e)
                EntityLinker.get_maximal_sets(i + 1, new_max_set,
                                               overlapping_sets, maximal_sets)
                maximal = False
        if maximal:  # This indicates no more entities is added, so it's a maximal set to be included
            maximal_sets.append(maximal_set)


"""
if __name__ == '__main__':
    # Using stanford CoreNLP with nltk: https://github.com/nltk/nltk/wiki/Stanford-CoreNLP-API-in-NLTK
    surface_index = surface_index_memory.EntitySurfaceIndexMemory(
        "data/entity_list_file_freebase_complete_all_mention", 
        "data/surface_map_file_freebase_complete_all_mention",
        "data/freebase_complete_all_mention")

    entity_linker=EntityLinker(surface_index, 7)

    parser = stanfordnlp.Pipeline()
    parse_result = parser("What team that won the 1957 Pequeña Copa del Mundo de Clubes championship did Trump play for?")
    # each token in tokens should has attributes of token (change to text), lemma, pos
    tokens = parse_result.sentences[0].words

    el_result = entity_linker.identify_entities_in_tokens(tokens)

    el_result = entity_linker.create_consistent_identification_sets(el_result)

    for el in el_result:
        print(el)
        entity_link_result = dict()
        surfacename = el.name
        # mention = " ".join([token.text for token in el.mention])
        mention = el.mention
        mid = el.entity.id
        score = el.surface_score
        popularity = el.score
        perfect_match = el.perfect_match
        print(surfacename, mention, mid, score, popularity, perfect_match)
"""
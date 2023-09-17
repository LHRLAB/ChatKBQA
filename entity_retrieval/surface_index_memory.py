import mmap
import logging
import os
import array
import marshal

# import globals
# from common.globals_args import fn_cwq_file
# from common.hand_files import write_set
from entity_retrieval import aqqu_entity_linker
from entity_retrieval.aqqu_util import normalize_entity_name
import collections

logger = logging.getLogger(__name__)
logging.basicConfig(level="INFO")

class EntitySurfaceIndexMemory(object):
    """A memory based index for finding entities.
    Remember to delete the old _mid_vocab and _surface_index if updated the file (or choose a different prefix)
    """

    def __init__(self, entity_list_file, surface_map_file, entity_index_prefix):
        self.entity_list_file = entity_list_file
        self.surface_map_file = surface_map_file

        # mid_vocabulary: {mid:offset}
        self.mid_vocabulary = self._get_entity_vocabulary(entity_index_prefix)
        # surface_indxe: 
        self.surface_index = self._get_surface_index(entity_index_prefix)

        self.entities_mm_f = open(entity_list_file, 'r')
        self.entities_mm = mmap.mmap(self.entities_mm_f.fileno(), 0,access=mmap.ACCESS_READ)
        logger.info("Done initializing surface index.")

    def _get_entity_vocabulary(self, index_prefix):
        """Return vocabulary by building a new or reading an existing one.

        :param index_prefix:
        :return:
        """
        vocab_file = index_prefix + "_mid_vocab"
        if os.path.isfile(vocab_file):
            """
            Mid vocabulary file already exists
            """
            logger.info("Loading entity vocabulary from disk.")
            vocabulary = marshal.load(open(vocab_file, 'rb'))
        else:
            """
            Mid vocabulary does not exist, build it.
            """
            vocabulary = self._build_entity_vocabulary()
            logger.info("Writing entity vocabulary to disk.")
            marshal.dump(vocabulary, open(vocab_file, 'wb'))
        return vocabulary

    def _get_surface_index(self, index_prefix):
        """Return surface index by building new or reading existing one.

        :param index_prefix:
        :return:
        """
        surface_index_file = index_prefix + "_surface_index"
        if os.path.isfile(surface_index_file):
            logger.info("Loading surfaces from disk.")
            surface_index = marshal.load(open(surface_index_file, 'rb'))
        else:
            surface_index = self._build_surface_index()
            logger.info("Writing entity surfaces to disk.")
            marshal.dump(surface_index, open(surface_index_file, 'wb'))
        return surface_index

    def _build_surface_index(self):
        """Build the surface index.

        Reads from the surface map on disk and creates a map from
        surface_form -> offset, score ....

        :return:
        """
        n_lines = 0
        surface_index = dict()
        num_not_found = 0
        with open(self.surface_map_file, 'r',encoding="utf-8") as f:
            for line in f:
                n_lines += 1
                if n_lines % 1000 == 0:
                    logger.info('Bulding surface-forms (%s/5996)' % (n_lines//10000))
                try:
                    cols = line.rstrip().split('\t')
                    surface_form = cols[0] # surface_form
                    # surface_form = normalize_entity_name(surface_form)
                    surface_form = normalize_entity_name(surface_form) # normalized entity name
                    score = float(cols[1]) # popularity score
                    mid = cols[2] # mid
                    entity_id = self.mid_vocabulary[mid] # offset
                    if not surface_form in surface_index:
                        surface_form_entries = array.array('d') # double (float with 8 bytes)
                        surface_index[surface_form] = surface_form_entries # {surface_form:[entity_id, score]}
                    surface_index[surface_form].append(entity_id) # entity_id
                    surface_index[surface_form].append(score) # score
                except KeyError:
                    num_not_found += 1
                    if num_not_found < 100:
                        logger.warn("Mid %s appears in surface map but "
                                    "not in entity list." % cols[2])
                    elif num_not_found == 100:
                        logger.warn("Suppressing further warnings about "
                                    "unfound mids.")
                if n_lines % 5000000 == 0:
                    logger.info('Stored %s surface-forms.' % n_lines)
        logger.warn("%s entity appearances in surface map w/o mapping to "
                    "entity list" % num_not_found)
        return surface_index

    def _build_entity_vocabulary(self):
        """Create mapping from MID to offset/ID.

        :return:
        """
        logger.info("Building entity mid vocabulary.")
        mid_vocab = dict() # {mid:offset}
        num_lines = 0
        # Remember the offset for each entity.
        with open(self.entity_list_file, 'r',encoding="utf-8") as f:
            # m=mmap.mmap(fileno, length[, flags[, prot[, access[, offset]]]])
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) # create a mmap object
            offset = mm.tell() # return the pointer
            line = mm.readline() # readline
            while line:
                num_lines += 1
                if num_lines % 5000000 == 0:
                    logger.info('Read %s lines' % num_lines)
                cols = line.decode().strip().split('\t')
                mid = cols[0] # mid
                mid_vocab[mid] = offset # offset
                offset = mm.tell()
                line = mm.readline()
        return mid_vocab

    def get_entity_for_mid(self, mid):
        """Returns the entity object for the MID or None if the MID is unknown.

        :param mid:
        :return:
        """
        try:
            offset = self.mid_vocabulary[mid]
            entity = self._read_entity_from_offset(int(offset))
            return entity
        except KeyError:
            logger.warn("Unknown entity mid: '%s'." % mid)
            return None

    def get_entities_for_surface(self, surface):
        """Return all entities for the surface form.
        :param surface:
        :return:
        """
        # I think we are going to make the mentions in our dataset case sensitive
        # surface = normalize_entity_name(surface)
        surface = normalize_entity_name(surface)
        try:
            # Only when read from an existing surface_index, bytestr is a byte string. If it's just created
            # in this call, then bytestr is an array
            bytestr = self.surface_index[surface] # bytestr
            if isinstance(bytestr, array.array):
                ids_array = bytestr
            else:
                ids_array = array.array('d')
                ids_array.frombytes(bytestr) # [offset1,surface_score1,...]
            result = []
            i = 0
            while i < len(ids_array) - 1:
                offset = ids_array[i]
                surface_score = ids_array[i + 1]
                entity = self._read_entity_from_offset(int(offset))
                # Check if the main name of the entity exactly matches the text.
                result.append((entity, surface_score))
                i += 2
            return result
        except KeyError:
            return []

    @staticmethod
    def _string_to_entity(line):
        """Instantiate entity from string representation.

        :param line:
        :return:
        """
        line = line.decode('utf-8')
        cols = line.strip().split('\t')
        mid = cols[0]
        name = cols[1]
        score = int(cols[2])
        aliases = cols[3:]
        return aqqu_entity_linker.KBEntity(name, mid, score, aliases)

    def _read_entity_from_offset(self, offset):
        """Read entity string representation from offset.

        :param offset:
        :return:
        """
        self.entities_mm.seek(offset)
        l = self.entities_mm.readline()
        return self._string_to_entity(l)

    # get second element of a list
    def get_indexrange_entity_el_pro_one_mention(self, mention, top_k=10):
        tuple_list = self.get_entities_for_surface(mention)
        if not tuple_list:
            return collections.OrderedDict()
        entities_dict = dict()
        for entity, surface_score in tuple_list:
            entities_dict[entity.id] = surface_score
        entities_tuple_list = sorted(entities_dict.items(), key=lambda d:d[1], reverse=True)
        result_entities_dict = collections.OrderedDict()
        for i, (entity_id, surface_score) in enumerate(entities_tuple_list):
            i += 1
            result_entities_dict[entity_id] = surface_score
            if i >= top_k:
                break
        return result_entities_dict

if __name__ == '__main__':
    # get_aqqu_mids(fn_cwq_file.entity_list_file,fn_cwq_file.surface_map_file,fn_cwq_file.aqqu_entity_contained)
    # def get_aqqu_mids(entity_file, surface_file, aqqu_entityall_file):
    #     mids = set()
    #     with open(entity_file, 'r', encoding="utf-8") as f:
    #         mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    #         line = mm.readline()
    #         while line:
    #             cols = line.decode().strip().split('\t')
    #             mid = cols[0]
    #             mids.add(mid)
    #             line = mm.readline()
    #     with open(surface_file, 'r', encoding="utf-8") as f:
    #         for line in f:
    #             cols = line.rstrip().split('\t')
    #             mid = cols[2]
    #             mids.add(mid)
    #     write_set(mids, aqqu_entityall_file)
    # main()
    # logging.basicConfig(
    #     format='%(asctime)s : %(levelname)s : %(module)s : %(message)s', level=logging.INFO)
    # for entity, surface_score in (
    #     entity_linking_aqqu_index.get_entities_for_surface("taylor lautner")):  # Albert Einstein
    #     print(entity.id, surface_score)
    # for entity, surface_score in (entity_linking_aqqu_index.get_entities_for_surface('Agusan del Sur')):
    #     print(entity.id, surface_score)
    # mention_to_entities('Agusan del Sur', top_k=10)
    # print(mention_to_entities('2010 Formula One World Championship', top_k=10))
    # print(mention_to_entities('Theresa Russo', top_k=10))
    # tuple_list.sort(key=takeSecond)
    # print('**************************')
    # for entity,surface_score in tuple_list:
    #     print(entity.id, surface_score)
    pass

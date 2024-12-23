from unitok.tokenizer import BertTokenizer
from unitok.utils.handler import JsonHandler


class JsonEntitiesTokenizer(BertTokenizer):
    def __call__(self, obj):
        entity_list = JsonHandler.loads(obj)
        if not entity_list:
            return []
        entity_string = ' '.join([entity['Label'] for entity in entity_list])
        return super().__call__(entity_string)

from unitok.tokenizer import BaseTokenizer


class EntityTokenizer(BaseTokenizer):
    return_list = False
    name = 'entity'
    param_list = []


class EntitiesTokenizer(BaseTokenizer):
    return_list = True
    name = 'entities'
    param_list = []

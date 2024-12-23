from unitok.utils.verbose import warning
from unitok.utils import Space


class Instance:
    def __init__(self, compulsory_space=False):
        self._space_instances = dict()
        self._compulsory_space = compulsory_space

    def current(self):
        space = Space.get_space()
        if space is None:
            if self._compulsory_space:
                raise ValueError('Required UniTok context, please use `with UniTok() as ut:`')
            warning('It is recommended to declare tokenizers and vocabularies in a UniTok context, using `with UniTok() as ut:`')
        if space not in self._space_instances:
            self._space_instances[space] = dict()
        return self._space_instances[space]

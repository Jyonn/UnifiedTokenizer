from typing import Hashable

from unitok import warning
from unitok.tokenizer import BaseTokenizer


class CachableTokenizer(BaseTokenizer):
    def __init__(self, use_cache=False, **kwargs):
        super().__init__(**kwargs)

        if not self.return_list and use_cache:
            warning(f'Only the tokenizer that return_list=True may need cache, use_cache of {self.get_classname()} will be set to False')
            use_cache = False
        self.use_cache = use_cache
        self._cache = dict()

    def __call__(self, objs):
        if self.use_cache and isinstance(objs, Hashable):
            if objs in self._cache:
                return self._cache[objs]
            value = super().__call__(objs)
            self._cache[objs] = value
            return value

        return super().__call__(objs)

from unitok.tokenizer import BaseTokenizer
from unitok.utils.index_set.index_set import IndexSet


class TokenizerSet(IndexSet[BaseTokenizer]):
    @staticmethod
    def _get_key(obj):
        return obj.get_tokenizer_id()

    def merge(self, other: IndexSet[BaseTokenizer], **kwargs):
        for obj in other:  # type: BaseTokenizer
            if obj in self:
                continue
            if self.has(self._get_key(obj)):
                current = self.get(self._get_key(obj))
                if not current.vocab.equals(obj.vocab):
                    raise ValueError(f'Conflict vocabulary: {current} and {obj}')
            else:
                self.add(obj)

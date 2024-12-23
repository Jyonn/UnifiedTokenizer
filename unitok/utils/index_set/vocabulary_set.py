from unitok.utils.index_set.index_set import IndexSet
from unitok.vocabulary import Vocabulary


class VocabularySet(IndexSet[Vocabulary]):
    @staticmethod
    def _get_key(obj: Vocabulary):
        return obj.name

    def merge(self, other: IndexSet[Vocabulary], **kwargs):
        for obj in other:  # type: Vocabulary
            if obj in self:
                continue
            if self.has(self._get_key(obj)):
                current = self.get(self._get_key(obj))
                if len(current) != len(obj):
                    raise ValueError(f'Conflict vocabulary size: {current} and {obj}')
            else:
                self.add(obj)

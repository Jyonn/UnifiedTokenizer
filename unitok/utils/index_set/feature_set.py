from unitok.feature import Feature
from unitok.tokenizer.union_tokenizer import UnionTokenizer
from unitok.utils.index_set.index_set import IndexSet


class FeatureSet(IndexSet[Feature]):
    @staticmethod
    def _get_key(obj: Feature):
        return obj.name

    def next_order(self):
        return max([feature.order for feature in self]) + 1

    def merge(self, other: IndexSet[Feature], **kwargs):
        key_feature = kwargs.get('key_feature')

        next_order = self.next_order()
        for feature in other:
            if feature is key_feature:
                continue
            if not feature.is_processed:
                raise ValueError(f'Merge unprocessed feature: {feature}')
            if self.has(self._get_key(feature)):
                raise ValueError(f'Conflict feature name: {feature.name}')
            # self.add(feature.clone(order=next_order, tokenizer=UnionTokenizer(feature.tokenizer)))
            self.add(feature.clone(order=next_order))

from unitok import Vocab
from unitok.tokenizer.base_tokenizer import BaseTokenizer

from unitok.feature import Feature
from unitok.meta import Meta


class Selector:
    def __init__(self, meta: Meta, *selectors):
        self.meta = meta
        self.selectors = selectors

    def _auto_select(self, sample, selector):
        if isinstance(selector, str):
            return {selector}
        if isinstance(selector, Feature):
            return {selector.name}
        if isinstance(selector, BaseTokenizer):
            return {name for name in sample if self.meta.features[name].tokenizer is selector}
        if isinstance(selector, Vocab):
            return {name for name in sample if self.meta.features[name].tokenizer.vocab.equals(selector)}
        raise ValueError(f'Unrecognized selector: {selector}')

    def __call__(self, sample: dict):
        name_set = set()
        for s in self.selectors:
            ns = self._auto_select(sample, s)
            name_set.update(ns)
        return {name: sample[name] for name in name_set}

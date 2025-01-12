import nltk

from unitok.vocabulary import VocabHub
from unitok.tokenizer import BaseTokenizer


class GloVeTokenizer(BaseTokenizer):
    return_list = True
    param_list = ['language']

    def __init__(self, vocab, language='english', **kwargs):
        if isinstance(vocab, str) and not VocabHub.has(vocab):
            raise ValueError('GloVeTokenizer requires a pre-filled Vocab object that stores valid tokens')

        super().__init__(vocab=vocab, **kwargs)

        self.language = language

    def __call__(self, obj):
        objs = nltk.tokenize.word_tokenize(obj.lower(), language=self.language)
        return [self.vocab[o] for o in objs if o in self.vocab]

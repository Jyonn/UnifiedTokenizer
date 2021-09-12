from tqdm import tqdm

from .analysis.analysis import Analysis
from .tok.tokenizer import Tokenizer, ListTokenizer


class Column:
    def __init__(self, name, tokenizer: Tokenizer):
        self.name = name
        self.tokenizer = tokenizer
        self.tok = tokenizer.tok
        self.data = []
        self.analysis = Analysis()

    def tokenize(self, objs):
        self.data = []
        for obj in tqdm(objs):
            ids = self.tokenizer.tokenize(obj)
            self.data.append(ids)

    def analyse(self, objs):
        if isinstance(self.tokenizer, ListTokenizer):
            self.analysis.clean()
            for obj in tqdm(objs):
                self.analysis.push(len(self.tok(obj)))
            self.analysis.analyse()
        else:
            print('[NOT ListTokenizer]')
            for obj in tqdm(objs):
                self.tok(obj)

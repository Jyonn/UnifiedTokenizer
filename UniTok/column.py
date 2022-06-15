from tqdm import tqdm

from .global_setting import Global
from .analysis.length_analysis import LengthAnalysis
from .tok import IdTok
from .tok.tokenizer import Tokenizer, ListTokenizer


class Column:
    def __init__(self, name, tokenizer: Tokenizer):
        self.name = name
        self.tokenizer = tokenizer
        self.tok = tokenizer.tok
        self.data = []
        self.length_analysis = LengthAnalysis()

    def tokenize(self, objs):
        self.data = []
        for obj in tqdm(objs, disable=Global.is_silence()):
            ids = self.tokenizer.tokenize(obj)
            self.data.append(ids)

    def analyse(self, objs):
        if isinstance(self.tokenizer, ListTokenizer):
            self.length_analysis.clean()
            for obj in tqdm(objs, disable=Global.is_silence()):
                ids = self.tok(obj)
                self.tok.vocab.frequency_count(*ids)
                self.length_analysis.push(len(ids))
            self.length_analysis.analyse()
        else:
            print('[NOT ListTokenizer]')
            for obj in tqdm(objs, disable=Global.is_silence()):
                self.tok.vocab.frequency_count(self.tok(obj))


class IndexColumn(Column):
    def __init__(self, name='index'):
        super().__init__(name, tokenizer=IdTok(name=name).as_sing())

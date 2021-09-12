from .tok import BaseTok


class IdTok(BaseTok):
    def __init__(self, name):
        super(IdTok, self).__init__(name=name)

    def t(self, obj):
        return self.vocab.append(obj)

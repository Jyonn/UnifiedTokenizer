from .tok import BaseTok


class IdTok(BaseTok):
    def t(self, obj):
        if self.pre_handler:
            obj = self.pre_handler(obj)
        return self.vocab.append(str(obj))

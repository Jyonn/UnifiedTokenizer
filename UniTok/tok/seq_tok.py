from .tok import BaseTok


class SeqTok(BaseTok):
    """
    Sequence Tokenizer

    Example:
        >>> from UniTok.tok.seq_tok import SeqTok
        >>> tok = SeqTok()
        >>> tok(['Hello', 'world'])  # [0, 1]
        >>> tok(['Debug', 'the', 'world'])  # [2, 3, 1]
    """
    return_list = True

    def t(self, obj: list):
        return [self.insert(o) for o in obj]

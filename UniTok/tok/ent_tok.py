from .tok import BaseTok


class EntTok(BaseTok):
    """
    Entity tokenizer

    Args:
        ...: other arguments of the BaseTok

    Example:
        >>> from UniTok.tok.ent_tok import EntTok
        >>> tok = EntTok(name='entity')
        >>> tok('JJ Lin')  # 0
        >>> tok('Jay Chou')  # 1
    """
    return_list = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pre_handler = str

    def t(self, obj):
        return self.insert(obj)

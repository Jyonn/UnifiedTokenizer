from smartify import E


@E.register(id_processor=E.idp_cls_prefix())
class VocabDepotError:
    ConflictName = E('Conflict name: {}')
    NotFound = E('Not found vocab {}')


class VocabDepot:
    def __init__(self):
        self.depot = {}
        self.col_map = {}

    def append(self, col):
        vocab = col.tok.vocab
        if vocab.name is not None:
            if vocab.name in self.depot and self.depot[vocab.name] != vocab:
                raise VocabDepotError.ConflictName(vocab.name)
            self.depot[vocab.name] = vocab
        if vocab.name in self.col_map:
            self.col_map[vocab.name].append(col.name)
        else:
            self.col_map[vocab.name] = [col.name]

    def get_vocab(self, name):
        if name in self.depot:
            return self.depot[name]
        raise VocabDepotError.NotFound(name)

class Symbol(str):
    def __init__(self, name: str = None):
        self.name = name

    def __str__(self):
        return f'<{self.name}>'


class Symbols:
    idx = Symbol('index')
    tokenizer = Symbol('tokenizer')

    # union type

    soft = Symbol('soft')
    hard = Symbol('hard')

    # status

    initialized = Symbol('initialized')
    tokenized = Symbol('tokenized')
    organized = Symbol('organized')


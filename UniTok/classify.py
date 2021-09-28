class Classify:
    def dict(self, *args):
        if args:
            dict_ = dict()
            for k in args:
                dict_[k] = self.d.get(k)
            return dict_
        return self.d

    def iter_list(self, l: list):
        new_l = []
        for v in l:
            if isinstance(v, dict):
                v = Classify(v)
            elif isinstance(v, list):
                v = self.iter_list(v)
            new_l.append(v)
        return new_l

    def iter_dict(self, d: dict):
        for k in d:
            if isinstance(d[k], dict):
                d[k] = Classify(d[k])
            elif isinstance(d[k], list):
                d[k] = self.iter_list(d[k])
        return d

    def __init__(self, d: dict):
        self.d = self.iter_dict(d)

    def __getattr__(self, item):
        return self.d[item]

    def __setattr__(self, key, value):
        if key == 'd':
            object.__setattr__(self, key, value)
        else:
            if isinstance(value, dict):
                value = Classify(value)
            self.d[key] = value

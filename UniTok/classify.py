class Classify:
    def dict_list(self, l: list):
        new_l = []
        for v in l:
            if isinstance(v, Classify):
                v = v.dict()
            elif isinstance(v, list):
                v = self.dict_list(v)
            new_l.append(v)
        return new_l

    def dict(self):
        d = dict()
        for k in self.d:
            if isinstance(self.d[k], list):
                d[k] = self.dict_list(self.d[k])
            elif isinstance(self.d[k], Classify):
                d[k] = self.d[k].dict()
            else:
                d[k] = self.d[k]
        return d

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
        if key in ['d']:
            object.__setattr__(self, key, value)
        else:
            if isinstance(value, dict):
                value = Classify(value)
            self.d[key] = value


if __name__ == '__main__':
    c = Classify(dict(a=1, b=dict(x='a', y=[3, [4, dict(y='l')], dict(z=1)])))
    print(c.b.y[1][1].y)

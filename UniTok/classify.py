import copy


class Classify:
    def dict(self):
        return self.__d

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
        self.__d = copy.deepcopy(d)
        self.d = self.iter_dict(d)

    def __getattr__(self, item):
        return self.d[item]

    def __setattr__(self, key, value):
        if key in ['d', '_Classify__d']:
            object.__setattr__(self, key, value)
        else:
            if isinstance(value, dict):
                value = Classify(value)
            self.d[key] = value


if __name__ == '__main__':
    print(Classify(dict(a=1, b=dict(x='a', y=[3,4]))).dict())

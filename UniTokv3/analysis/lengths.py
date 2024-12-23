from .plot import Plot


class Lengths:
    def __init__(self):
        self.lengths = []
        self.max_length = -1

    def push(self, length):
        self.lengths.append(length)
        if length > self.max_length:
            self.max_length = length

    def summarize(self):
        max_length = max(self.lengths)
        min_length = min(self.lengths)
        avg_length = '%.4f' % (sum(self.lengths) * 1.0 / len(self.lengths))
        print('[ MIN:', min_length, ']')
        print('[ MAX:', max_length, ']')
        print('[ AVG:', avg_length, ']')
        Plot(self.lengths, groups=100).plot()

    def clean(self):
        self.lengths = []
        self.max_length = -1


if __name__ == '__main__':
    import random

    analysis = Lengths()
    analysis.clean()
    for _ in range(67923):
        analysis.push(random.randint(0, 99999) % 100)
    analysis.summarize()

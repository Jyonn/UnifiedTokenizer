import collections
import math

import termplot


class Plot:
    def __init__(self,
                 array: list,
                 groups: int = 50,
                 height: int = 10,
                 ):
        self.array = array
        self.min = min(self.array)
        self.max = max(self.array)

        self.height = height
        self.groups = groups
        self.range = self.max - self.min + 1
        if self.max - self.min + 1 < self.groups:
            self.groups = self.range
        self.interval = math.ceil(self.range / self.groups)
        self.groups = math.ceil(self.range / self.interval)

        self.counts = [0] * self.groups
        for n in self.array:
            i = (n - self.min) // self.interval
            self.counts[i] += 1

    def plot(self):
        print('[ X-INT:', self.interval, ']')
        print('[ Y-INT:', max(self.counts) // self.height, ']')
        termplot.plot(
            x=self.counts,
            plot_char='|',
            plot_height=self.height,
        )
        print('-' * self.groups)


if __name__ == '__main__':
    a = [1, 2, 3, 4, -5, 5, -4, -1, 0, -10, -4, -2, 3, 5, 8, 10, 12, 10, 8, 7, 6, 5, 4, 3, 2, 1]
    print(collections.Counter(a))
    Plot(a).plot()

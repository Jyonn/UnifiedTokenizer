import numpy as np


class Counter:
    def __init__(self):
        self._activate = False
        self._count = dict()

    def activate(self):
        self._activate = True
        return self

    def deactivate(self):
        self._activate = False
        return self

    def initialize(self):
        self._count = dict()
        return self

    def __call__(self, indices: list):
        if not self._activate:
            return self

        if isinstance(indices, int):
            indices = [indices]

        for index in indices:
            assert isinstance(index, int)
            self._count[index] = self._count.get(index, 0) + 1

    def trim(self, min_count):
        """
        trim vocab by min frequency
        :return: trimmed tokens
        """
        valid_indices = []
        for index in self._count:
            if self._count[index] >= min_count:
                valid_indices.append(index)
        return valid_indices

    def summarize(self, base=10):
        """
        summarize vocab by frequency
        :param base: display base, default 10
        :return: counts of clustered bounds, e.g., { (1, 2): 100, (2, 3): 200, ... }
        """
        max_count = max(self._count.values())
        digits_max = base
        while digits_max < max_count:
            digits_max = digits_max * base

        bounds = []
        while digits_max >= base:
            digits_min = digits_max // base
            left_bound = (np.arange(base - 1)[::-1] + 1) * digits_min
            right_bound = left_bound + digits_min
            bounds.extend(zip(left_bound, right_bound))
            digits_max = digits_min
        bounds.reverse()  # [(1, 2), ..., (9, 10), (10, 20), ..., (90, 100), (100, 200), ..., ...]

        counts = dict()
        for bound in bounds:
            counts[bound] = 0

        for index in self._count:
            count = self._count[index]
            # binary search
            left, right = 0, len(bounds) - 1
            while left <= right:
                mid = (left + right) // 2
                if bounds[mid][0] <= count < bounds[mid][1]:
                    counts[bounds[mid]] += 1
                    break
                elif count < bounds[mid][0]:
                    right = mid - 1
                else:
                    left = mid + 1

        for bound in bounds:
            if not counts[bound]:
                del counts[bound]

        return counts

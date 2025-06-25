import warnings
from unitok.feature import Feature, FeatureHub


class Job(Feature):
    def __init__(self, **kwargs):
        warnings.warn(f'`Job` class is deprecated, use `Feature`.', DeprecationWarning, stacklevel=2)
        super().__init__(**kwargs)


JobHub = FeatureHub

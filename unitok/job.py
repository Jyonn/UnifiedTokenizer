import warnings
from unitok.feature import Feature, FeatureHub


class Job(Feature):
    def __init__(self, **kwargs):
        warnings.deprecated(f'Job is deprecated, use Feature instead.')
        super().__init__(**kwargs)


JobHub = FeatureHub

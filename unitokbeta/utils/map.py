class Map(dict):
    def __call__(self, *args, **kwargs):
        return self.__getitem__(*args, **kwargs)

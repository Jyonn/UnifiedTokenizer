class Verbose:
    DEBUG = 3
    INFO = 2
    WARNING = 1
    ERROR = 0

    level = INFO

    @classmethod
    def set_level(cls, level):
        cls.level = level

    @classmethod
    def get_printer(cls, level):
        def printer(*args, **kwargs):
            if level <= cls.level:
                print(*args, **kwargs)
        return printer

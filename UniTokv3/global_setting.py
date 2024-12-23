class Global:
    _silence = False

    @classmethod
    def set_silence(cls, silence):
        cls._silence = silence

    @classmethod
    def is_silence(cls):
        return cls._silence

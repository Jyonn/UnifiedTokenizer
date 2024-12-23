class Space:
    """
    UniTok allows multiple instances to be created, but the "with" statement can only be used with one instance.
    """

    _active_instance = None

    @classmethod
    def set(cls, obj):
        """
        Lock the unitok instance as the current active instance
        """
        if cls._active_instance is not None:
            raise ValueError(f'Space is already locked to {cls._active_instance}')
        cls._active_instance = obj

    @classmethod
    def unset(cls):
        """
        Unlock the current active instance
        """
        cls._active_instance = None

    @classmethod
    def get_space(cls):
        """
        Get the current active instance
        """
        return cls._active_instance

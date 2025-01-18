class Space:
    """
    UniTok allows multiple instances to be created, but the "with" statement can only be used with one instance.
    """

    _stack = []

    @classmethod
    def push(cls, obj):
        """
        Lock the unitok instance as the current active instance
        """
        cls._stack.append(obj)
        # if cls._active_instance is not None:
        #     raise ValueError(f'Space is already locked to {cls._active_instance}')

    @classmethod
    def pop(cls, obj):
        """
        Unlock the current active instance
        """
        # cls._active_instance = None
        if not cls._stack:
            raise ValueError('Space stack is empty')
        if cls._stack[-1] != obj:
            raise ValueError('Space stack is not in order')
        cls._stack.pop()

    @classmethod
    def get_space(cls):
        """
        Get the current active instance
        """
        # return cls._active_space
        return cls._stack[-1] if cls._stack else None
